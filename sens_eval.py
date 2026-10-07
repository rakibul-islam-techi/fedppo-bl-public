"""Evaluation-only sensitivity checks on trained checkpoints (no retraining).

  sic   Imperfect SIC. Devices are decoded in the order set by the
        action. A device decoded later sees a residual xi * P_j * g_j from every
        earlier device that was decoded, and, under propagation, the full power of
        every earlier device whose task was dropped. Within a device, the private
        stream also sees xi times the common-stream power. With xi = 0 and no
        propagation the outcomes are identical, draw for draw, to sim_core.
  zeta  Channel correlation per interval. Under Clarke's model,
        zeta = J0(2 pi f_D T_c). zeta = 0.9 at T_c = 5 ms gives f_D = 20.4 Hz,
        so T_c = 2 ms and 1 ms correspond to zeta = 0.9837 and 0.9959.

    set FEDPPO_T_HARQ_MS=0.125
    python sens_eval.py sic  --ckpt ckpt --out results/sic_sensitivity.json
    python sens_eval.py zeta --ckpt ckpt --out results/zeta_sensitivity.json
"""
from __future__ import annotations
import argparse, dataclasses, json, os
from pathlib import Path
import numpy as np


def _make_env_cls():
    from sim_core import AerialEdgeEnv, block_error, stream_bits

    class ImperfectSICEnv(AerialEdgeEnv):
        xi = 0.0
        propagate = False

        def step(self, action_vec):
            cfg = self.cfg
            rho, q, order, f_alloc, n_block = self._decode_action(action_vec)
            n = self.n_dev
            rx = self.gain * self.Pk
            L1, L2 = stream_bits(self.Lk, rho)
            R1, R2 = L1 / n_block, L2 / n_block
            U = self.rng_dec.random((n, 2, cfg.n_harq))
            resid = np.zeros(n)
            eps1 = np.zeros(n); eps2 = np.zeros(n)
            A = np.zeros((n, 2), dtype=int); sdrop = np.zeros((n, 2), dtype=bool)
            for k in np.argsort(order):
                later = rx[order > order[k]].sum()
                earlier = resid[order < order[k]].sum()
                I = later + earlier + self.noise
                g1 = rho[k] * rx[k] / ((1.0 - rho[k]) * rx[k] + I)
                g2 = (1.0 - rho[k]) * rx[k] / (self.xi * rho[k] * rx[k] + I)
                e = np.array([block_error(np.array([g1]), np.array([R1[k]]), np.array([n_block[k]]))[0],
                              block_error(np.array([g2]), np.array([R2[k]]), np.array([n_block[k]]))[0]])
                eps1[k], eps2[k] = e
                ok = U[k] < (1.0 - e)[:, None]
                first = np.where(ok.any(axis=-1), ok.argmax(axis=-1) + 1, cfg.n_harq + 1)
                A[k] = np.minimum(first, cfg.n_harq); sdrop[k] = first > cfg.n_harq
                dk = bool(sdrop[k].any())
                resid[k] = rx[k] if (self.propagate and dk) else self.xi * rx[k]
            W = cfg.bandwidth_hz
            t_tx = np.maximum(L1 / (W * R1) + (A[:, 0] - 1) * cfg.t_harq_s,
                              L2 / (W * R2) + (A[:, 1] - 1) * cfg.t_harq_s)
            t_comp = self.Lk * self.ck / f_alloc
            dropped = sdrop.any(axis=1)
            T = np.where(dropped, t_tx, t_tx + t_comp)
            dl = cfg.deadline_s
            lat = np.minimum(T, cfg.latency_cap_mult * dl)
            violated = dropped | (T > dl)
            reward = -float(lat.mean() / dl + cfg.kappa_violation * violated.mean())
            self._recent_lat.extend(lat.tolist()); self._recent_lat = self._recent_lat[-256:]
            self._recent_drop = float(dropped.mean()); self._last_lat = lat.copy()
            self.gain = self.large_scale * self.fading.step(self.rng)
            info = dict(latencies=lat, dropped=dropped, violated=violated, n_block=n_block,
                        t_tx=t_tx, t_comp=t_comp, sic_order=order,
                        mean_eps=float(np.mean(1.0 - (1.0 - eps1) * (1.0 - eps2))))
            return self._state(), reward, False, info

    return ImperfectSICEnv


def run_eval(policy_fn, env, cfg, n_episodes):
    lat, raw, drop, vio, nb = [], [], [], [], []
    for _ in range(n_episodes):
        s = env.reset(); df, _ = env.device_features()
        for _ in range(cfg.steps_per_episode):
            a = policy_fn(env, s, df)
            s, _, _, info = env.step(a); df, _ = env.device_features()
            lat.append(info["latencies"]); drop.append(info["dropped"]); vio.append(info["violated"])
            raw.append(np.where(info["dropped"], info["t_tx"], info["t_tx"] + info["t_comp"]))
            nb.append(info["n_block"])
    raw = np.concatenate(raw) * 1e3; drop = np.concatenate(drop); vio = np.concatenate(vio)
    deliv = raw[~drop]
    return dict(viol_total=100.0 * float(vio.mean()), drop_rate=100.0 * float(drop.mean()),
                late_rate=100.0 * float(np.mean(~drop & (raw > 1.0))),
                over_1ms=100.0 * float(np.mean(raw > 1.0)), over_2ms=100.0 * float(np.mean(raw > 2.0)),
                p99_deliv=float(np.percentile(deliv, 99)) if deliv.size else float("nan"),
                median_n=float(np.median(np.concatenate(nb))))


def _gain_order(env, cfg, a, descending=True):
    a = np.array(a, dtype=float, copy=True)
    D, n = cfg.max_devices, env.n_dev
    lg = np.log10(env.gain[:n] + 1e-30)
    a[D:2 * D][:n] = -(lg - lg.mean()) if descending else (lg - lg.mean())
    return a


def _policies(cfg, ckpt, seed):
    import torch
    from rl import CVaRPPOAgent, heuristic_action, heuristic_adaptive_n_action
    out = {}
    for tag, over in (("FedPPO-QR", {}), ("FedPPO-Mean", dict(n_quantiles=1, lambda_cvar=0.0))):
        p = Path(ckpt) / f"agent_{tag}_{seed}.pt"
        if not p.exists():
            continue
        c = dataclasses.replace(cfg, **over)
        ag = CVaRPPOAgent(c, seed)
        ag.actor.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["actor"])
        out[tag] = (lambda e, s, df, ag=ag: ag.act_greedy(s, df))
        if tag == "FedPPO-QR":
            out["FedPPO-QR, descending-gain order"] = (
                lambda e, s, df, ag=ag: _gain_order(e, cfg, ag.act_greedy(s, df), True))
            out["FedPPO-QR, ascending-gain order"] = (
                lambda e, s, df, ag=ag: _gain_order(e, cfg, ag.act_greedy(s, df), False))
    out["Static heuristic"] = lambda e, s, df: heuristic_action(e, cfg)
    out["Adaptive heuristic"] = lambda e, s, df: heuristic_adaptive_n_action(e, cfg, 1e-2)
    return out


def _job(job):
    kind, cond, seed, ckpt, episodes = job
    import torch
    torch.set_num_threads(1)
    from config import full_config
    from sim_core import AerialEdgeEnv
    cfg = full_config()
    res = {}
    if kind == "sic":
        Env = _make_env_cls()
        xi, prop = cond
        for name, fn in _policies(cfg, ckpt, seed).items():
            if name == "FedPPO-Mean":
                continue
            env = Env(cfg, 0, 10_000 + 100 + seed)
            env.xi, env.propagate = xi, prop
            res[name] = run_eval(fn, env, cfg, episodes)
    else:
        c = dataclasses.replace(cfg, channel_rho=float(cond))
        for name, fn in _policies(c, ckpt, seed).items():
            if "order" in name:
                continue
            env = AerialEdgeEnv(c, 0, 10_000 + 100 + seed)
            res[name] = run_eval(fn, env, c, episodes)
    return kind, cond, seed, res


def _summary(res, conds, label):
    from scipy import stats
    names = list(next(iter(res[conds[0]].values())).keys())
    ref = conds[0]
    print(f"\n{'policy':36s}{'condition':>18s}{'v_tot %':>10s}{'drop %':>9s}{'late %':>8s}"
          f"{'>2 ms %':>9s}{'d vs ref':>10s}{'95% CI':>18s}{'p':>8s}")
    for nm in names:
        for c in conds:
            seeds = sorted(res[c])
            v = np.array([res[c][s][nm]["viol_total"] for s in seeds])
            d = v - np.array([res[ref][s][nm]["viol_total"] for s in seeds])
            m = lambda k: np.mean([res[c][s][nm][k] for s in seeds])
            if c == ref or d.std(ddof=1) == 0:
                extra = ""
            else:
                se = d.std(ddof=1) / np.sqrt(d.size); tc = stats.t.ppf(0.975, d.size - 1)
                p = 2 * stats.t.sf(abs(d.mean() / se), d.size - 1)
                extra = f"{d.mean():+10.2f}  [{d.mean() - tc * se:+6.2f},{d.mean() + tc * se:+6.2f}]{p:8.3f}"
            print(f"{nm:36s}{label(c):>18s}{v.mean():10.2f}{m('drop_rate'):9.2f}{m('late_rate'):8.2f}"
                  f"{m('over_2ms'):9.3f}{extra}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["sic", "zeta"])
    ap.add_argument("--ckpt", default="ckpt")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--xi", type=float, nargs="+", default=[0.0, 0.001, 0.01, 0.1])
    ap.add_argument("--zeta", type=float, nargs="+", default=[0.9, 0.9837, 0.9959])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.kind == "sic":
        conds = [(0.0, False)] + [(x, True) for x in a.xi]
        label = lambda c: "ideal" if not c[1] else f"prop, xi={c[0]:g}"
    else:
        conds = list(a.zeta)
        label = lambda c: f"zeta={c:g}"
    jobs = [(a.kind, c, sd, a.ckpt, a.episodes) for c in conds for sd in range(a.seeds)]
    res = {c: {} for c in conds}
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for k, c, sd, r in pool.imap_unordered(_job, jobs):
                res[c][sd] = r
    else:
        for j in jobs:
            k, c, sd, r = _job(j); res[c][sd] = r
    _summary(res, conds, label)
    out = a.out or f"results/{a.kind}_sensitivity.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({label(c): {str(s): res[c][s] for s in sorted(res[c])} for c in conds},
              open(out, "w"), indent=1)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
