from __future__ import annotations
import argparse, dataclasses, json, sys
from pathlib import Path
import numpy as np

def cvar_ru(x, alpha=0.95):
    """Empirical CVaR as in Problem (14): mean of the worst (1 - alpha) share of
    outcomes by probability mass. Ties at the alpha-quantile are split, so the
    tail always holds exactly (1 - alpha) of the samples."""
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    k = n * (1.0 - alpha)
    m = int(np.floor(k))
    top = x[n - m:].sum() if m > 0 else 0.0
    frac = k - m
    nxt = x[n - m - 1] if frac > 0 else 0.0
    return float((top + frac * nxt) / k)



def evaluate(policy_fn, cfg, env_seed, n_episodes):
    from sim_core import AerialEdgeEnv
    env = AerialEdgeEnv(cfg, 0, 10_000 + 100 + env_seed)
    lat, drop, vio, eps, nb, raw = [], [], [], [], [], []
    for _ in range(n_episodes):
        s = env.reset(); df, _ = env.device_features()
        for _ in range(cfg.steps_per_episode):
            a = policy_fn(env, s, df)
            s, _, _, info = env.step(a); df, _ = env.device_features()
            lat.append(info["latencies"]); drop.append(info["dropped"])
            vio.append(info["violated"]); eps.append(info["mean_eps"])
            nb.append(info["n_block"])
            raw.append(np.where(info["dropped"], info["t_tx"], info["t_tx"] + info["t_comp"]))
    lat = np.concatenate(lat) * 1e3
    raw = np.concatenate(raw) * 1e3
    drop = np.concatenate(drop)
    vio = np.concatenate(vio)
    nb = np.concatenate(nb)
    deliv = raw[~drop]

    def st(x):
        if x.size == 0:
            return dict(p99=float("nan"), p999=float("nan"), cvar=float("nan"),
                        mean=float("nan"))
        return dict(p99=float(np.percentile(x, 99)), p999=float(np.percentile(x, 99.9)),
                    cvar=cvar_ru(x, 0.95), mean=float(x.mean()))
    return dict(
        n_tasks=int(lat.size),
        viol_total=100.0 * float(vio.mean()),
        viol_reported=100.0 * float(np.mean(lat > 1.0)),
        drop_rate=100.0 * float(np.mean(drop)),
        cap_rate=100.0 * float(np.mean(raw > 1e3 * cfg.latency_cap_mult * cfg.deadline_s)),
        viol_delivered=100.0 * float(np.mean(deliv > 1.0)) if deliv.size else float("nan"),
        median_eps=float(np.median(eps)),
        median_n=float(np.median(nb)),
        n_iqr=[float(np.percentile(nb, 25)), float(np.percentile(nb, 75))],
        delivered=st(deliv), all_tasks=st(lat),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--out", default="main_comparison.json")
    a = ap.parse_args()

    import torch
    from config import full_config, describe
    from rl import CVaRPPOAgent, heuristic_action, heuristic_adaptive_n_action, heuristic_fixed_n_action

    base = full_config()
    a.episodes = a.episodes or base.eval_episodes
    a.seeds = a.seeds or base.seeds
    base = dataclasses.replace(base, eval_episodes=a.episodes)
    print(describe(base), flush=True)
    ck = Path(a.ckpt)

    ARMS = {
        "FedPPO-QR":      {},
        "FedPPO-Mean":    dict(n_quantiles=1, lambda_cvar=0.0),
        "Centralized-QR": {},
    }
    out = {}

    from rl import DDQNAgent
    rows = []
    for sd in range(a.seeds):
        p = ck / f"agent_FedAvg-DDQN_{sd}.pt"
        if not p.exists():
            print(f"  [skip] {p} not found"); continue
        ddqn = DDQNAgent(base, sd)
        ddqn.q.load_state_dict(torch.load(p, map_location="cpu",
                                          weights_only=False)["q"])
        ddqn.eps = 0.0
        r = evaluate(lambda e, s, df: ddqn.act(s, e)[0], base, sd, a.episodes)
        rows.append(r)
        print(f"  {'FedAvg-DDQN':16s} seed {sd}: total {r['viol_total']:6.2f}%  "
              f"reported {r['viol_reported']:6.2f}%  drops {r['drop_rate']:6.2f}%")
    if rows:
        out["FedAvg-DDQN"] = rows

    for tag, over in ARMS.items():
        cfg = dataclasses.replace(base, **over) if over else base
        rows = []
        for sd in range(a.seeds):
            p = ck / f"agent_{tag}_{sd}.pt"
            if not p.exists():
                print(f"  [skip] {p} not found"); continue
            ag = CVaRPPOAgent(cfg, sd)
            d = torch.load(p, map_location="cpu", weights_only=False)
            ag.actor.load_state_dict(d["actor"])
            if "critic" in d:
                ag.critic.load_state_dict(d["critic"])
            r = evaluate(lambda e, s, df: ag.act_greedy(s, df), cfg, sd, a.episodes)
            rows.append(r)
            print(f"  {tag:16s} seed {sd}: total {r['viol_total']:6.2f}%  "
                  f"reported {r['viol_reported']:6.2f}%  drops {r['drop_rate']:6.2f}%")
        if rows:
            out[tag] = rows

    for tag, fn in (("Heuristic", lambda e, s, df: heuristic_action(e, base)),
                    ("Heuristic-adaptive-n",
                     lambda e, s, df: heuristic_adaptive_n_action(e, base, 1e-2))):
        rows = []
        for sd in range(a.seeds):
            r = evaluate(fn, base, sd, a.episodes)
            rows.append(r)
            print(f"  {tag:20s} seed {sd}: total {r['viol_total']:6.2f}%  "
                  f"reported {r['viol_reported']:6.2f}%  drops {r['drop_rate']:6.2f}%")
        out[tag] = rows

    json.dump(out, open(a.out, "w"), indent=2)

    fixed = {}
    for nk in (350, 500, 800):
        fixed[f"n{nk}"] = [evaluate(lambda e, s, df, nk=nk: heuristic_fixed_n_action(e, base, nk),
                                    base, sd, a.episodes) for sd in range(a.seeds)]
        print(f"  Heuristic, n_k fixed at {nk}: total "
              f"{np.mean([r['viol_total'] for r in fixed[f'n{nk}']]):6.2f}%", flush=True)
    Path("results").mkdir(exist_ok=True)
    json.dump(fixed, open("results/heuristic_fixed_n.json", "w"), indent=2)

    print(f"\n{'arm':22s}{'viol_total':>12}{'viol_reported':>15}{'drop_rate':>11}"
          f"{'p99 deliv':>11}{'med eps':>10}")
    for tag, rows in out.items():
        m = lambda k: np.mean([r[k] for r in rows])
        print(f"{tag:22s}{m('viol_total'):12.2f}{m('viol_reported'):15.2f}"
              f"{m('drop_rate'):11.2f}"
              f"{np.mean([r['delivered']['p99'] for r in rows]):11.3f}"
              f"{np.median([r['median_eps'] for r in rows]):10.2e}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
