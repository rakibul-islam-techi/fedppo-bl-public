from __future__ import annotations
import argparse, dataclasses, itertools, json, os, pathlib
import numpy as np

CKPT = pathlib.Path("ckpt")
ROWS = pathlib.Path("results") / "rows"


def _clip_count(sched, cfg):
    a = np.asarray(sched, float)
    return dict(alpha_clip_rounds=int(np.sum((a <= cfg.alpha_min) | (a >= cfg.alpha_max))), rounds=int(a.size))


def _train_adaptive(job):
    cfg, sd, threads, tag = job
    import torch
    torch.set_num_threads(max(1, threads))
    from train import train_ppo
    from evaluate_arms import evaluate
    ag, *_ = train_ppo(cfg, sd, mode="fed")
    torch.save({"actor": ag.actor.state_dict(), "critic": ag.critic.state_dict(),
                "alpha_sched": ag._alpha_sched, "viol_trace": ag._viol_trace},
               CKPT / f"agent_{tag}_{sd}.pt")
    row = evaluate(lambda e, s, df: ag.act_greedy(s, df), cfg, sd, cfg.eval_episodes)
    row["alpha_final"] = float(ag._alpha_r)
    row.update(_clip_count(ag._alpha_sched, cfg))
    json.dump(row, open(ROWS / f"{tag}_{sd}.json", "w"))
    return tag, sd, row


def _reuse(cfg, key, sd, alpha_from_curve):
    import torch
    from rl import CVaRPPOAgent
    from evaluate_arms import evaluate
    p = CKPT / f"agent_{key}_{sd}.pt"
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing; run the main training first")
    ag = CVaRPPOAgent(cfg, sd)
    ag.actor.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["actor"])
    row = evaluate(lambda e, s, df: ag.act_greedy(s, df), cfg, sd, cfg.eval_episodes)
    if alpha_from_curve:
        sched = json.load(open(CKPT / f"curve_{key}_{sd}.json"))["alpha_sched"]
        row["alpha_final"] = float(sched[-1])
        row.update(_clip_count(sched, cfg))
    else:
        row["alpha_final"] = float(cfg.cvar_alpha)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="results/adaptive_alpha.json")
    a = ap.parse_args()

    from config import full_config, describe
    base = full_config()
    seeds = a.seeds or base.seeds
    print(describe(base), flush=True)
    ROWS.mkdir(parents=True, exist_ok=True); CKPT.mkdir(exist_ok=True)
    mean_cfg = dataclasses.replace(base, n_quantiles=1, lambda_cvar=0.0)

    res = {"qr_fixed": [], "qr_adaptive": [], "mean_adaptive": []}
    for sd in range(seeds):
        res["qr_fixed"].append(_reuse(base, "FedPPO-QR", sd, False))
        res["mean_adaptive"].append(_reuse(dataclasses.replace(mean_cfg, adaptive_alpha=True),
                                           "FedPPO-Mean", sd, True))
    print("  qr_fixed = main FedPPO-QR checkpoints; mean_adaptive = main FedPPO-Mean "
          "checkpoints (with lambda = 0 the risk level never enters the update)", flush=True)

    tag = "qr_adaptive"
    cfg = dataclasses.replace(base, adaptive_alpha=True)
    rows, todo = {}, []
    for sd in range(seeds):
        f = ROWS / f"{tag}_{sd}.json"
        if f.exists():
            rows[sd] = json.load(open(f))
        else:
            todo.append(sd)
    threads = max(1, (os.cpu_count() or 2) // max(1, a.workers))
    jobs = [(cfg, sd, threads, tag) for sd in todo]
    if jobs and a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(processes=min(a.workers, len(jobs))) as pool:
            for _, sd, r in pool.imap_unordered(_train_adaptive, jobs):
                rows[sd] = r
                print(f"  {tag} seed {sd}: total {r['viol_total']:6.2f}%  "
                      f"alpha_final {r['alpha_final']:.3f}", flush=True)
    else:
        for j in jobs:
            _, sd, r = _train_adaptive(j); rows[sd] = r
            print(f"  {tag} seed {sd}: total {r['viol_total']:6.2f}%  "
                  f"alpha_final {r['alpha_final']:.3f}", flush=True)
    res[tag] = [rows[s] for s in sorted(rows)]
    json.dump(res, open(a.out, "w"), indent=2)

    print(f"\n{'arm':16s}{'viol_total':>12}{'viol_recorded':>15}{'drops':>9}"
          f"{'p99':>8}{'p99.9':>9}{'alpha_end':>11}")
    for t, rr in res.items():
        m = lambda f: np.mean([f(r) for r in rr])
        print(f"{t:16s}{m(lambda r: r['viol_total']):12.2f}{m(lambda r: r['viol_reported']):15.2f}"
              f"{m(lambda r: r['drop_rate']):9.2f}{m(lambda r: r['delivered']['p99']):8.3f}"
              f"{m(lambda r: r['delivered']['p999']):9.3f}{m(lambda r: r['alpha_final']):11.3f}")

    from scipy import stats as st
    print("\nPAIRED, matched seeds, delivered-task latency")
    for x, y in itertools.combinations(res, 2):
        for k, g in (("viol_total", None), ("p99", "delivered"), ("p999", "delivered")):
            f = (lambda r: r[g][k]) if g else (lambda r: r[k])
            d = np.array([f(r) for r in res[x]]) - np.array([f(r) for r in res[y]])
            n = len(d)
            if n < 2:
                continue
            se = d.std(ddof=1) / np.sqrt(n)
            t = d.mean() / se if se > 0 else np.inf
            tc = st.t.ppf(0.975, n - 1)
            print(f"  {x:14s} - {y:14s} {k:10s} {d.mean():+8.3f}  "
                  f"[{d.mean()-tc*se:+7.3f},{d.mean()+tc*se:+7.3f}]  "
                  f"p={2*st.t.sf(abs(t), n-1):.4f}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
