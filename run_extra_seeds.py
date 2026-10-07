from __future__ import annotations
import argparse, json, os, pathlib
import numpy as np

CKPT = pathlib.Path("ckpt")
ROWS = pathlib.Path("results") / "rows"
TRAINED = ("FedPPO-QR", "Centralized-QR")


def _eval(job):
    arm, cfg, sd = job
    import torch
    from rl import CVaRPPOAgent, heuristic_action, heuristic_adaptive_n_action
    from evaluate_arms import evaluate
    if arm in TRAINED:
        ag = CVaRPPOAgent(cfg, sd)
        ag.actor.load_state_dict(torch.load(CKPT / f"agent_{arm}_{sd}.pt", map_location="cpu",
                                            weights_only=False)["actor"])
        fn = lambda e, s, df: ag.act_greedy(s, df)
    elif arm == "Heuristic":
        fn = lambda e, s, df: heuristic_action(e, cfg)
    else:
        fn = lambda e, s, df: heuristic_adaptive_n_action(e, cfg, 1e-2)
    row = evaluate(fn, cfg, sd, cfg.eval_episodes)
    json.dump(row, open(ROWS / f"s30_{arm}_{sd}.json", "w"))
    return arm, sd, row


def _pool(fn, jobs, workers):
    if jobs and workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(processes=min(workers, len(jobs))) as pool:
            for out in pool.imap_unordered(fn, jobs):
                yield out
    else:
        for j in jobs:
            yield fn(j)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=30)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="results/seeds30.json")
    a = ap.parse_args()

    from config import full_config, describe
    from run_chunked import _train_one, _seed_done
    base = full_config()
    total = a.total if base.name != "quick" else 2 * base.seeds
    print(describe(base), flush=True)
    CKPT.mkdir(exist_ok=True); ROWS.mkdir(parents=True, exist_ok=True)

    threads = max(1, (os.cpu_count() or 2) // max(1, a.workers))
    jobs = [(key, base, sd, threads) for key in TRAINED for sd in range(total) if not _seed_done(key, sd)]
    print(f"[seeds] {len(jobs)} training jobs for {', '.join(TRAINED)} up to seed {total - 1}", flush=True)
    for _ in _pool(_train_one, jobs, a.workers):
        pass

    arms = list(TRAINED) + ["Heuristic", "Heuristic-adaptive-n"]
    ev = [(arm, base, sd) for arm in arms for sd in range(total) if not (ROWS / f"s30_{arm}_{sd}.json").exists()]
    print(f"[seeds] {len(ev)} evaluations", flush=True)
    for _ in _pool(_eval, ev, a.workers):
        pass
    res = {arm: [json.load(open(ROWS / f"s30_{arm}_{sd}.json")) for sd in range(total)] for arm in arms}
    json.dump(res, open(a.out, "w"), indent=2)

    from paper_stats import paired
    qr = [r["viol_total"] for r in res["FedPPO-QR"]]
    print(f"\n{total} seeds, total violation rate (mean over seeds): " +
          ", ".join(f"{k} {np.mean([r['viol_total'] for r in v]):.2f}%" for k, v in res.items()))
    for arm in arms[1:]:
        for key in ("viol_total", "viol_reported"):
            r = paired([x[key] for x in res[arm]], [x[key] for x in res["FedPPO-QR"]])
            print(f"  {arm:22s} minus FedPPO-QR, {key:13s} {r['delta']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] "
                  f"p={r['p']:.4f} MDE {r['mde']:.2f}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
