"""Recorded latency of every evaluated task, per method and training seed.

Evaluates the trained FedPPO-QR, FedPPO-Mean, non-federated (Centralized-QR) and
FedAvg-DDQN checkpoints and both heuristics on the evaluation environments of
evaluate_arms.py (environment seed 10100 + s, greedy policies), and stores the
recorded latencies in seconds. Fig. 6 and Table III are computed from this file
(make_figures.py, cvar_recompute.py).

    python pooled_latencies.py --ckpt ckpt --seeds 10 --workers 5
"""
from __future__ import annotations
import argparse, dataclasses, pickle
from pathlib import Path
import numpy as np

METHODS = ("FedPPO-QR", "FedPPO-Mean", "Centralized-QR", "FedAvg-DDQN", "Heuristic", "Heuristic-adaptive-n")


def _job(job):
    method, sd, ckpt = job
    import torch
    torch.set_num_threads(1)
    from config import full_config
    from sim_core import AerialEdgeEnv
    from rl import CVaRPPOAgent, DDQNAgent, heuristic_action, heuristic_adaptive_n_action
    cfg = full_config()
    if method in ("FedPPO-QR", "FedPPO-Mean", "Centralized-QR"):
        c = dataclasses.replace(cfg, n_quantiles=1, lambda_cvar=0.0) if method == "FedPPO-Mean" else cfg
        ag = CVaRPPOAgent(c, sd)
        ag.actor.load_state_dict(torch.load(f"{ckpt}/agent_{method}_{sd}.pt", map_location="cpu",
                                            weights_only=False)["actor"])
        policy = lambda e, s, df: ag.act_greedy(s, df)
    elif method == "FedAvg-DDQN":
        ag = DDQNAgent(cfg, sd)
        ag.q.load_state_dict(torch.load(f"{ckpt}/agent_FedAvg-DDQN_{sd}.pt", map_location="cpu",
                                        weights_only=False)["q"])
        ag.eps = 0.0
        policy = lambda e, s, df: ag.act(s, e)[0]
    elif method == "Heuristic":
        policy = lambda e, s, df: heuristic_action(e, cfg)
    else:
        policy = lambda e, s, df: heuristic_adaptive_n_action(e, cfg, 1e-2)
    env = AerialEdgeEnv(cfg, 0, 10_000 + 100 + sd)
    lat = []
    for _ in range(cfg.eval_episodes):
        s = env.reset(); df, _ = env.device_features()
        for _ in range(cfg.steps_per_episode):
            s, _, _, info = env.step(policy(env, s, df)); df, _ = env.device_features()
            lat.append(info["latencies"])
    return method, sd, np.concatenate(lat)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="results/latencies_per_seed.pkl")
    a = ap.parse_args()
    jobs = [(m, sd, a.ckpt) for m in METHODS for sd in range(a.seeds)]
    out = {m: [None] * a.seeds for m in METHODS}
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for m, sd, x in pool.imap_unordered(_job, jobs):
                out[m][sd] = x
    else:
        for j in jobs:
            m, sd, x = _job(j); out[m][sd] = x
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(out, f)
    for m in METHODS:
        x = np.concatenate(out[m]) * 1e3
        print(f"{m:22s} tasks {x.size:7d}  mean {x.mean():.3f} ms  p99 {np.percentile(x, 99):.3f} ms")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
