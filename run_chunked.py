import os
import sys
import json
import time
import pickle
import numpy as np
import torch

from config import smoke_config, full_config, describe
from train import (train_ppo, train_ddqn, eval_policy, cvar, bootstrap_ci,
                   sic_heatmap, _clone)
from rl import heuristic_action

CK = "ckpt_sparse" if "--sparse" in sys.argv else "ckpt"
os.makedirs(CK, exist_ok=True)


def _p(name): return os.path.join(CK, name)


def _train_one(task):
    key, cfg, sd, threads = task
    torch.set_num_threads(max(1, threads))
    t = time.time()
    if key == "FedAvg-DDQN":
        agent, r, c, x = train_ddqn(cfg, sd)
        state = {"q": agent.q.state_dict()}
    elif key == "Centralized-QR":
        agent, r, c, x = train_ppo(cfg, sd, federated=False)
        state = {"actor": agent.actor.state_dict(
        ), "critic": agent.critic.state_dict()}
    else:
        agent, r, c, x = train_ppo(cfg, sd, federated=True)
        state = {"actor": agent.actor.state_dict(
        ), "critic": agent.critic.state_dict()}
    torch.save(state, _p(f"agent_{key}_{sd}.pt"))
    json.dump({"reward": np.asarray(r).tolist(), "cvar": np.asarray(c).tolist(),
               "x": np.asarray(x).tolist(),
               "alpha_sched": list(getattr(agent, "_alpha_sched", [])),
               "viol_trace": list(getattr(agent, "_viol_trace", []))},
              open(_p(f"curve_{key}_{sd}.json"), "w"))
    dt = time.time() - t
    print(f"  [done] {key} seed {sd}: {dt/60:.1f} min", flush=True)
    return sd, dt


def _seed_done(key, sd):
    return (os.path.exists(_p(f"curve_{key}_{sd}.json"))
            and os.path.exists(_p(f"agent_{key}_{sd}.pt")))


def _run_pool(tasks, workers):
    if workers > 1 and len(tasks) > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(workers, len(tasks))) as pool:
            for _ in pool.imap_unordered(_train_one, tasks):
                pass
    else:
        for tsk in tasks:
            _train_one(tsk)


def _aggregate(cfg, key):
    rc, cc, xs = [], [], None
    for sd in range(cfg.seeds):
        d = json.load(open(_p(f"curve_{key}_{sd}.json")))
        rc.append(d["reward"])
        cc.append(d["cvar"])
        xs = np.asarray(d["x"])
    json.dump({"episodes": xs.tolist(),
               "reward_mean": np.mean(rc, 0).tolist(),
               "reward_std": np.std(rc, 0).tolist(),
               "cvar_mean": np.mean(cc, 0).tolist(),
               "cvar_std": np.std(cc, 0).tolist(),
               "seeds": cfg.seeds},
              open(_p(f"curve_{key}.json"), "w"))
    print(f"[done] {key}", flush=True)


def train_stack(cfg, key, workers):
    train_many({key: cfg}, workers)


def train_many(cfgs, workers):
    threads = max(1, (os.cpu_count() or 2) // max(1, workers))
    tasks = [(key, c, sd, threads) for key, c in cfgs.items()
             for sd in range(c.seeds) if not _seed_done(key, sd)]
    for key, c in cfgs.items():
        n = sum(t[0] == key for t in tasks)
        print(f"[stage] {key}: {n}/{c.seeds} seeds to run", flush=True)
    if tasks:
        print(f"[stage] {len(tasks)} training jobs on {min(workers, len(tasks))} "
              f"worker(s), {threads} thread(s) each", flush=True)
        _run_pool(tasks, workers)
    for key, c in cfgs.items():
        _aggregate(c, key)


def stage_main(cfg, workers):
    train_many({"FedAvg-DDQN": cfg,
                "FedPPO-QR": cfg,
                "FedPPO-Mean": _clone(cfg, n_quantiles=1, lambda_cvar=0.0),
                "Centralized-QR": cfg}, workers)


def stage_qr(cfg, workers):
    train_stack(cfg, "FedPPO-QR", workers)


def stage_mean(cfg, workers):
    train_stack(_clone(cfg, n_quantiles=1, lambda_cvar=0.0),
                "FedPPO-Mean", workers)


def stage_ddqn(cfg, workers):
    train_stack(cfg, "FedAvg-DDQN", workers)


def stage_cen(cfg, workers):
    train_stack(cfg, "Centralized-QR", workers)


def _load_ppo_agent(cfg, key, sd):
    from rl import CVaRPPOAgent
    ag = CVaRPPOAgent(cfg, sd)
    d = torch.load(_p(f"agent_{key}_{sd}.pt"))
    ag.actor.load_state_dict(d["actor"])
    ag.critic.load_state_dict(d["critic"])
    return ag


def _load_ddqn_agent(cfg, sd):
    from rl import DDQNAgent
    ag = DDQNAgent(cfg, sd)
    ag.q.load_state_dict(torch.load(_p(f"agent_FedAvg-DDQN_{sd}.pt"))["q"])
    return ag


def stage_evals(cfg, workers):
    if os.path.exists(_p("evals.pkl")):
        print("[skip] evals")
        return
    cfg_m = _clone(cfg, n_quantiles=1, lambda_cvar=0.0)

    def make_policies(sd):
        qr = _load_ppo_agent(cfg, "FedPPO-QR", sd)
        mean = _load_ppo_agent(cfg_m, "FedPPO-Mean", sd)
        cen = _load_ppo_agent(cfg, "Centralized-QR", sd)
        ddqn = _load_ddqn_agent(cfg, sd)
        ddqn.eps = 0.0
        return {
            "FedPPO-QR": lambda e, s, df: qr.act_greedy(s, df),
            "FedPPO-Mean": lambda e, s, df: mean.act_greedy(s, df),
            "FedAvg-DDQN": lambda e, s, df: ddqn.act(s, e)[0],
            "Centralized-QR": lambda e, s, df: cen.act_greedy(s, df),
            "Heuristic": lambda e, s, df: heuristic_action(e, cfg),
        }

    pooled = {k: [] for k in ["FedPPO-QR", "FedPPO-Mean", "FedAvg-DDQN",
                              "Centralized-QR", "Heuristic"]}
    for sd in range(cfg.seeds):
        pols = make_policies(sd)
        for name, pol in pols.items():
            lat, _ = eval_policy(
                pol, cfg, seeds=[100 + sd], n_episodes=cfg.eval_episodes)
            pooled[name].append(lat)
        print(f"  [eval] seed {sd} done", flush=True)

    with open(_p("pooled_per_seed.pkl"), "wb") as _f:
        pickle.dump(pooled, _f)
    print(f"  [saved] {_p('pooled_per_seed.pkl')}", flush=True)

    tail_table, eval_lat = {}, {}
    rng = np.random.default_rng(0)
    for name, lats in pooled.items():
        lat = np.concatenate(lats)
        viol = float(np.mean(lat > cfg.deadline_s))
        lat_ms = lat * 1e3
        eval_lat[name] = lat_ms
        m, mlo, mhi = bootstrap_ci(lat_ms, np.mean, rng=rng)
        p99, p99lo, p99hi = bootstrap_ci(
            lat_ms, lambda z: np.percentile(z, 99), rng=rng)
        p999, p999lo, p999hi = bootstrap_ci(
            lat_ms, lambda z: np.percentile(z, 99.9), rng=rng)
        cv, cvlo, cvhi = bootstrap_ci(
            lat_ms, lambda z: cvar(z, cfg.cvar_alpha), rng=rng)
        tail_table[name] = {"n_samples": int(lat_ms.size),
                            "mean": m, "p95": float(np.percentile(lat_ms, 95)),
                            "p99": p99, "p99_ci": [p99lo, p99hi],
                            "p999": p999, "p999_ci": [p999lo, p999hi],
                            "cvar99": cv, "cvar99_ci": [cvlo, cvhi], "violation": viol}
        print(f"  {name:15s} viol={viol*100:5.2f}%  p99={p99:.3f}  cvar={cv:.3f} ms",
              flush=True)
    surfaces = [sic_heatmap(_load_ppo_agent(cfg, "FedPPO-QR", sd), cfg) for sd in range(cfg.seeds)]
    gr, qr_ = surfaces[0][0], surfaces[0][1]
    Zs = np.stack([z for _, _, z in surfaces])
    Z = Zs.mean(axis=0)
    agree = float(np.mean(np.sign(Zs) == np.sign(Z)[None]))
    print(f"  SIC surface: mean over {len(surfaces)} seeds; per-seed sign agrees with the mean "
          f"in {100 * agree:.1f}% of cells", flush=True)
    pickle.dump({"tail_table": tail_table, "eval_lat": eval_lat,
                 "fig6": {"gain_ratio": gr.tolist(), "demand_ratio": qr_.tolist(),
                          "Z": Z.tolist(), "seeds": len(surfaces), "sign_agreement": agree}},
                open(_p("evals.pkl"), "wb"))
    print("[done] evals", flush=True)


def stage_assemble(cfg, workers):
    results = {"config": cfg.as_dict(), "curves": {}}
    for key in ["FedPPO-QR", "FedPPO-Mean", "FedAvg-DDQN", "Centralized-QR"]:
        results["curves"][key] = json.load(open(_p(f"curve_{key}.json")))
    ev = pickle.load(open(_p("evals.pkl"), "rb"))
    results["tail_table"] = ev["tail_table"]
    results["fig6"] = ev["fig6"]
    json.dump(results, open("results.json", "w"), indent=2)
    np.savez("results.npz", **{f"lat_{k}": v for k,
             v in ev["eval_lat"].items()})
    print("[done] assemble -> results.json / results.npz", flush=True)


STAGES = {"main": stage_main, "qr": stage_qr, "mean": stage_mean, "ddqn": stage_ddqn, "cen": stage_cen,
          "evals": stage_evals, "assemble": stage_assemble}

if __name__ == "__main__":
    args = sys.argv[1:]
    stage = args[0] if args and not args[0].startswith("-") else "all"
    mode = "full" if "full" in args else (
        "smoke" if "smoke" in args else "smoke")
    cfg = full_config() if mode == "full" else smoke_config()
    if "--sparse" in args:
        import dataclasses
        cfg = dataclasses.replace(cfg, devices_per_abs=(2, 4))
        print("[run] SPARSE scenario: 2-4 devices/ABS -> checkpoints in ckpt_sparse/")
    if "--seeds" in args:
        import dataclasses
        cfg = dataclasses.replace(cfg, seeds=int(
            args[args.index("--seeds") + 1]))
    if "--kappa" in args:
        import dataclasses
        cfg = dataclasses.replace(cfg, kappa_violation=float(
            args[args.index("--kappa") + 1]))
    workers = int(args[args.index("--workers") + 1]
                  ) if "--workers" in args else 1
    print(
        f"[run] mode={cfg.name} seeds={cfg.seeds} workers={workers}", flush=True)
    print(f"[run] {describe(cfg)}", flush=True)
    order = ["main", "evals", "assemble"]
    for st in (order if stage == "all" else [stage]):
        STAGES[st](cfg, workers)
