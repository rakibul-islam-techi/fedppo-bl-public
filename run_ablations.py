from __future__ import annotations
import argparse, dataclasses, json, os, pathlib
import numpy as np

OUT = pathlib.Path("results")
CKPT = pathlib.Path("ckpt")
ROWS = OUT / "rows"


def _mean_rows(rows):
    out = {}
    for k, v in rows[0].items():
        if isinstance(v, dict):
            out[k] = {kk: float(np.mean([r[k][kk] for r in rows])) for kk in v}
        elif isinstance(v, (int, float)):
            out[k] = float(np.mean([r[k] for r in rows]))
    return out


def _one(job):
    cfg, mode, sd, threads, tag = job
    import torch
    torch.set_num_threads(max(1, threads))
    from train import train_ppo
    from evaluate_arms import evaluate
    local = mode in ("local", "local_shared")
    head, *_ = train_ppo(cfg, sd, mode=mode, return_all=local)
    CKPT.mkdir(exist_ok=True)
    if local:
        rows = []
        for b, ag in enumerate(head):
            torch.save({"actor": ag.actor.state_dict(), "critic": ag.critic.state_dict()},
                       CKPT / f"agent_{tag}_{sd}_{b}.pt")
            rows.append(evaluate(lambda e, s, df: ag.act_greedy(s, df), cfg,
                                 sd, cfg.eval_episodes))
        row = _mean_rows(rows)
    else:
        torch.save({"actor": head.actor.state_dict(), "critic": head.critic.state_dict()},
                   CKPT / f"agent_{tag}_{sd}.pt")
        row = evaluate(lambda e, s, df: head.act_greedy(s, df), cfg, sd, cfg.eval_episodes)
    json.dump(row, open(ROWS / f"{tag}_{sd}.json", "w"))
    return tag, sd, row


def _show(tag, sd, r):
    print(f"    {tag} seed {sd}: total {r['viol_total']:6.2f}%  "
          f"recorded {r['viol_reported']:6.2f}%  drops {r['drop_rate']:5.2f}%", flush=True)


def arms(cmd, base, a):
    out = []
    if cmd in ("all", "scale"):
        for B in a.abs:
            out.append((f"scale{B}", dataclasses.replace(base, n_abs=B), "fed",
                        f"federation scale B = {B}"))
    if cmd in ("all", "block"):
        for nk in a.nk:
            out.append((f"block{int(nk)}", dataclasses.replace(base, freeze_blocklength=float(nk)),
                        "fed", f"blocklength fixed at {int(nk)}, head kept"))
    if cmd in ("all", "nohead"):
        for nk in a.nohead_nk:
            out.append((f"block{int(nk)}_nohead",
                        dataclasses.replace(base, freeze_blocklength=float(nk), drop_blocklength_head=True),
                        "fed", f"blocklength fixed at {int(nk)}, head removed"))
    if cmd in ("all", "local"):
        out.append(("local", base, "local", "independent local learning"))
    if cmd in ("all", "agg"):
        for rule in ("median", "trimmed"):
            out.append((f"agg_{rule}", dataclasses.replace(base, agg_rule=rule), "fed",
                        f"aggregation rule = {rule}"))
    if cmd in ("all", "rho"):
        for rv in a.rho:
            out.append((f"rho{str(rv).replace('.', '')}", dataclasses.replace(base, freeze_rho=float(rv)),
                        "fed", f"power split fixed at rho = {rv}"))
    if cmd == "quantile":
        for M in a.quantiles:
            out.append((f"M{M}", dataclasses.replace(base, n_quantiles=M), "fed",
                        f"quantile critic with M = {M}"))
    if cmd == "extra":
        out.append(("local_shared", base, "local_shared",
                    "independent local learning from a shared initialization"))
        for lr in a.pooled_lr:
            out.append((f"pooled_lr{lr:g}", dataclasses.replace(base, lr_actor=lr, lr_critic=lr), "pooled",
                        f"non-federated PPO-QR, learning rate {lr:g}"))
    if cmd == "kappa":
        for kv in a.kappa:
            out.append((f"kappa{kv:g}_qr", dataclasses.replace(base, kappa_violation=kv), "fed",
                        f"FedPPO-QR, violation penalty kappa = {kv:g}"))
            out.append((f"kappa{kv:g}_mean",
                        dataclasses.replace(base, kappa_violation=kv, n_quantiles=1, lambda_cvar=0.0),
                        "fed", f"FedPPO-Mean, violation penalty kappa = {kv:g}"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all", "block", "nohead", "local", "agg", "scale", "rho", "extra", "quantile",
                                    "kappa"])
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--nk", type=float, nargs="+", default=[350, 500, 800])
    ap.add_argument("--nohead-nk", type=float, nargs="+", default=[500])
    ap.add_argument("--abs", type=int, nargs="+", default=[10])
    ap.add_argument("--rho", type=float, nargs="+", default=[0.5])
    ap.add_argument("--pooled-lr", type=float, nargs="+", default=[1e-4, 1e-3])
    ap.add_argument("--quantiles", type=int, nargs="+", default=[20, 50, 100, 500])
    ap.add_argument("--kappa", type=float, nargs="+", default=[1.0])
    ap.add_argument("--out", default="results/ablations.json")
    a = ap.parse_args()

    from config import full_config, describe
    base = full_config()
    seeds = a.seeds or base.seeds
    print(describe(base), flush=True)
    OUT.mkdir(exist_ok=True); ROWS.mkdir(exist_ok=True)

    todo = arms(a.cmd, base, a)
    threads = max(1, (os.cpu_count() or 2) // max(1, a.workers))
    jobs = [(cfg, mode, sd, threads, tag) for tag, cfg, mode, _ in todo
            for sd in range(seeds) if not (ROWS / f"{tag}_{sd}.json").exists()]
    for tag, _, _, what in todo:
        n = sum(j[4] == tag for j in jobs)
        print(f"[ablation] {tag:16s} {what}: {n}/{seeds} seeds to run", flush=True)
    if jobs:
        print(f"[ablation] {len(jobs)} training jobs on {min(a.workers, len(jobs))} worker(s), "
              f"{threads} thread(s) each", flush=True)
    if jobs and a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(processes=min(a.workers, len(jobs))) as pool:
            for tag, sd, r in pool.imap_unordered(_one, jobs):
                _show(tag, sd, r)
    else:
        for j in jobs:
            tag, sd, r = _one(j); _show(tag, sd, r)

    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for tag, _, _, _ in todo:
        res[tag] = [json.load(open(ROWS / f"{tag}_{sd}.json")) for sd in range(seeds)]
    json.dump(res, open(a.out, "w"), indent=2)

    print(f"\n{'arm':16s}{'viol_total':>12}{'viol_recorded':>15}{'drops':>9}{'p99_deliv':>11}")
    for k, rows in res.items():
        m = lambda f: np.mean([f(r) for r in rows])
        print(f"{k:16s}{m(lambda r: r['viol_total']):12.2f}"
              f"{m(lambda r: r['viol_reported']):15.2f}"
              f"{m(lambda r: r['drop_rate']):9.2f}"
              f"{m(lambda r: r['delivered']['p99']):11.3f}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
