from __future__ import annotations
import argparse, dataclasses, json, os, pathlib
import numpy as np

CKPT = pathlib.Path("ckpt")
ROWS = pathlib.Path("results") / "rows"


def _train(job):
    cfg, mult, sd, threads = job
    import torch
    torch.set_num_threads(max(1, threads))
    from train import train_ppo
    from evaluate_arms import evaluate
    ag, r, c, x = train_ppo(cfg, sd, mode="fed")
    tag = f"budget{mult}x"
    torch.save({"actor": ag.actor.state_dict(), "critic": ag.critic.state_dict()},
               CKPT / f"agent_{tag}_{sd}.pt")
    json.dump({"reward": np.asarray(r).tolist()}, open(CKPT / f"curve_{tag}_{sd}.json", "w"))
    row = evaluate(lambda e, s, df: ag.act_greedy(s, df), cfg, sd, cfg.eval_episodes)
    json.dump(row, open(ROWS / f"{tag}_{sd}.json", "w"))
    return mult, sd, row


def _reference(cfg, sd):
    import torch
    from rl import CVaRPPOAgent
    from evaluate_arms import evaluate
    f = ROWS / f"budget1x_{sd}.json"
    if f.exists():
        return json.load(open(f))
    ag = CVaRPPOAgent(cfg, sd)
    ag.actor.load_state_dict(torch.load(CKPT / f"agent_FedPPO-QR_{sd}.pt", map_location="cpu",
                                        weights_only=False)["actor"])
    row = evaluate(lambda e, s, df: ag.act_greedy(s, df), cfg, sd, cfg.eval_episodes)
    json.dump(row, open(f, "w"))
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--mult", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="results/budget.json")
    a = ap.parse_args()

    from config import full_config, describe
    base = full_config()
    print(describe(base), flush=True)
    CKPT.mkdir(exist_ok=True); ROWS.mkdir(parents=True, exist_ok=True)

    res = {"1x": [_reference(base, sd) for sd in range(a.seeds)]}
    threads = max(1, (os.cpu_count() or 2) // max(1, a.workers))
    jobs = [(dataclasses.replace(base, fed_rounds=base.fed_rounds * m), m, sd, threads)
            for m in sorted(a.mult, reverse=True) for sd in range(a.seeds)
            if not (ROWS / f"budget{m}x_{sd}.json").exists()]
    print(f"[budget] {len(jobs)} training jobs: FedPPO-QR at "
          + ", ".join(f"{base.fed_rounds * m} rounds" for m in a.mult) + f", seeds 0-{a.seeds - 1}", flush=True)
    if jobs and a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(processes=min(a.workers, len(jobs))) as pool:
            for m, sd, r in pool.imap_unordered(_train, jobs):
                print(f"  {m}x seed {sd}: total {r['viol_total']:6.2f}%", flush=True)
    else:
        for j in jobs:
            m, sd, r = _train(j)
            print(f"  {m}x seed {sd}: total {r['viol_total']:6.2f}%", flush=True)
    for m in a.mult:
        res[f"{m}x"] = [json.load(open(ROWS / f"budget{m}x_{sd}.json")) for sd in range(a.seeds)]
    json.dump(res, open(a.out, "w"), indent=2)

    from fig_budget import main as plot_budget
    plot_budget()

    print(f"\n{'budget':8s}{'rounds':>8}{'total':>9}{'recorded':>10}{'drops':>8}{'p99 deliv':>11}{'median n_k':>12}{'n_k IQR':>16}")
    for lab, rows in res.items():
        m = int(lab[:-1])
        f = lambda k: np.mean([r[k] for r in rows])
        iqr = np.mean([r["n_iqr"] for r in rows], axis=0)
        print(f"{lab:8s}{base.fed_rounds * m:8d}{f('viol_total'):9.2f}{f('viol_reported'):10.2f}{f('drop_rate'):8.2f}"
              f"{np.mean([r['delivered']['p99'] for r in rows]):11.3f}{f('median_n'):12.0f}"
              f"   [{iqr[0]:.0f}, {iqr[1]:.0f}]")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
