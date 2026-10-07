from __future__ import annotations
import argparse, dataclasses, json
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8.5, "legend.fontsize": 6.6,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "TeX Gyre Termes"],
    "mathtext.fontset": "stix", "axes.grid": True, "grid.alpha": 0.28, "grid.linewidth": 0.4,
    "lines.linewidth": 1.2, "lines.markersize": 3.2, "axes.linewidth": 0.6,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
})
COLORS = ["#0072B2", "#D55E00", "#009E73"]
STYLES = [("-", "o"), ("--", "s"), (":", "^")]
SEEDS, EPISODES = 20, 5
N_GRID = [150, 200, 250, 300, 350, 400, 450, 500, 525, 550, 600, 650, 700, 750, 800, 850, 900]

POWER_DBM = [17.0, 20.0, 23.0]
CPU_GHZ = [6.0, 8.0, 10.0]
ALLOCATION = ["own", "equal"]
TRAINED = {"FedPPO-QR": {}, "FedPPO-Mean": dict(n_quantiles=1, lambda_cvar=0.0),
           "Centralized-QR": {}, "FedAvg-DDQN": {}}
CKPT = Path("ckpt")


def rollout(cfg, policy, allocation, n_k=None, seeds=SEEDS, episodes=EPISODES):
    from sim_core import AerialEdgeEnv
    from rl import blocklength_logit
    D = cfg.max_devices
    lat, drop, vio, nb, raw = [], [], [], [], []
    for sd in range(seeds):
        env = AerialEdgeEnv(cfg, sd % cfg.n_abs, 6000 + sd)
        for _ in range(episodes):
            s = env.reset(); df, _ = env.device_features()
            for _ in range(cfg.steps_per_episode):
                a = np.array(policy(env, s, df), dtype=float)
                if allocation == "equal":
                    a[2 * D:3 * D] = 0.0
                if n_k is not None:
                    a[3 * D:4 * D] = blocklength_logit(cfg, n_k)
                s, _, _, info = env.step(a); df, _ = env.device_features()
                lat.append(info["latencies"]); drop.append(info["dropped"])
                vio.append(info["violated"]); nb.append(info["n_block"])
                raw.append(np.where(info["dropped"], info["t_tx"], info["t_tx"] + info["t_comp"]))
    lat = np.concatenate(lat) * 1e3
    drop = np.concatenate(drop); vio = np.concatenate(vio); nb = np.concatenate(nb)
    deliv = np.concatenate(raw)[~drop] * 1e3
    return dict(n_k=float(np.median(nb)),
                mean_ms=float(lat.mean()),
                viol_total=100.0 * float(vio.mean()),
                viol_reported=100.0 * float(np.mean(lat > 1.0)),
                drop_rate=100.0 * float(drop.mean()),
                p99_delivered_ms=float(np.percentile(deliv, 99)) if deliv.size else float("nan"),
                n_tasks=int(lat.size))


def trained_policies(base, seeds):
    import torch
    from rl import CVaRPPOAgent, DDQNAgent
    out = {}
    for arm, over in TRAINED.items():
        cfg = dataclasses.replace(base, **over) if over else base
        pols = []
        for sd in range(seeds):
            p = CKPT / f"agent_{arm}_{sd}.pt"
            if not p.exists():
                continue
            d = torch.load(p, map_location="cpu", weights_only=False)
            if arm == "FedAvg-DDQN":
                ag = DDQNAgent(cfg, sd)
                ag.q.load_state_dict(d["q"])
                ag.eps = 0.0
                pols.append((sd, lambda e, s, df, ag=ag: ag.act(s, e)[0]))
            else:
                ag = CVaRPPOAgent(cfg, sd)
                ag.actor.load_state_dict(d["actor"])
                pols.append((sd, lambda e, s, df, ag=ag: ag.act_greedy(s, df)))
        if pols:
            out[arm] = pols
    return out


def settings(base):
    out = []
    for p in POWER_DBM:
        out.append(("power", f"P_k = {p:g} dBm", dataclasses.replace(base, pmax_dbm=p), "own"))
    for f in CPU_GHZ:
        out.append(("cpu_capacity", f"F_max = {f:g} GHz", dataclasses.replace(base, fmax_hz=f * 1e9), "own"))
    for al in ALLOCATION:
        out.append(("cpu_allocation", f"{al} CPU allocation", base, al))
    return out


def run_setting(job):
    factor, label, cfg, alloc, base, n_seeds = job
    import torch
    torch.set_num_threads(1)
    from config import check
    from rl import heuristic_action, heuristic_adaptive_n_action
    check(cfg)
    static = lambda e, s, df: heuristic_action(e, cfg)
    adaptive = lambda e, s, df: heuristic_adaptive_n_action(e, cfg, 1e-2)
    rows = [rollout(cfg, static, alloc, n_k=nk) for nk in N_GRID]
    trained = {arm: [dict(seed=sd, **rollout(cfg, pol, alloc)) for sd, pol in pols]
               for arm, pols in trained_policies(base, n_seeds).items()}
    return dict(factor=factor, label=label, pmax_dbm=cfg.pmax_dbm, fmax_ghz=cfg.fmax_hz / 1e9,
                allocation=alloc, sweep=rows, static_heuristic=rows[N_GRID.index(525)],
                adaptive_heuristic=rollout(cfg, adaptive, alloc), trained=trained)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sweep_grid")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    import sys; sys.path.insert(0, ".")
    from config import full_config, describe
    base = full_config()
    print(describe(base))
    print(f"{SEEDS} environments x {EPISODES} episodes x {base.steps_per_episode} intervals per point; "
          f"trained arms from {CKPT}/, "
          f"trained at {base.pmax_dbm:g} dBm and {base.fmax_hz / 1e9:g} GHz, evaluated without retraining", flush=True)
    jobs = [(f, l, c, al, base, base.seeds) for f, l, c, al in settings(base)]
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(processes=min(a.workers, len(jobs))) as pool:
            res = pool.map(run_setting, jobs)
    else:
        res = [run_setting(j) for j in jobs]

    for r in res:
        best = min(r["sweep"], key=lambda x: x["viol_total"])
        line = (f"{r['label']:22s} best n_k {best['n_k']:4.0f} total {best['viol_total']:6.2f}%  "
                f"static {r['static_heuristic']['viol_total']:6.2f}%  adaptive {r['adaptive_heuristic']['viol_total']:6.2f}%")
        for arm, rows in r["trained"].items():
            line += f"  {arm} {np.mean([x['viol_total'] for x in rows]):6.2f}%"
        print(line, flush=True)
    json.dump(res, open(a.out + ".json", "w"), indent=2)

    titles = {"power": "(a) transmit power", "cpu_capacity": "(b) CPU capacity",
              "cpu_allocation": "(c) CPU allocation"}
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.1), sharex=True, sharey="row")
    for col_i, factor in enumerate(titles):
        ax, axm = axes[0, col_i], axes[1, col_i]
        group = [r for r in res if r["factor"] == factor]
        any_rec, hidden_star = False, False
        for r, col, (ls, mk) in zip(group, COLORS, STYLES):
            n = np.array([x["n_k"] for x in r["sweep"]])
            tot = np.array([x["viol_total"] for x in r["sweep"]])
            rec = np.array([x["viol_reported"] for x in r["sweep"]])
            mean = np.array([x["mean_ms"] for x in r["sweep"]])
            lab = (r["label"].replace("P_k", "$P_k$").replace("F_max", r"$F_i^{\max}$")
                   .replace(" allocation", ""))
            ax.plot(n[tot > 0], tot[tot > 0], ls, marker=mk, color=col, mfc="white", mew=0.9, label=lab)
            if np.any(rec > 0):
                any_rec = True
                ax.plot(n[rec > 0], rec[rec > 0], ls, color=col, lw=0.7, alpha=0.6)
            axm.plot(n, mean, ls, marker=mk, color=col, mfc="white", mew=0.9)
            qr = r["trained"].get("FedPPO-QR")
            if qr:
                qn = np.mean([x["n_k"] for x in qr])
                qt = np.mean([x["viol_total"] for x in qr])
                qm = np.mean([x["mean_ms"] for x in qr])
                if qt > 0:
                    ax.plot([qn], [qt], "*", ms=8, color=col, mec="black", mew=0.4, zorder=5)
                else:
                    hidden_star = True
                axm.plot([qn], [qm], "*", ms=8, color=col, mec="black", mew=0.4, zorder=5)
        notes = []
        if not any_rec:
            notes.append("recorded-latency rate 0% at every $n_k$")
        if hidden_star:
            notes.append("FedPPO-QR at 0% not shown")
        if notes:
            ax.text(0.03, 0.04, "\n".join(notes), transform=ax.transAxes, fontsize=6.3, color="0.35")
        ax.set_yscale("log")
        ax.set_xlim(150, 900)
        ax.set_title(titles[factor], fontsize=8)
        h, l = ax.get_legend_handles_labels()
        if any(r["trained"].get("FedPPO-QR") for r in group):
            h.append(Line2D([], [], marker="*", ls="none", ms=7, color="0.55", mec="black", mew=0.4))
            l.append("FedPPO-QR")
        ax.legend(h, l, loc="upper right", frameon=True, framealpha=0.92, edgecolor="0.75", fancybox=False)
        axm.set_xlabel(r"Blocklength $n_k$ (channel uses)")
    axes[0, 0].set_ylabel("Violation rate, drops counted (%)")
    axes[1, 0].set_ylabel("Mean latency (ms)")
    fig.tight_layout(); fig.savefig(a.out + ".pdf"); plt.close(fig)
    print(f"wrote {a.out}.pdf and {a.out}.json (curves: static heuristic with n_k fixed; "
          f"stars: FedPPO-QR at its median n_k, mean over training seeds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
