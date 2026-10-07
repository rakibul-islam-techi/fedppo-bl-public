from __future__ import annotations
import argparse, json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.figsize": (3.5, 2.6), "font.size": 8, "axes.labelsize": 8.5,
    "legend.fontsize": 7, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "TeX Gyre Termes"],
    "axes.grid": True, "grid.alpha": 0.28, "grid.linewidth": 0.4,
    "axes.linewidth": 0.6, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "stix",
})
C_A, C_B = "#0072B2", "#CC79A7"


def collect(ckpt, arm, seeds, episodes, cfg):
    import torch
    from rl import CVaRPPOAgent
    from sim_core import AerialEdgeEnv
    NK, SINR, DEM, SID = [], [], [], []
    for sd in range(seeds):
        p = f"{ckpt}/agent_{arm}_{sd}.pt"
        ag = CVaRPPOAgent(cfg, sd)
        ag.actor.load_state_dict(
            torch.load(p, map_location="cpu", weights_only=False)["actor"])
        env = AerialEdgeEnv(cfg, 0, 10_000 + 100 + sd)
        for _ in range(episodes):
            s = env.reset(); df, _ = env.device_features()
            for _ in range(cfg.steps_per_episode):
                a = ag.act_greedy(s, df)
                s, _, _, info = env.step(a); df, _ = env.device_features()
                gamma = (1.0 + info["gamma1"]) * (1.0 + info["gamma2"]) - 1.0
                NK.extend(info["n_block"].tolist())
                SINR.extend((10.0 * np.log10(gamma)).tolist())
                DEM.extend(np.log10(env.workload).tolist())
                SID.extend([sd] * len(gamma))
    return map(np.asarray, (NK, SINR, DEM, SID))


def binned(x, y, sid, nb=8):
    from scipy import stats as st
    e = np.unique(np.quantile(x, np.linspace(0, 1, nb + 1)))
    c, m, lo, hi = [], [], [], []
    for i in range(len(e) - 1):
        sel = (x >= e[i]) & (x <= e[i + 1] if i == len(e) - 2 else x < e[i + 1])
        per_seed = [y[sel & (sid == s)].mean() for s in np.unique(sid) if (sel & (sid == s)).sum() >= 5]
        if len(per_seed) < 2:
            continue
        v = np.asarray(per_seed)
        h = st.t.ppf(0.975, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size)
        c.append(0.5 * (e[i] + e[i + 1])); m.append(v.mean())
        lo.append(v.mean() - h); hi.append(v.mean() + h)
    return map(np.asarray, (c, m, lo, hi))


def per_seed_r(x, y, sid):
    r = [np.corrcoef(x[sid == s], y[sid == s])[0, 1] for s in np.unique(sid)
         if np.std(y[sid == s]) > 0 and np.std(x[sid == s]) > 0]
    return [float(v) for v in r]


def auto_ylim(curves, median, n_min, n_max):
    lo = min([min(c["lo"]) for c in curves.values() if c["lo"]] + [median])
    hi = max([max(c["hi"]) for c in curves.values() if c["hi"]] + [median])
    span = max(hi - lo, 20.0)
    step = 10.0 if span <= 150 else 25.0
    lo = np.floor((lo - 0.12 * span) / step) * step
    hi = np.ceil((hi + 0.12 * span) / step) * step
    return float(max(lo, n_min)), float(min(hi, n_max))


def plot(stats, ylim, out):
    fig, ax = plt.subplots()
    for (lab, c), col, mk in zip(stats["curves"].items(), (C_A, C_B), ("o", "s")):
        x, m = np.asarray(c["x"]), np.asarray(c["mean"])
        ax.plot(x, m, "-" + mk, color=col, ms=3.4, lw=1.3, label=lab)
        ax.fill_between(x, c["lo"], c["hi"], color=col, alpha=0.18, lw=0)
    ax.axhline(stats["median"], color="0.5", lw=0.7, ls=(0, (2, 2)))
    ax.set_xlabel("state feature (normalized within its own range)")
    ax.set_ylabel(r"selected blocklength $n_k$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(*ylim)
    ax.legend(loc="best", frameon=True, framealpha=0.93, edgecolor="0.75",
              fancybox=False, borderpad=0.35)
    fig.tight_layout(); fig.savefig(out + ".pdf"); plt.close(fig)


def plot_map(NK, SINR, DEM, out, nb=6):
    xe = np.unique(np.quantile(SINR, np.linspace(0, 1, nb + 1)))
    ye = np.unique(np.quantile(DEM, np.linspace(0, 1, nb + 1)))
    Z = np.full((len(ye) - 1, len(xe) - 1), np.nan)
    for i in range(len(ye) - 1):
        for j in range(len(xe) - 1):
            sel = ((SINR >= xe[j]) & (SINR <= xe[j + 1] if j == len(xe) - 2 else SINR < xe[j + 1]) &
                   (DEM >= ye[i]) & (DEM <= ye[i + 1] if i == len(ye) - 2 else DEM < ye[i + 1]))
            if sel.sum() >= 5:
                Z[i, j] = NK[sel].mean()
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    im = ax.pcolormesh(xe, 10 ** ye, Z, cmap="viridis", shading="flat")
    ax.set_yscale("log")
    ax.set_xlabel("post-SIC SINR (dB)")
    ax.set_ylabel(r"computation demand $L_kc_k$ (cycles)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label(r"mean selected blocklength $n_k$")
    fig.tight_layout(); fig.savefig(out + "_map.pdf"); plt.close(fig)
    return dict(sinr_edges=xe.tolist(), demand_edges=(10 ** ye).tolist(),
                mean_n=[[None if np.isnan(v) else float(v) for v in row] for row in Z])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt")
    ap.add_argument("--arm", default="FedPPO-QR")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--out", default="fig_conditional_nk")
    ap.add_argument("--ylim", type=float, nargs=2, default=None)
    ap.add_argument("--full-range", action="store_true")
    ap.add_argument("--replot", action="store_true")
    a = ap.parse_args()

    import sys; sys.path.insert(0, ".")
    from config import full_config
    cfg = full_config()

    if a.replot:
        stats = json.load(open(a.out + ".json"))
    else:
        seeds = a.seeds or cfg.seeds
        NK, SINR, DEM, SID = collect(a.ckpt, a.arm, seeds, a.episodes or cfg.eval_episodes, cfg)
        varies = np.std(NK) > 0
        rs, rd = per_seed_r(SINR, NK, SID), per_seed_r(DEM, NK, SID)
        stats = dict(
            arm=a.arm, n=int(NK.size), seeds=int(seeds),
            median=float(np.median(NK)),
            iqr=[float(np.percentile(NK, 25)), float(np.percentile(NK, 75))],
            iqr_frac_of_range=float((np.percentile(NK, 75) - np.percentile(NK, 25))
                                    / (cfg.n_max - cfg.n_min)),
            r_sinr=float(np.corrcoef(SINR, NK)[0, 1]) if varies else float("nan"),
            r_demand=float(np.corrcoef(DEM, NK)[0, 1]) if varies else float("nan"),
            r_sinr_per_seed=rs, r_demand_per_seed=rd,
        )
        curves = {}
        for x, lab in ((SINR, "post-SIC SINR"), (DEM, r"computation demand $L_kc_k$")):
            xr = (x - x.min()) / max(x.max() - x.min(), 1e-12)
            c, m, lo, hi = binned(xr, NK, SID)
            curves[lab] = dict(x=c.tolist(), mean=m.tolist(), lo=lo.tolist(), hi=hi.tolist())
        stats["curves"] = curves
        stats["map"] = plot_map(NK, SINR, DEM, a.out)

    if a.full_range:
        ylim = (float(cfg.n_min), float(cfg.n_max))
    elif a.ylim:
        ylim = tuple(a.ylim)
    else:
        ylim = auto_ylim(stats["curves"], stats["median"], cfg.n_min, cfg.n_max)
    stats["ylim"] = list(ylim)
    plot(stats, ylim, a.out)
    json.dump(stats, open(a.out + ".json", "w"), indent=2)

    print(f"{stats['arm']}: n={stats['n']}  median {stats['median']:.1f}  "
          f"IQR [{stats['iqr'][0]:.1f}, {stats['iqr'][1]:.1f}]  "
          f"= {100*stats['iqr_frac_of_range']:.1f}% of the admissible range")
    print(f"  pearson r  vs post-SIC SINR {stats['r_sinr']:+.3f}"
          f"   vs computation demand {stats['r_demand']:+.3f}")
    for key, lab in (("r_sinr_per_seed", "post-SIC SINR"), ("r_demand_per_seed", "computation demand")):
        v = np.asarray(stats.get(key, []), float)
        if v.size >= 2:
            from scipy import stats as st
            h = st.t.ppf(0.975, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size)
            print(f"  per-seed r vs {lab}: mean {v.mean():+.3f} [{v.mean() - h:+.3f}, {v.mean() + h:+.3f}] over {v.size} seeds")
    print(f"  y-axis {ylim[0]:.0f} to {ylim[1]:.0f}")
    print(f"wrote {a.out}.pdf, {a.out}_map.pdf and {a.out}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
