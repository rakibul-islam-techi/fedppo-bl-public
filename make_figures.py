"""Figures 2 to 7 of the paper from a finished run folder.

    python make_figures.py --run runs/sec3_tharq0.125ms --out runs/sec3_tharq0.125ms/figs

Fig. 2 fig2_reward, Fig. 3 fig3_sweep_grid, Fig. 4 fig4_blocklength, Fig. 5 fig5_budget,
Fig. 6 fig6_ccdf, Fig. 7 fig7_sic_priority (PDF).
"""
from __future__ import annotations
import argparse, json, os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy import stats as st

SINGLE = (3.5, 2.45)
DOUBLE_W = 7.16

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "TeX Gyre Termes"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "lines.linewidth": 1.1, "lines.markersize": 3.5,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
    "legend.frameon": True, "legend.framealpha": 0.95, "legend.edgecolor": "0.75", "legend.fancybox": False,
    "savefig.bbox": None, "pdf.fonttype": 42, "ps.fonttype": 42,
})

METHOD = {
    "FedPPO-QR":            ("FedPPO-QR", "#0072B2", "-", "o"),
    "FedPPO-Mean":          ("FedPPO-Mean", "#E69F00", "--", "s"),
    "Centralized-QR":       ("Non-federated PPO-QR", "#56B4E9", "-.", "D"),
    "FedAvg-DDQN":          ("FedAvg-DDQN", "#009E73", ":", "^"),
    "Heuristic":            ("Static heuristic", "#CC79A7", (0, (3, 1, 1, 1)), "v"),
    "Heuristic-adaptive-n": ("Adaptive heuristic", "#000000", (0, (5, 1.5)), "P"),
}
DEADLINE = "#D55E00"
N_MIN, N_MAX, N_STATIC = 150, 900, 525


def single():
    return plt.subplots(figsize=SINGLE, layout="constrained")


def reward(res, out):
    fig, ax = single()
    cur = res["curves"]
    for k in ("FedPPO-QR", "FedPPO-Mean", "Centralized-QR", "FedAvg-DDQN"):
        if k not in cur:
            continue
        lab, c, ls, mk = METHOD[k]
        y = np.asarray(cur[k]["reward_mean"], float); e = np.asarray(cur[k]["reward_std"], float)
        x = np.arange(1, len(y) + 1)
        ax.plot(x, y, color=c, ls=ls, marker=mk, markevery=max(1, len(y) // 8), ms=3, mew=0, label=lab, zorder=3)
        ax.fill_between(x, y - e, y + e, color=c, alpha=0.13, lw=0, zorder=2)
    ax.set_xlim(1, len(y))
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Training reward")
    ax.legend(loc="lower right", ncol=2, columnspacing=1.0, handlelength=2.2)
    fig.savefig(os.path.join(out, "fig2_reward.pdf")); plt.close(fig)


def ccdf(per_seed, out):
    fig, ax = single()
    grid = np.linspace(0.0, 1.2, 900)
    for k in ("FedPPO-QR", "FedPPO-Mean", "Centralized-QR", "FedAvg-DDQN", "Heuristic", "Heuristic-adaptive-n"):
        if k not in per_seed:
            continue
        lab, c, ls, _ = METHOD[k]
        x = np.sort(np.concatenate(per_seed[k]))
        y = 1.0 - np.searchsorted(x, grid, side="right") / x.size
        ax.plot(grid, np.maximum(y, 1e-6), color=c, ls=ls, label=lab, zorder=3)
    ax.axvline(1.0, color=DEADLINE, ls=(0, (4, 2)), lw=1.0, zorder=2)
    ax.text(0.985, 1.3e-3, "deadline", rotation=90, ha="right", va="bottom", color=DEADLINE, fontsize=6.5)
    ax.set_yscale("log")
    ax.set_xlim(0, 1.2); ax.set_ylim(1e-3, 1.05)
    ax.set_xlabel(r"Latency $t$ (ms)")
    ax.set_ylabel(r"$\Pr(\widetilde{T}>t)$")
    ax.legend(loc="upper right", bbox_to_anchor=(0.975, 0.985), fontsize=6, handlelength=2.4,
              borderpad=0.35, labelspacing=0.25)
    fig.savefig(os.path.join(out, "fig6_ccdf.pdf")); plt.close(fig)


def sic(res, out):
    g = np.asarray(res["fig6"]["gain_ratio"], float)
    q = np.asarray(res["fig6"]["demand_ratio"], float)
    Z = np.asarray(res["fig6"]["Z"], float)
    fig, ax = single()
    lim = max(abs(Z.min()), abs(Z.max()), 1e-6)
    im = ax.pcolormesh(g, q, Z, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
                       shading="auto", rasterized=True, zorder=1)
    if Z.min() < 0 < Z.max():
        ax.contour(g, q, Z, levels=[0.0], colors="black", linewidths=1.4, zorder=4)
    ax.grid(False)
    ax.set_xlabel(r"Channel-gain ratio $|h_1|^2/|h_2|^2$")
    ax.set_ylabel(r"Demand ratio $L_1c_1/(L_2c_2)$")
    bb = dict(boxstyle="round,pad=0.22", fc="white", ec="0.5", lw=0.5, alpha=0.95)
    for (x, y, ha, va), z in (((0.97, 0.96, "right", "top"), Z[-1, -1]), ((0.03, 0.04, "left", "bottom"), Z[0, 0])):
        ax.text(x, y, "device 1 decoded first" if z < 0 else "device 1 decoded last", transform=ax.transAxes,
                ha=ha, va=va, fontsize=6.3, bbox=bb, zorder=6)
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label(r"Score difference $q_1-q_2$")
    cb.ax.tick_params(labelsize=6.5); cb.outline.set_linewidth(0.6)
    fig.savefig(os.path.join(out, "fig7_sic_priority.pdf")); plt.close(fig)


def grid(sw, out):
    titles = {"power": "(a) Transmit power", "cpu_capacity": "(b) CPU capacity", "cpu_allocation": "(c) CPU allocation"}
    colors = ["#0072B2", "#D55E00", "#009E73"]
    styles = [("-", "o"), ("--", "s"), (":", "^")]
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_W, 3.3), sharex=True, sharey="row", layout="constrained")
    for j, factor in enumerate(titles):
        ax, axm = axes[0, j], axes[1, j]
        group = [r for r in sw if r["factor"] == factor]
        for r, c, (ls, mk) in zip(group, colors, styles):
            n = np.array([x["n_k"] for x in r["sweep"]])
            tot = np.array([x["viol_total"] for x in r["sweep"]])
            mean = np.array([x["mean_ms"] for x in r["sweep"]])
            lab = (r["label"].replace("P_k", "$P_k$").replace("F_max", r"$F_i^{\max}$")
                   .replace("own CPU allocation", "own split").replace("equal CPU allocation", "equal split"))
            ax.plot(n, tot, ls, marker=mk, color=c, mfc="white", mew=0.8, ms=3, label=lab)
            axm.plot(n, mean, ls, marker=mk, color=c, mfc="white", mew=0.8, ms=3)
            qr = r["trained"].get("FedPPO-QR")
            if qr:
                qn = np.mean([x["n_k"] for x in qr])
                ax.plot([qn], [np.mean([x["viol_total"] for x in qr])], "*", ms=7, color=c, mec="black", mew=0.4, zorder=5)
                axm.plot([qn], [np.mean([x["mean_ms"] for x in qr])], "*", ms=7, color=c, mec="black", mew=0.4, zorder=5)
        ax.set_yscale("log"); ax.set_xlim(N_MIN, N_MAX)
        ax.set_title(titles[factor], loc="left")
        h, l = ax.get_legend_handles_labels()
        h.append(Line2D([], [], marker="*", ls="none", ms=6, color="0.55", mec="black", mew=0.4)); l.append("FedPPO-QR")
        ax.legend(h, l, loc="upper right", fontsize=6)
        axm.set_xlabel(r"Blocklength $n_k$")
    axes[0, 0].set_ylabel("Violation rate (%)")
    axes[1, 0].set_ylabel("Mean recorded latency (ms)")
    fig.savefig(os.path.join(out, "fig3_sweep_grid.pdf")); plt.close(fig)


def budget(run, out):
    path = os.path.join(run, "results", "budget.json")
    if not os.path.exists(path):
        print("results/budget.json not found, Fig. 5 skipped")
        return
    res = json.load(open(path, encoding="utf-8-sig"))
    labs = sorted(res, key=lambda s: int(s[:-1]))
    mults = [int(s[:-1]) for s in labs]
    seeds = len(res[labs[0]])
    longest = max(mults)
    ref = np.array([json.load(open(os.path.join(run, "ckpt", f"curve_budget{longest}x_{sd}.json")))["reward"]
                    for sd in range(seeds)], float)
    base = ref.shape[1] // longest
    lab, c, _, mk = METHOD["FedPPO-QR"]
    fig, (ax, axv) = plt.subplots(1, 2, figsize=SINGLE, layout="constrained", gridspec_kw=dict(width_ratios=[1.25, 1.0]))
    x = np.arange(1, ref.shape[1] + 1)
    mu, sd_ = ref.mean(0), ref.std(0)
    ax.plot(x, mu, color=c, lw=1.0)
    ax.fill_between(x, mu - sd_, mu + sd_, color=c, alpha=0.18, lw=0)
    for m in mults:
        if m != longest:
            ax.axvline(base * m, color="0.35", ls=":", lw=0.8)
    ax.set_xlim(0, x[-1]); ax.set_xticks([0, 200, 400, 600])
    ax.set_xlabel("Communication round"); ax.set_ylabel("Training reward")
    rounds = np.array([base * m for m in mults])
    V = np.array([[r["viol_total"] for r in res[l]] for l in labs])
    for j in range(seeds):
        axv.plot(rounds, V[:, j], color="0.6", lw=0.7, marker="o", ms=2.4, mfc="white", mew=0.6, zorder=2)
    h_ = st.t.ppf(0.975, seeds - 1) * V.std(1, ddof=1) / np.sqrt(seeds)
    axv.errorbar(rounds, V.mean(1), yerr=h_, color=c, marker=mk, ms=3.6, lw=1.1, capsize=2.2, zorder=3)
    pad = 0.1 * (rounds[-1] - rounds[0])
    axv.set_xticks(rounds); axv.set_xlim(rounds[0] - pad, rounds[-1] + pad); axv.set_ylim(0, None)
    axv.set_xlabel("Training rounds"); axv.set_ylabel("Total violation (%)")
    for a_, t in ((ax, "(a)"), (axv, "(b)")):
        a_.set_title(t, loc="left")
    fig.savefig(os.path.join(out, "fig5_budget.pdf")); plt.close(fig)


def t_ci(v):
    v = np.asarray(v, float)
    return v.mean(), st.t.ppf(0.975, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size)


def policy(arms, cond, out):
    fig = plt.figure(figsize=(DOUBLE_W, 2.45), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.0, 1.05])
    ax, axr, axm = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    rows = [k for k in ("FedPPO-QR", "FedPPO-Mean", "Centralized-QR", "FedAvg-DDQN", "Heuristic-adaptive-n") if k in arms]
    for i, k in enumerate(rows):
        y0 = len(rows) - 1 - i
        lab, c, _, mk = METHOD[k]
        for o, r in zip(np.linspace(-0.32, 0.32, len(arms[k])), arms[k]):
            lo, hi = r["n_iqr"]
            ax.plot([lo, hi], [y0 + o, y0 + o], color=c, lw=0.9, alpha=0.75, solid_capstyle="butt", zorder=2)
            ax.plot(r["median_n"], y0 + o, ls="none", marker=mk, ms=2.8, color=c, mfc="white", mew=0.7, zorder=3)
    ax.axvline(N_STATIC, color=METHOD["Heuristic"][1], ls="--", lw=0.9)
    ax.text(N_STATIC + 8, -0.55, "static", color=METHOD["Heuristic"][1], fontsize=6.5, va="bottom")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([METHOD[k][0].replace("Non-federated PPO-QR", "Non-federated") for k in rows][::-1])
    ax.set_xlim(N_MIN, N_MAX); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(r"Selected blocklength $n_k$"); ax.grid(axis="y", visible=False)

    labels, j = [], 0
    for k in ("FedPPO-QR", "FedPPO-Mean"):
        lab, c, _, mk = METHOD[k]
        for key, feat in (("r_sinr_per_seed", "SINR"), ("r_demand_per_seed", "demand")):
            v = np.asarray(cond[k][key], float); y0 = 3 - j
            axr.plot(v, np.full(v.size, y0) + np.linspace(-0.18, 0.18, v.size), ls="none", marker=mk, ms=2.8,
                     color=c, mfc="white", mew=0.6)
            m, h = t_ci(v)
            axr.errorbar(m, y0, xerr=h, fmt=mk, ms=4, color=c, ecolor=c, elinewidth=1.0, capsize=2.0)
            labels.append(f"{'QR' if k == 'FedPPO-QR' else 'Mean'}, {feat}"); j += 1
    axr.axvline(0, color="0.3", lw=0.7)
    axr.set_yticks(range(4)); axr.set_yticklabels(labels[::-1])
    axr.set_xlim(-1, 1); axr.set_ylim(-0.6, 3.6)
    axr.set_xlabel(r"Within-seed correlation with $n_k$"); axr.grid(axis="y", visible=False)

    mp = cond["FedPPO-QR"]["map"]
    xe = np.asarray(mp["sinr_edges"], float); ye = np.asarray(mp["demand_edges"], float)
    Z = np.array([[np.nan if v is None else v for v in row] for row in mp["mean_n"]], float)
    im = axm.pcolormesh(xe, ye, Z, cmap="viridis", vmin=N_MIN, vmax=N_MAX, shading="flat", rasterized=True)
    axm.set_yscale("log"); axm.grid(False)
    axm.set_xlabel("Post-SIC SINR (dB)"); axm.set_ylabel(r"Demand $L_kc_k$ (cycles)")
    cb = fig.colorbar(im, ax=axm, pad=0.02, fraction=0.06)
    cb.set_label(r"Mean $n_k$, FedPPO-QR"); cb.ax.tick_params(labelsize=6.5)
    for a_, t in ((ax, "(a)"), (axr, "(b)"), (axm, "(c)")):
        a_.set_title(t, loc="left")
    fig.savefig(os.path.join(out, "fig4_blocklength.pdf")); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=".", help="run folder with results.json, main_comparison.json, sweep_grid.json")
    ap.add_argument("--pkl", default=None, help="recorded latencies (default: <run>/results/latencies_per_seed.pkl)")
    ap.add_argument("--out", default="figs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = lambda f: os.path.join(a.run, f)
    res = json.load(open(R("results.json"), encoding="utf-8-sig"))
    reward(res, a.out)
    sic(res, a.out)
    grid(json.load(open(R("sweep_grid.json"), encoding="utf-8-sig")), a.out)
    arms = json.load(open(R("main_comparison.json"), encoding="utf-8-sig"))
    cond = {k: json.load(open(R(f"fig_conditional_nk_{t}.json"))) for k, t in (("FedPPO-QR", "qr"), ("FedPPO-Mean", "mean"))}
    policy(arms, cond, a.out)
    budget(a.run, a.out)
    pkl = a.pkl or R(os.path.join("results", "latencies_per_seed.pkl"))
    with open(pkl, "rb") as f:
        per_seed = {k: [np.asarray(v, float) * 1e3 for v in runs] for k, runs in pickle.load(f).items()}
    ccdf(per_seed, a.out)
    print("wrote", ", ".join(sorted(os.listdir(a.out))), "to", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
