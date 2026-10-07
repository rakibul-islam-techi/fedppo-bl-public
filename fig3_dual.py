from __future__ import annotations
import json
import numpy as np

from paper_style import apply, METHOD, COL_W

plt = apply()

SEEDS, EPISODES = 20, 5
N_GRID = [150, 200, 250, 300, 350, 400, 450, 500, 525, 550, 600, 650, 700, 750, 800, 850, 900]


def sweep(cfg, n_k, seeds=SEEDS, episodes=EPISODES):
    from sim_core import AerialEdgeEnv
    from rl import heuristic_fixed_n_action
    lat, drop, vio = [], [], []
    for sd in range(seeds):
        env = AerialEdgeEnv(cfg, sd % cfg.n_abs, 6000 + sd)
        for _ in range(episodes):
            env.reset()
            for _ in range(cfg.steps_per_episode):
                _, _, _, info = env.step(heuristic_fixed_n_action(env, cfg, n_k))
                lat.append(info["latencies"]); drop.append(info["dropped"]); vio.append(info["violated"])
    lat = np.concatenate(lat) * 1e3
    drop = np.concatenate(drop); vio = np.concatenate(vio)
    return dict(n_k=float(n_k),
                mean_ms=float(lat.mean()),
                viol_reported=100.0 * float(np.mean(lat > 1.0)),
                viol_total=100.0 * float(vio.mean()),
                drop_rate=100.0 * float(drop.mean()),
                n_tasks=int(lat.size))


def plot(rows, out="fig3_dual.pdf"):
    n = np.array([r["n_k"] for r in rows])
    ms = np.array([r["mean_ms"] for r in rows])
    vr = np.array([r["viol_reported"] for r in rows])
    vt = np.array([r["viol_total"] for r in rows])
    st = METHOD["Heuristic"]
    i525 = int(np.argmin(np.abs(n - 525)))
    fig, (ax, axm) = plt.subplots(2, 1, figsize=(COL_W, 3.2), sharex=True,
                                  gridspec_kw=dict(height_ratios=[1.25, 1.0], hspace=0.22))
    ax.plot(n, vt, color=st["color"], ls="-", marker="o", mfc="white", mew=0.9,
            label="total (drops counted)")
    if np.any(vr > 0):
        ax.plot(n[vr > 0], vr[vr > 0], color="0.4", ls=":", marker="s", mfc="white", mew=0.9,
                label="recorded latency only")
    ax.plot(n[i525], vt[i525], ls="none", marker="o", ms=5.5, color=st["color"], zorder=4,
            label="static heuristic")
    ax.set_yscale("log")
    ax.set_ylabel("Violation rate (%)")
    ax.legend(loc="upper right")
    axm.plot(n, ms, color=st["color"], ls="-", marker="o", mfc="white", mew=0.9)
    axm.plot(n[i525], ms[i525], ls="none", marker="o", ms=5.5, color=st["color"], zorder=4)
    axm.set_ylabel("Mean latency (ms)")
    axm.set_xlabel(r"Blocklength $n_k$ (channel uses)")
    axm.set_xlim(n.min() - 15, n.max() + 15)
    for a_, lab in ((ax, "(a)"), (axm, "(b)")):
        a_.set_title(lab, fontsize=8, loc="left", pad=2)
    fig.align_ylabels((ax, axm))
    fig.savefig(out); plt.close(fig)
    return n, ms, vr, vt


def main() -> int:
    import sys; sys.path.insert(0, ".")
    if "--replot" in sys.argv:
        rows = json.load(open("fig3_dual.json"))
    else:
        from config import full_config, describe
        cfg = full_config()
        print(describe(cfg))
        rows = [sweep(cfg, nk) for nk in N_GRID]
        json.dump(rows, open("fig3_dual.json", "w"), indent=2)
    n, ms, vr, vt = plot(rows)

    print(f"{'n_k':>7}{'mean_ms':>9}{'viol_rep':>10}{'viol_tot':>10}{'drop%':>8}")
    for r in rows:
        print(f"{r['n_k']:7.0f}{r['mean_ms']:9.4f}{r['viol_reported']:10.2f}"
              f"{r['viol_total']:10.2f}{r['drop_rate']:8.2f}")
    i_t, i_m = int(np.argmin(vt)), int(np.argmin(ms))
    print(f"\n  lowest total violation : n_k {n[i_t]:.0f} ({vt[i_t]:.2f}%)")
    print(f"  lowest mean latency    : n_k {n[i_m]:.0f} ({ms[i_m]:.4f} ms)")
    print(f"  largest recorded-latency violation rate: {vr.max():.2f}%")
    from scipy.stats import spearmanr
    print(f"  Spearman(n_k, viol_total) = {spearmanr(n, vt).statistic:+.3f}")
    print("\nwrote fig3_dual.pdf and fig3_dual.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
