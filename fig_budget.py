from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import stats as st

from paper_style import apply, METHOD, COL_W

plt = apply()
CKPT = Path("ckpt")


def curve(name, sd):
    return np.asarray(json.load(open(CKPT / f"curve_{name}_{sd}.json"))["reward"], float)


def main() -> int:
    res = json.load(open("results/budget.json"))
    labs = sorted(res, key=lambda s: int(s[:-1]))
    mults = [int(s[:-1]) for s in labs]
    seeds = len(res[labs[0]])
    name = {1: "FedPPO-QR"} | {m: f"budget{m}x" for m in mults if m != 1}

    longest = max(mults)
    base = len(curve(name[1], 0)) if 1 in mults else len(curve(name[longest], 0)) // longest
    ref = np.array([curve(name[longest], sd) for sd in range(seeds)])
    for m in mults:
        if m == longest:
            continue
        c = np.array([curve(name[m], sd) for sd in range(seeds)])
        gap = float(np.max(np.abs(c - ref[:, :c.shape[1]])))
        print(f"{base * m} rounds: largest difference from the first {c.shape[1]} rounds "
              f"of the {base * longest}-round run: {gap:.2e}")

    qr = METHOD["FedPPO-QR"]
    fig, (ax, axv) = plt.subplots(2, 1, figsize=(COL_W, 3.3),
                                  gridspec_kw=dict(height_ratios=[1.1, 1.0], hspace=0.55))
    x = np.arange(1, ref.shape[1] + 1)
    mu, sd_ = ref.mean(0), ref.std(0)
    ax.plot(x, mu, color=qr["color"], lw=1.1)
    ax.fill_between(x, mu - sd_, mu + sd_, color=qr["color"], alpha=0.18, lw=0)
    for m in mults:
        if m != longest:
            ax.axvline(base * m, color="0.35", ls=":", lw=0.9)
    ax.set_xlim(0, x[-1]); ax.set_xlabel("Communication round"); ax.set_ylabel("Training reward")

    rounds = np.array([base * m for m in mults])
    V = np.array([[r["viol_total"] for r in res[l]] for l in labs])
    for j in range(seeds):
        axv.plot(rounds, V[:, j], color="0.6", lw=0.7, marker="o", ms=2.6, mfc="white", mew=0.6, zorder=2)
    m_ = V.mean(1)
    h_ = st.t.ppf(0.975, seeds - 1) * V.std(1, ddof=1) / np.sqrt(seeds)
    axv.errorbar(rounds, m_, yerr=h_, color=qr["color"], marker=qr["marker"], ms=4.2, lw=1.2,
                 capsize=2.5, zorder=3)
    pad = 0.08 * (rounds[-1] - rounds[0])
    axv.set_xticks(rounds); axv.set_xlim(rounds[0] - pad, rounds[-1] + pad)
    axv.set_ylim(0, None)
    axv.set_xlabel("Training rounds"); axv.set_ylabel("Total violation (%)")
    for a_, lab in ((ax, "(a)"), (axv, "(b)")):
        a_.set_title(lab, fontsize=8, loc="left", pad=3)
    fig.align_ylabels((ax, axv))
    fig.savefig("fig_budget.pdf"); plt.close(fig)

    print(f"\n{'rounds':>8}{'mean total %':>14}{'95% half':>10}  per seed")
    for r, m, h, row in zip(rounds, m_, h_, V):
        print(f"{r:8d}{m:14.2f}{h:10.2f}  " + "  ".join(f"{v:.2f}" for v in row))
    print(f"reward over the {x[-1]}-round run: {mu[0]:.3f} -> {mu[-1]:.3f} (mean over {seeds} seeds)")
    print("wrote fig_budget.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
