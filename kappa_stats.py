from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_stats import _paired, interval


def col(rows, key, sub=None):
    return [r[sub][key] if sub else r[key] for r in rows]


def show(name, x, ref, ref_name, key="viol_total", sub=None):
    p = _paired(col(x, key, sub), col(ref, key, sub))
    ps = "<0.001" if p["p"] < 0.001 else f"{p['p']:.3f}"
    print(f"  {name} minus {ref_name}: {p['delta']:+.2f} [{p['lo']:+.2f}, {p['hi']:+.2f}] p={ps} MDE {p['mde']:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa-file", default="results/kappa.json")
    ap.add_argument("--main", default="main_comparison.json")
    a = ap.parse_args()
    kap = json.load(open(a.kappa_file, encoding="utf-8-sig"))
    main_ = json.load(open(a.main, encoding="utf-8-sig"))
    ref = {"qr": "FedPPO-QR", "mean": "FedPPO-Mean"}
    for tag, rows in sorted(kap.items()):
        n = len(rows)
        arm = ref[tag.rsplit("_", 1)[1]]
        base = main_[arm][:n]
        m, lo, hi = interval(col(rows, "viol_total"))
        print(f"\n{tag} ({n} seeds): total {m:.2f}% [{max(lo, 0):.2f}, {hi:.2f}]  "
              f"recorded {np.mean(col(rows, 'viol_reported')):.2f}%  drop {np.mean(col(rows, 'drop_rate')):.2f}%  "
              f"median n_k {np.median(col(rows, 'median_n')):.0f}  "
              f"delivered mean {np.mean(col(rows, 'mean', 'delivered')):.3f} ms  "
              f"p99 {np.mean(col(rows, 'p99', 'delivered')):.3f} ms")
        show(tag, rows, base, f"{arm} (kappa 0)")
        show(tag, rows, main_["Heuristic"][:n], "static heuristic")
        show(tag, rows, main_["Heuristic-adaptive-n"][:n], "adaptive heuristic")
        show(tag, rows, base, f"{arm} (kappa 0), delivered p99 ms", "p99", "delivered")
        show(tag, rows, base, f"{arm} (kappa 0), delivered mean ms", "mean", "delivered")
    tags = sorted(kap)
    for i in range(0, len(tags) - 1):
        for j in range(i + 1, len(tags)):
            if tags[i].rsplit("_", 1)[0] == tags[j].rsplit("_", 1)[0]:
                print()
                show(tags[j], kap[tags[j]], kap[tags[i]], tags[i])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
