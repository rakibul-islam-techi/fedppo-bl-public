"""Table III and the CVaR_0.95 column of Table II from the recorded latencies of
every evaluated task (pooled_latencies.py).

    python cvar_recompute.py --pkl results/latencies_per_seed.pkl

CVaR follows Problem (14): the mean of the worst 5% of outcomes by probability
mass, with ties at the 95th percentile split. A dropped task is identified by
its recorded latency n/W + (N_HARQ - 1) t_HARQ with integer n in [150, 900].
"""
import argparse, pickle
import numpy as np
from scipy import stats
from evaluate_arms import cvar_ru

W, T_HARQ, N_HARQ = 1e7, 0.125e-3, 4


def is_drop(x_s):
    n = np.round((x_s - (N_HARQ - 1) * T_HARQ) * W)
    return (np.abs(x_s - (n / W + (N_HARQ - 1) * T_HARQ)) < 1e-12) & (n >= 150) & (n <= 900)


def drops_in_tail(x, dr, alpha=0.95):
    o = np.argsort(x, kind="stable"); xs, ds = x[o], dr[o]
    n = x.size; k = n * (1 - alpha)
    var = xs[n - int(np.ceil(k))]
    above = xs > var + 1e-12; at = np.abs(xs - var) <= 1e-12
    need = k - above.sum()
    d_tail = (ds & above).sum() + (need * (ds & at).sum() / at.sum() if at.any() else 0.0)
    return 100.0 * d_tail / k, var


def t_ci(v):
    v = np.asarray(v); m = v.mean()
    h = stats.t.ppf(0.975, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size)
    return m, m - h, m + h


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pkl", default="results/latencies_per_seed.pkl")
    a = ap.parse_args()
    d = pickle.load(open(a.pkl, "rb"))
    deliv = {}
    print("Table II, CVaR_0.95 of delivered tasks, mean over seeds [95% t-interval]; Table III, all tasks pooled")
    print(f"{'method':24s}{'Table II CVaR':>24s}{'VaR':>8s}{'CVaR':>8s}{'max drop':>10s}{'v_tot %':>9s}{'drops in tail %':>17s}")
    for meth, seeds in d.items():
        xs = [np.asarray(s, float) for s in seeds]
        to_ms = 1.0 if xs[0].max() > 0.1 else 1e3            # stored in ms or in s
        xs = [s * to_ms for s in xs]                          # now in ms
        drops = [is_drop(s * 1e-3) for s in xs]               # is_drop works in seconds
        deliv[meth] = np.array([cvar_ru(s[~dr]) for s, dr in zip(xs, drops)])
        pooled = np.concatenate(xs)
        share, var = drops_in_tail(pooled, np.concatenate(drops))
        m, lo, hi = t_ci(deliv[meth])
        dr_all = np.concatenate(drops)
        mx = pooled[dr_all].max() if dr_all.any() else float("nan")
        print(f"{meth:24s}{f'{m:.3f} [{lo:.3f}, {hi:.3f}]':>24s}{var:8.3f}{cvar_ru(pooled):8.3f}{mx:10.3f}"
              f"{100 * dr_all.mean():9.2f}{share:17.2f}")
    if "FedPPO-QR" in deliv and "FedPPO-Mean" in deliv:
        dd = deliv["FedPPO-QR"] - deliv["FedPPO-Mean"]; m, lo, hi = t_ci(dd)
        se = dd.std(ddof=1) / np.sqrt(dd.size); tost = abs(m) + stats.t.ppf(0.95, dd.size - 1) * se
        p = stats.ttest_rel(deliv["FedPPO-QR"], deliv["FedPPO-Mean"]).pvalue
        print(f"QR - Mean delivered CVaR: {m:.3f} [{lo:.3f}, {hi:.3f}], p = {p:.3f}, TOST margin {tost:.3f} ms "
              f"({100 * tost / deliv['FedPPO-QR'].mean():.1f}% of QR)")


if __name__ == "__main__":
    main()
