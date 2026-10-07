import json
from pathlib import Path

import numpy as np
from scipy import optimize, stats


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def mde_factor(n, power=0.8):
    if n < 3:
        return float("nan")
    df = n - 1
    tc = stats.t.ppf(0.975, df)

    def achieved(d):
        nc = d * np.sqrt(n)
        return stats.nct.sf(tc, df, nc) + stats.nct.cdf(-tc, df, nc)

    hi = 0.5
    while achieved(hi) < power:
        hi *= 1.5
    return optimize.brentq(lambda d: achieved(d) - power, 1e-3, hi)


def paired(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return _paired(a, b)


def _paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    n = d.size
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n)
    t95, t90 = stats.t.ppf(0.975, n - 1), stats.t.ppf(0.95, n - 1)
    return dict(delta=d.mean(), lo=d.mean() - t95 * se, hi=d.mean() + t95 * se,
                p=2 * stats.t.sf(abs(d.mean() / se), n - 1), mde=mde_factor(n) * sd,
                tost=max(abs(d.mean() - t90 * se), abs(d.mean() + t90 * se)), sd=sd)


def interval(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size)
    return x.mean(), x.mean() - h, x.mean() + h


def col(rows, key, group=None):
    return [r[group][key] if group else r[key] for r in rows]


def seeds_needed(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    sd = d.std(ddof=1)
    return next((n for n in range(5, 500) if mde_factor(n) * sd <= abs(d.mean())), None)


def fmt_p(p):
    return "n/a" if not np.isfinite(p) else f"{p:.4f}"


def pooled_table(res):
    print("\nPooled latency statistics (ms), all seeds pooled")
    print(f"{'method':16s}{'N':>10}{'mean':>8}{'p95':>8}{'p99':>8}{'p99.9':>8}"
          f"{'CVaR95':>8}{'viol%':>8}{'dl pct':>8}{'alpha*':>8}")
    for name, t in res["tail_table"].items():
        v = 100 * t["violation"]
        print(f"{name:16s}{t['n_samples']:>10d}{t['mean']:8.3f}{t['p95']:8.3f}{t['p99']:8.3f}"
              f"{t['p999']:8.3f}{t['cvar99']:8.3f}{v:8.2f}{100 - v:8.2f}{1 - v / 100:8.3f}")


def seed_table(main):
    print("\nSeed-level statistics: mean over seeds [95% t-interval]")
    for name, rows in main.items():
        parts = []
        for key in ("mean", "p99", "p999", "cvar"):
            m, lo, hi = interval(col(rows, key, "delivered"))
            parts.append(f"{key} {m:.3f} [{lo:.3f}, {hi:.3f}]")
        for key in ("drop_rate", "viol_total", "viol_reported"):
            m, lo, hi = interval(col(rows, key))
            parts.append(f"{key} {m:.2f} [{lo:.2f}, {hi:.2f}]")
        print(f"{name:22s} " + "  ".join(parts))


def paired_table(arms, ref):
    print("\nPaired comparison against FedPPO-QR (positive delta = more violations)")
    print(f"{'arm':22s}{'total':>8}{'delta':>9}{'p':>8}{'MDE':>7}   {'recorded':>8}{'delta':>9}{'p':>8}{'MDE':>7}")
    rows = []
    for name, r in arms.items():
        if name == "FedPPO-QR":
            continue
        t = paired(col(r, "viol_total"), col(ref, "viol_total"))
        c = paired(col(r, "viol_reported"), col(ref, "viol_reported"))
        rows.append((np.mean(col(r, "viol_total")), name, t, np.mean(col(r, "viol_reported")), c))
    for vt, name, t, vr, c in sorted(rows, key=lambda x: -x[0]):
        print(f"{name:22s}{vt:8.2f}{t['delta']:+9.2f}{fmt_p(t['p']):>8}{t['mde']:7.2f}   "
              f"{vr:8.2f}{c['delta']:+9.2f}{fmt_p(c['p']):>8}{c['mde']:7.2f}")
    sig_t = sum(t["p"] < 0.05 for _, _, t, _, _ in rows)
    sig_r = sum(c["p"] < 0.05 for _, _, _, _, c in rows)
    print(f"significant at 0.05: {sig_t} of {len(rows)} on the total rate, {sig_r} on the recorded-latency rate")


def text_numbers(main, ablations):
    qr, mean_, heur, adapt = (main[k] for k in ("FedPPO-QR", "FedPPO-Mean", "Heuristic", "Heuristic-adaptive-n"))
    print("\nTotal minus recorded-latency rate (failed tasks recorded at or below the deadline)")
    for name in ("FedPPO-QR", "Heuristic", "Heuristic-adaptive-n"):
        rows = main[name]
        print(f"  {name:22s} {np.mean(col(rows, 'viol_total')) - np.mean(col(rows, 'viol_reported')):.2f}")
    print("\nMedian per-attempt task error probability 1-(1-eps_k1)(1-eps_k2), averaged over devices in each interval")
    for name in main:
        print(f"  {name:22s} {np.median(col(main[name], 'median_eps')):.2e}")

    print("\nMedian selected blocklength in evaluation (median over seeds)")
    for name in main:
        if "median_n" in main[name][0]:
            print(f"  {name:22s} {np.median(col(main[name], 'median_n')):.0f}")

    print("\nQuantile critic against mean critic, delivered tasks (QR minus Mean)")
    for key in ("p99", "cvar", "p999", "mean"):
        r = paired(col(qr, key, "delivered"), col(mean_, key, "delivered"))
        share = 100 * round(r["tost"], 3) / round(float(np.mean(col(qr, key, "delivered"))), 3)
        print(f"  {key:5s} delta {r['delta']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] p={r['p']:.3f} "
              f"MDE {r['mde']:.3f}  equivalence bound {r['tost']:.3f} ({share:.1f}% of the mean)")

    print("\nHeuristic against FedPPO-QR")
    r = paired(col(heur, "p99", "delivered"), col(qr, "p99", "delivered"))
    print(f"  delivered p99, heuristic minus QR: {r['delta']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}] p={r['p']:.4f}")
    r = paired(col(heur, "viol_reported"), col(qr, "viol_reported"))
    print(f"  recorded rate, heuristic minus QR: {r['delta']:.2f} p={r['p']:.4f}")
    r = paired(col(qr, "viol_total"), col(heur, "viol_total"))
    print(f"  total rate, QR minus heuristic: {r['delta']:.2f} p={r['p']:.4f} MDE {r['mde']:.2f}")

    print("\nAdaptive-blocklength heuristic")
    r = paired(col(heur, "viol_total"), col(adapt, "viol_total"))
    print(f"  heuristic minus adaptive: {r['delta']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}] p={r['p']:.4f} MDE {r['mde']:.2f}")
    for name, rows in (("FedPPO-QR", qr), ("FedPPO-Mean", mean_)):
        r = paired(col(rows, "viol_total"), col(adapt, "viol_total"))
        n = seeds_needed(col(rows, "viol_total"), col(adapt, "viol_total"))
        print(f"  {name} minus adaptive: {r['delta']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}] p={r['p']:.4f} "
              f"MDE {r['mde']:.2f}; seeds for MDE below the difference: {n if n else '>500'}")

    print("\nFederated arms against FedPPO-QR, total violations")
    for name in ("Centralized-QR", "FedPPO-Mean", "FedAvg-DDQN"):
        if name in main:
            r = paired(col(main[name], "viol_total"), col(qr, "viol_total"))
            print(f"  {name:22s} minus QR: {r['delta']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] p={r['p']:.4f}")

    print("\nRatios of total violation rate to FedPPO-QR")
    base = np.mean(col(qr, "viol_total"))
    for name, rows in list(main.items()) + list(ablations.items()):
        if name not in ("FedPPO-QR", "FedAvg-DDQN"):
            print(f"  {name:22s} {np.mean(col(rows, 'viol_total')) / base:.2f}")

    print("\nFixed-blocklength arms")
    for nk in ("block350", "block500", "block800"):
        if nk not in ablations:
            continue
        rows = ablations[nk]
        r = paired(col(rows, "viol_total"), col(heur, "viol_total"))
        m = paired(col(rows, "mean", "all_tasks"), col(qr, "mean", "all_tasks"))
        q = paired(col(rows, "p99", "all_tasks"), col(qr, "p99", "all_tasks"))
        print(f"  {nk}: minus heuristic {r['delta']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}] p={r['p']:.2g};  "
              f"all-task mean vs QR {m['delta']:+.3f} (p={m['p']:.2g}), p99 {q['delta']:+.3f} (p={q['p']:.2g})")

    if "block500" in ablations and "block500_nohead" in ablations:
        r = paired(col(ablations["block500_nohead"], "viol_total"), col(ablations["block500"], "viol_total"))
        print(f"  head removed minus head kept, n_k = 500, total: {r['delta']:+.2f} "
              f"[{r['lo']:+.2f}, {r['hi']:+.2f}] p={r['p']:.4f}")

    for tag in ("local", "local_shared", "agg_median", "agg_trimmed", "scale10", "rho05"):
        if tag in ablations:
            r = paired(col(ablations[tag], "viol_total"), col(qr, "viol_total"))
            print(f"  {tag:12s} minus QR, total: {r['delta']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] p={r['p']:.4f}")
    if "local" in ablations and "local_shared" in ablations:
        r = paired(col(ablations["local"], "viol_total"), col(ablations["local_shared"], "viol_total"))
        print(f"  initialization effect, local minus local_shared, total: {r['delta']:+.2f} "
              f"[{r['lo']:+.2f}, {r['hi']:+.2f}] p={r['p']:.4f}")
        r = paired(col(ablations["local_shared"], "viol_total"), col(qr, "viol_total"))
        print(f"  federation effect, local_shared minus QR, total: {r['delta']:+.2f} "
              f"[{r['lo']:+.2f}, {r['hi']:+.2f}] p={r['p']:.4f}")
    lr_arms = sorted((k for k in ablations if k.startswith("pooled_lr")), key=lambda k: float(k[9:]))
    if lr_arms and "Centralized-QR" in main:
        print("\n  Non-federated PPO-QR learning-rate sweep, total violation, minus FedPPO-QR")
        for name, rows in [(k, ablations[k]) for k in lr_arms] + [("pooled_lr0.0003 (main)", main["Centralized-QR"])]:
            r = paired(col(rows, "viol_total"), col(qr, "viol_total"))
            print(f"    {name:24s} {np.mean(col(rows, 'viol_total')):6.2f}%  {r['delta']:+.2f} "
                  f"[{r['lo']:+.2f}, {r['hi']:+.2f}] p={r['p']:.4f}")

    print("\nTasks above the 3 T_max recording cap, mean over seeds")
    for name, rows in main.items():
        if "cap_rate" in rows[0]:
            print(f"  {name:22s} {np.mean(col(rows, 'cap_rate')):.4f}%")

    print("\nSelected blocklength (conditional-blocklength queries)")
    arms = {}
    for tag in ("qr", "mean"):
        f = Path(f"fig_conditional_nk_{tag}.json")
        if f.exists():
            arms[tag] = load(f)
    for tag, d in arms.items():
        print(f"  {d['arm']:12s} median {d['median']:.1f}, IQR [{d['iqr'][0]:.1f}, {d['iqr'][1]:.1f}] "
              f"({100 * d['iqr_frac_of_range']:.1f}% of the range), r(post-SIC SINR) {d['r_sinr']:+.3f}, r(demand) {d['r_demand']:+.3f}")
    if len(arms) == 2:
        q, m = arms["qr"], arms["mean"]
        wq, wm = q["iqr"][1] - q["iqr"][0], m["iqr"][1] - m["iqr"][0]
        print(f"  median gap {100 * (q['median'] - m['median']) / q['median']:.1f}%, IQR ratio {wq / wm:.1f}")


def adaptive_alpha(doc):
    print("\nAdaptive risk level, alpha_r = 1 - total violation rate of the round r-1 rollouts "
          "(environment seeds 10100 + s; qr_fixed and mean_adaptive are the main checkpoints)")
    for name, rows in doc.items():
        print(f"  {name:14s} alpha_final {np.mean(col(rows, 'alpha_final')):.3f}  "
              f"1 - recorded {1 - np.mean(col(rows, 'viol_reported')) / 100:.3f}  "
              f"1 - total {1 - np.mean(col(rows, 'viol_total')) / 100:.3f}  "
              f"drop {np.mean(col(rows, 'drop_rate')):.2f}  total {np.mean(col(rows, 'viol_total')):.2f}")
    for name in ("qr_adaptive", "mean_adaptive"):
        rows = doc[name]
        err = np.mean(col(rows, "alpha_final")) - (1 - np.mean(col(rows, "viol_total")) / 100)
        print(f"  {name:14s} alpha_final minus (1 - total violation of the evaluated policy): {err:+.3f}")
        if "alpha_clip_rounds" in rows[0]:
            print(f"  {name:14s} rounds at the alpha clip: {np.mean(col(rows, 'alpha_clip_rounds')):.1f} "
                  f"of {rows[0]['rounds']:.0f}")
    qa, qf, ma = doc["qr_adaptive"], doc["qr_fixed"], doc["mean_adaptive"]
    if len(qa) != len(qf):
        print("  qr_adaptive is incomplete; paired tests skipped")
        return
    for key in ("p99", "p999"):
        r = paired(col(qa, key, "delivered"), col(ma, key, "delivered"))
        print(f"  adaptive QR minus adaptive Mean, delivered {key}: {r['delta']:+.3f} p={r['p']:.3f}")
    r = paired(col(qa, "viol_total"), col(qf, "viol_total"))
    print(f"  adaptive minus fixed QR, total: {r['delta']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}] p={r['p']:.4f}")
    for key in ("p99", "p999"):
        r = paired(col(qa, key, "delivered"), col(qf, key, "delivered"))
        print(f"  adaptive minus fixed QR, delivered {key}: {r['delta']:+.3f} p={r['p']:.4f}")


def corr(fn, x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return "undefined (constant)"
    return f"{fn(x, y)[0]:+.3f}"


def sweep(rows):
    nk = np.array(col(rows, "n_k"), float)
    rec, tot, drop = (np.array(col(rows, k), float) for k in ("viol_reported", "viol_total", "drop_rate"))
    i = int(np.argmin(nk))
    print("\nBlocklength sweep")
    print(f"  n_k = {nk[i]:.0f}: recorded {rec[i]:.2f}, total {tot[i]:.2f}, drop {drop[i]:.2f}")
    print(f"  Spearman with n_k: recorded {corr(stats.spearmanr, nk, rec)}, total {corr(stats.spearmanr, nk, tot)}, "
          f"drop {corr(stats.spearmanr, nk, drop)}; Pearson(total - recorded, drop) {corr(stats.pearsonr, tot - rec, drop)}")
    j = int(np.argmin(tot))
    print(f"  lowest total at n_k = {nk[j]:.0f} ({tot[j]:.2f}%); recorded maximum {rec.max():.2f}%")


def training_curves(res):
    print("\nTraining reward, mean over seeds")
    for name in res["curves"]:
        n = len(list(Path("ckpt").glob(f"curve_{name}_*.json")))
        if n:
            print(f"  {name:16s} per-seed curve files in ckpt/: {n}")
    for name, c in res["curves"].items():
        r = np.asarray(c["reward_mean"], float)
        print(f"  {name:16s} first round {r[0]:.3f}, last round {r[-1]:.3f} "
              f"({100 * (r[-1] - r[0]) / abs(r[0]):+.1f}%), best round {int(np.argmax(r)) + 1}")


def sic_surface(res):
    g = np.asarray(res["fig6"]["gain_ratio"], float)
    q = np.asarray(res["fig6"]["demand_ratio"], float)
    z = np.asarray(res["fig6"]["Z"], float)
    dq = np.gradient(z, q, axis=0).mean()
    dg = np.gradient(z, g, axis=1).mean()
    print("\nSIC priority surface, FedPPO-QR, mean over "
          f"{res['fig6'].get('seeds', 1)} seeds (per-seed sign agreement "
          f"{100 * res['fig6'].get('sign_agreement', float('nan')):.1f}%)")
    print(f"  mean sensitivity: {dq:+.3f} per unit computation-demand ratio, {dg:+.3f} per unit gain ratio")


def same_blocklength(fixed, ablations, heur):
    print("\nSame blocklength, static versus learned other controls, total violation, same environments")
    for key, rows in fixed.items():
        nk = int(key[1:])
        line = f"  n_k {nk}: static controls {np.mean(col(rows, 'viol_total')):.2f}%"
        tag = f"block{nk}"
        if tag in ablations and len(ablations[tag]) == len(rows):
            r = paired(col(ablations[tag], "viol_total"), col(rows, "viol_total"))
            line += (f", learned controls {np.mean(col(ablations[tag], 'viol_total')):.2f}%, "
                     f"learned minus static {r['delta']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] p={fmt_p(r['p'])}")
        print(line)
    print(f"  static heuristic, n_k 525: {np.mean(col(heur, 'viol_total')):.2f}%")


def quantile_resolution(doc, qr):
    print("\nQuantile resolution M, FedPPO-QR, seed-level (M = 200 is the principal arm)")
    for tag in sorted(doc, key=lambda t: int(t[1:])):
        rows = doc[tag]
        ref = qr[:len(rows)]
        t = paired(col(rows, "viol_total"), col(ref, "viol_total"))
        d = paired(col(rows, "p99", "delivered"), col(ref, "p99", "delivered"))
        print(f"  M = {tag[1:]:>3s}: total {np.mean(col(rows, 'viol_total')):.2f}%, minus M=200 {t['delta']:+.2f} "
              f"[{t['lo']:+.2f}, {t['hi']:+.2f}] p={fmt_p(t['p'])};  delivered p99 minus M=200 "
              f"{d['delta']:+.3f} ms p={fmt_p(d['p'])}  ({len(rows)} seeds)")
    print(f"  M = 200: total {np.mean(col(qr[:3], 'viol_total')):.2f}% (seeds 0-2)")

def sweep_grid(doc):
    print("\nBlocklength sweep under other power and CPU settings")
    for g in doc:
        rows = g["sweep"]
        nk = np.array([r["n_k"] for r in rows])
        tot = np.array([r["viol_total"] for r in rows])
        rec = np.array([r["viol_reported"] for r in rows])
        mean = np.array([r["mean_ms"] for r in rows])
        i = int(np.argmin(tot))
        rho = corr(stats.spearmanr, nk, tot)
        print(f"  {g['label']:22s} best n_k {nk[i]:4.0f} ({tot[i]:.2f}%), Spearman(n_k, total) {rho}, "
              f"recorded max {rec.max():.2f}%, mean latency {mean.min():.3f}-{mean.max():.3f} ms")
    print("\nEvery method at every setting (trained at the nominal setting, not retrained;"
          " trained arms: mean [95% CI] over training seeds)")
    print(f"  {'setting':22s}{'method':22s}{'total %':>22}{'recorded %':>12}{'drop %':>9}"
          f"{'mean ms':>9}{'p99 deliv':>11}{'median n_k':>12}")
    for g in doc:
        entries = [("Heuristic", [g["static_heuristic"]]), ("Heuristic-adaptive-n", [g["adaptive_heuristic"]])]
        entries += list(g.get("trained", {}).items())
        for arm, rows in entries:
            f = lambda k: np.mean([r[k] for r in rows])
            if len(rows) >= 2:
                m, lo, hi = interval([r["viol_total"] for r in rows])
                tot = f"{m:.2f} [{lo:.2f}, {hi:.2f}]"
            else:
                tot = f"{rows[0]['viol_total']:.2f}"
            print(f"  {g['label']:22s}{arm:22s}{tot:>22}{f('viol_reported'):12.2f}{f('drop_rate'):9.2f}"
                  f"{f('mean_ms'):9.3f}{f('p99_delivered_ms'):11.3f}{f('n_k'):12.0f}")
    nominal = next((g for g in doc if g["factor"] == "power" and g["pmax_dbm"] == 23.0), None)
    if nominal and nominal.get("trained", {}).get("FedPPO-QR"):
        print("\n  FedPPO-QR, setting minus nominal (paired over training seeds), total violation")
        ref = [r["viol_total"] for r in nominal["trained"]["FedPPO-QR"]]
        for g in doc:
            rows = g["trained"].get("FedPPO-QR", [])
            if len(rows) != len(ref) or len(rows) < 3:
                continue
            if g["pmax_dbm"] == 23.0 and g["fmax_ghz"] == 8.0 and g["allocation"] == "own":
                continue
            r = paired([x["viol_total"] for x in rows], ref)
            print(f"    {g['label']:22s} {r['delta']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] p={fmt_p(r['p'])}")


def budget(doc):
    print("\nTraining budget, FedPPO-QR, mean over seeds")
    ref = doc["1x"]
    for lab, rows in doc.items():
        f = lambda k: np.mean([r[k] for r in rows])
        line = (f"  {lab:4s} total {f('viol_total'):.2f}%  recorded {f('viol_reported'):.2f}%  drop {f('drop_rate'):.2f}%  "
                f"median n_k {f('median_n'):.0f}  n_k IQR {np.mean([r['n_iqr'][0] for r in rows]):.0f}-"
                f"{np.mean([r['n_iqr'][1] for r in rows]):.0f}")
        if lab != "1x" and len(rows) == len(ref) and len(rows) >= 3:
            r = paired([x["viol_total"] for x in rows], [x["viol_total"] for x in ref])
            line += f"  vs 1x {r['delta']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] p={r['p']:.3f}"
        print(line)


def seeds30(doc):
    n = len(doc["FedPPO-QR"])
    print(f"\nComparisons with {n} seeds")
    for arm, rows in doc.items():
        m, lo, hi = interval(col(rows, "viol_total"))
        print(f"  {arm:22s} total {m:.2f} [{lo:.2f}, {hi:.2f}]  recorded {np.mean(col(rows, 'viol_reported')):.2f}")
    for arm in [k for k in doc if k != "FedPPO-QR"]:
        for key in ("viol_total", "viol_reported"):
            r = paired(col(doc[arm], key), col(doc["FedPPO-QR"], key))
            print(f"  {arm:22s} minus FedPPO-QR, {key:13s} {r['delta']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] "
                  f"p={r['p']:.4f} MDE {r['mde']:.2f}")


def optional(path):
    p = Path(path)
    return load(p) if p.exists() else None


def main():
    res = optional("results.json")
    if res and "config" in res:
        c = res["config"]
        print(f"Section III model  t_HARQ {1e3 * c['t_harq_s']:g} ms  N_HARQ {c['n_harq']}  "
              f"P_k {c['pmax_dbm']:g} dBm  kappa {c['kappa_violation']:g}  seeds {c['seeds']}")
    main_ = load("main_comparison.json")
    ablations = optional("results/ablations.json") or {}
    arms = dict(main_)
    arms.update(ablations)
    if res:
        pooled_table(res)
    seed_table(main_)
    paired_table(arms, main_["FedPPO-QR"])
    text_numbers(main_, ablations)
    fixed_n = optional("results/heuristic_fixed_n.json")
    if fixed_n:
        same_blocklength(fixed_n, ablations, main_["Heuristic"])
    quant = optional("results/quantile.json")
    if quant:
        quantile_resolution(quant, main_["FedPPO-QR"])
    for path, fn in (("results/adaptive_alpha.json", adaptive_alpha),
                     ("fig3_dual.json", sweep),
                     ("sweep_grid.json", sweep_grid),
                     ("results/budget.json", budget),
                     ("results/seeds30.json", seeds30)):
        doc = optional(path)
        if doc:
            fn(doc)
    if res:
        training_curves(res)
        sic_surface(res)


if __name__ == "__main__":
    main()
