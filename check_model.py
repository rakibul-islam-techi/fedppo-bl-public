import math
import numpy as np

from config import full_config, describe
from sim_core import AerialEdgeEnv
from rl import heuristic_action, heuristic_adaptive_n_action, random_action

results = []


def report(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")


def Q(x):
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def Qinv(p):
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if Q(mid) > p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def reference_step(cfg, env, action, gain):
    D, n = cfg.max_devices, env.n_dev
    rho = [1.0 / (1.0 + math.exp(-action[k])) for k in range(n)]
    if cfg.freeze_rho > 0:
        rho = [cfg.freeze_rho] * n
    q = list(action[D:2 * D][:n])
    pos = {k: sorted(range(n), key=lambda u: q[u]).index(k) for k in range(n)}
    fl = list(action[2 * D:3 * D][:n])
    m = max(fl)
    e = [math.exp(x - m) for x in fl]
    f = [(cfg.cpu_floor / n + (1 - cfg.cpu_floor) * x / sum(e)) * cfg.fmax_hz for x in e]
    nb = [min(max(round(cfg.n_min + (cfg.n_max - cfg.n_min) / (1.0 + math.exp(-action[3 * D + k]))),
                  cfg.n_min), cfg.n_max) for k in range(n)]
    W = cfg.bandwidth_hz
    N0W = 10 ** ((cfg.noise_psd_dbm_hz + 10 * math.log10(W)) / 10.0)
    P = 10 ** (cfg.pmax_dbm / 10.0)
    out = []
    for k in range(n):
        interf = sum(P * gain[u] for u in range(n) if pos[u] > pos[k])
        g1 = rho[k] * P * gain[k] / ((1 - rho[k]) * P * gain[k] + interf + N0W)
        g2 = (1 - rho[k]) * P * gain[k] / (interf + N0W)
        L = env.Lk[k]
        L1 = min(max(round(rho[k] * L), 1), L - 1)
        L2 = L - L1
        eps = []
        for g, Ls in ((g1, L1), (g2, L2)):
            V = 1 - (1 + g) ** -2
            R = Ls / nb[k]
            eps.append(Q(math.sqrt(nb[k] / V) * (math.log2(1 + g) - R) / math.log2(math.e)))
        out.append(dict(g1=g1, g2=g2, L1=L1, L2=L2, R1=L1 / nb[k], R2=L2 / nb[k], eps=eps,
                        f=f[k], n=nb[k], pos=pos[k], rho=rho[k]))
    return out


def check_policy(cfg, label, policy, seeds, episodes):
    W, dl, th, N = cfg.bandwidth_hz, cfg.deadline_s, cfg.t_harq_s, cfg.n_harq
    cap = cfg.latency_cap_mult * dl
    worst = dict(sinr=0.0, eps=0.0, ttx=0.0, tcomp=0.0, lat=0.0, bits=0)
    bad = dict(perm=0, cpu=0, rho=0, n=0, attempts=0, drop=0, violation=0, overlap=0)
    eA = oA = vA = 0.0
    tasks = 0
    for sd in seeds:
        env = AerialEdgeEnv(cfg, 0, sd)
        for _ in range(episodes):
            env.reset()
            for _ in range(cfg.steps_per_episode):
                a = policy(env)
                gain = env.gain.copy()
                ref = reference_step(cfg, env, a, gain)
                _, _, _, info = env.step(a)
                n = env.n_dev
                if sorted(int(x) for x in info["sic_order"]) != list(range(n)):
                    bad["perm"] += 1
                if not (np.all(info["f_alloc"] > 0) and info["f_alloc"].sum() <= cfg.fmax_hz * (1 + 1e-12)):
                    bad["cpu"] += 1
                for k, r in enumerate(ref):
                    tasks += 1
                    if not 0.0 < info["rho"][k] < 1.0:
                        bad["rho"] += 1
                    nk = info["n_block"][k]
                    if nk != r["n"] or nk != int(nk) or not cfg.n_min <= nk <= cfg.n_max:
                        bad["n"] += 1
                    worst["sinr"] = max(worst["sinr"], abs(info["gamma1"][k] / r["g1"] - 1),
                                        abs(info["gamma2"][k] / r["g2"] - 1))
                    worst["bits"] = max(worst["bits"], abs(info["L1"][k] - r["L1"]) + abs(info["L2"][k] - r["L2"]))
                    worst["eps"] = max(worst["eps"], abs(info["eps1"][k] - r["eps"][0]),
                                       abs(info["eps2"][k] - r["eps"][1]))
                    A = info["attempts"][k]
                    if not (1 <= A[0] <= N and 1 <= A[1] <= N):
                        bad["attempts"] += 1
                    ttx = max(r["L1"] / (W * r["R1"]) + (A[0] - 1) * th,
                              r["L2"] / (W * r["R2"]) + (A[1] - 1) * th)
                    worst["ttx"] = max(worst["ttx"], abs(info["t_tx"][k] - ttx))
                    tcomp = env.Lk[k] * env.ck[k] / r["f"]
                    worst["tcomp"] = max(worst["tcomp"], abs(info["t_comp"][k] - tcomp))
                    dropped = bool(info["dropped"][k])
                    if dropped and max(A) != N:
                        bad["drop"] += 1
                    T = ttx if dropped else ttx + tcomp
                    worst["lat"] = max(worst["lat"], abs(info["latencies"][k] - min(T, cap)))
                    if bool(info["violated"][k]) != (dropped or T > dl):
                        bad["violation"] += 1
                    if T >= cfg.ctrl_interval_s:
                        bad["overlap"] += 1
                    for e in r["eps"]:
                        ea = sum(e ** j for j in range(N))
                        eA += ea
                        vA += sum((2 * j + 1) * e ** j for j in range(N)) - ea ** 2
                    oA += A[0] + A[1]
    tag = f"{label} ({tasks} tasks)"
    report(f"{tag}: SINRs equal Eqs. (4)-(5) with every device of M_i transmitting", worst["sinr"] < 1e-9,
           f"max relative diff {worst['sinr']:.1e}")
    report(f"{tag}: L_k1 + L_k2 = L_k with L_k1 = round(rho_k L_k)", worst["bits"] == 0)
    report(f"{tag}: block-error probability equals Eq. (6) with V from Eq. (7) and R_ks/W_i = L_ks/n_k",
           worst["eps"] < 1e-9, f"max abs diff {worst['eps']:.1e}")
    report(f"{tag}: T_tx equals Eq. (10)", worst["ttx"] < 1e-15, f"max abs diff {worst['ttx']:.1e} s")
    report(f"{tag}: T_comp equals Eq. (12)", worst["tcomp"] < 1e-15, f"max abs diff {worst['tcomp']:.1e} s")
    report(f"{tag}: latency equals Eq. (13) (airtime only for a dropped task)", worst["lat"] < 1e-15,
           f"max abs diff {worst['lat']:.1e} s")
    for k, v in bad.items():
        name = {"perm": "decoding order is a permutation of M_i", "cpu": "CPU allocation satisfies Eq. (11)",
                "rho": "0 < rho_k < 1", "n": "n_k is an integer in [n_min, n_max]",
                "attempts": "1 <= A_ks <= N_HARQ", "drop": "a dropped task used N_HARQ attempts",
                "violation": "violation = dropped or T_k > T_max",
                "overlap": "every task ends before the next control interval"}[k]
        report(f"{tag}: {name}", v == 0, f"{v} violations")
    z = (oA - eA) / math.sqrt(max(vA, 1e-12))
    report(f"{tag}: attempt counts follow Eq. (9)", abs(z) < 4.0,
           f"observed {oA:.0f}, expected {eA:.1f}, z = {z:+.2f}")


def harq_distribution(cfg):
    from sim_core import harq_attempts
    rng = np.random.default_rng(3)
    N = cfg.n_harq
    for e in (0.05, 0.3, 0.7):
        A, drop = harq_attempts(rng, np.full(400_000, e), N)
        pmf = [(1 - e) * e ** (a - 1) for a in range(1, N)] + [e ** (N - 1)]
        emp = [np.mean(A == a) for a in range(1, N + 1)]
        ok = all(abs(x - y) < 0.004 for x, y in zip(emp, pmf)) and abs(drop.mean() - e ** N) < 0.003
        report(f"Eq. (9) with eps = {e}: distribution of A = min(G, N_HARQ) and Pr(G > N_HARQ)", ok,
               "P(A=a) " + ", ".join(f"{x:.4f}/{y:.4f}" for x, y in zip(emp, pmf)) +
               f"; drop {drop.mean():.4f}/{e ** N:.4f}")


def eq8_inverse(cfg):
    from sim_core import block_error
    rng = np.random.default_rng(5)
    worst = 0.0
    for _ in range(2000):
        g = 10 ** rng.uniform(-1.0, 2.0)
        n = int(rng.integers(cfg.n_min, cfg.n_max + 1))
        eps = 10 ** rng.uniform(-6, -0.5)
        V = 1 - (1 + g) ** -2
        R = math.log2(1 + g) - math.sqrt(V / n) * Qinv(eps) * math.log2(math.e)
        if R <= 0:
            continue
        worst = max(worst, abs(float(block_error(np.array(g), np.array(R), n)) / eps - 1))
    report("Eq. (8) is the inverse of Eq. (6)", worst < 1e-6, f"max relative diff {worst:.1e}")


def main():
    cfg = full_config()
    print(describe(cfg))
    harq_distribution(cfg)
    eq8_inverse(cfg)
    check_policy(cfg, "static heuristic", lambda e: heuristic_action(e, cfg), [1, 2], 3)
    check_policy(cfg, "adaptive-blocklength heuristic", lambda e: heuristic_adaptive_n_action(e, cfg), [3], 2)
    check_policy(cfg, "random actions", lambda e: random_action(e, cfg), [4, 5], 3)
    print(f"\n{sum(results)} of {len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
