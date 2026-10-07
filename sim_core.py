from __future__ import annotations
import numpy as np
from scipy.special import ndtr
from config import Config


def aerial_path_loss_db(horiz_dist_m: np.ndarray, cfg: Config) -> np.ndarray:
    h = cfg.abs_height_m
    r = np.maximum(horiz_dist_m, 1.0)
    d3d = np.sqrt(r ** 2 + h ** 2)
    theta_deg = np.degrees(np.arctan2(h, r))
    p_los = 1.0 / (1.0 + cfg.plos_a * np.exp(-cfg.plos_b * (theta_deg - cfg.plos_a)))
    c = 3e8
    fspl = 20 * np.log10(d3d) + 20 * np.log10(cfg.fc_hz) + 20 * np.log10(4 * np.pi / c)
    return p_los * (fspl + cfg.eta_los_db) + (1 - p_los) * (fspl + cfg.eta_nlos_db)


def sample_large_scale(rng, n_dev: int, cfg: Config) -> np.ndarray:
    r = rng.uniform(cfg.horiz_dist_m[0], cfg.horiz_dist_m[1], size=n_dev)
    pl_db = aerial_path_loss_db(r, cfg)
    shadow_db = rng.normal(0.0, cfg.shadow_sigma_db, size=n_dev)
    return 10 ** (-(pl_db + shadow_db) / 10.0)


class GaussMarkovFading:
    def __init__(self, rng, n_dev: int, rho: float):
        self.rho = float(rho)
        s = 1.0 / np.sqrt(2.0)
        self.h = rng.normal(0, s, n_dev) + 1j * rng.normal(0, s, n_dev)

    def step(self, rng) -> np.ndarray:
        s = 1.0 / np.sqrt(2.0)
        w = rng.normal(0, s, len(self.h)) + 1j * rng.normal(0, s, len(self.h))
        self.h = self.rho * self.h + np.sqrt(1.0 - self.rho ** 2) * w
        return np.abs(self.h) ** 2


def sic_sinr(gain, P, rho, order, noise):
    rx = gain * P
    later = np.array([rx[order > order[k]].sum() for k in range(len(gain))])
    gamma1 = rho * rx / ((1.0 - rho) * rx + later + noise)
    gamma2 = (1.0 - rho) * rx / (later + noise)
    return gamma1, gamma2


def dispersion(gamma):
    return 1.0 - (1.0 + gamma) ** -2


def block_error(gamma, R, n):
    V = np.maximum(dispersion(gamma), 1e-300)
    return ndtr(-np.sqrt(n / V) * (np.log2(1.0 + gamma) - R) / np.log2(np.e))


def stream_bits(L, rho):
    L1 = np.clip(np.round(rho * L), 1.0, L - 1.0)
    return L1, L - L1


def harq_attempts(rng, eps, n_harq):
    ok = rng.random(eps.shape + (n_harq,)) < (1.0 - eps)[..., None]
    first = np.where(ok.any(axis=-1), ok.argmax(axis=-1) + 1, n_harq + 1)
    return np.minimum(first, n_harq), first > n_harq


class AerialEdgeEnv:
    STATE_DIM = 9
    DEV_FEAT = 4

    def __init__(self, cfg: Config, abs_id: int, seed: int):
        self.cfg = cfg
        self.abs_id = abs_id
        self.rng = np.random.default_rng(seed)
        self.rng_dec = np.random.default_rng(np.random.SeedSequence([seed, 7]))
        self.noise = 10 ** ((cfg.noise_psd_dbm_hz + 10 * np.log10(cfg.bandwidth_hz)) / 10.0)
        self.pmax = 10 ** (cfg.pmax_dbm / 10.0)
        self._reset_devices()
        self.reset()

    def _reset_devices(self):
        cfg = self.cfg
        lo, hi = cfg.devices_per_abs
        self.n_dev = int(self.rng.integers(lo, hi + 1))
        self.Lk = self.rng.integers(cfg.packet_bits[0], cfg.packet_bits[1] + 1,
                                    size=self.n_dev).astype(float)
        self.ck = self.rng.uniform(*cfg.comp_intensity, size=self.n_dev)
        self.workload = self.Lk * self.ck
        self.Pk = np.full(self.n_dev, self.pmax)
        self.rate_req = self.Lk / cfg.blocklength

    def reset(self):
        self.large_scale = sample_large_scale(self.rng, self.n_dev, self.cfg)
        self.fading = GaussMarkovFading(self.rng, self.n_dev, self.cfg.channel_rho)
        self.gain = self.large_scale * self.fading.step(self.rng)
        self._recent_lat = [self.cfg.deadline_s]
        self._recent_drop = 0.0
        self._last_lat = np.full(self.n_dev, self.cfg.deadline_s)
        return self._state()

    def device_features(self):
        D = self.cfg.max_devices
        feats = np.zeros((D, self.DEV_FEAT), dtype=np.float32)
        mask = np.zeros(D, dtype=np.float32)
        for k in range(self.n_dev):
            feats[k] = [np.log10(self.gain[k] + 1e-30),
                        np.log10(self.workload[k]),
                        self.Lk[k] / self.cfg.packet_bits[1],
                        self._last_lat[k] / self.cfg.deadline_s]
            mask[k] = 1.0
        return np.nan_to_num(feats, nan=0.0, posinf=5.0, neginf=-5.0), mask

    def _state(self):
        g = self.gain
        recent = np.array(self._recent_lat[-64:])
        feas = np.mean(np.log2(1 + g * self.Pk / self.noise) >= self.rate_req)
        s = np.array([
            np.log10(np.mean(g) + 1e-30),
            np.log10(np.std(g) + 1e-30),
            np.log10(np.mean(self.workload)),
            np.mean(recent) / self.cfg.deadline_s,
            np.quantile(recent, 0.9) / self.cfg.deadline_s,
            np.mean(recent > self.cfg.deadline_s),
            feas,
            self.n_dev / self.cfg.max_devices,
            self._recent_drop,
        ], dtype=np.float32)
        return np.nan_to_num(s, nan=0.0, posinf=5.0, neginf=-5.0)

    def _decode_action(self, action_vec):
        cfg = self.cfg
        D = cfg.max_devices
        n = self.n_dev
        rho = 1.0 / (1.0 + np.exp(-action_vec[:D][:n]))
        if cfg.freeze_rho > 0.0:
            rho = np.full(n, float(cfg.freeze_rho))
        q_scores = action_vec[D:2 * D][:n]
        f_logits = action_vec[2 * D:3 * D][:n]
        sic_order = np.argsort(np.argsort(q_scores))
        e = np.exp(f_logits - np.max(f_logits))
        f_alloc = (cfg.cpu_floor / n + (1.0 - cfg.cpu_floor) * e / np.sum(e)) * cfg.fmax_hz
        if len(action_vec) >= 4 * D:
            sig = 1.0 / (1.0 + np.exp(-action_vec[3 * D:4 * D][:n]))
            n_block = cfg.n_min + (cfg.n_max - cfg.n_min) * sig
            if cfg.freeze_blocklength > 0.0:
                n_block = np.full(n, float(cfg.freeze_blocklength))
        else:
            if cfg.freeze_blocklength <= 0.0:
                raise ValueError("drop_blocklength_head=True requires freeze_blocklength > 0")
            n_block = np.full(n, float(cfg.freeze_blocklength))
        n_block = np.clip(np.round(n_block), cfg.n_min, cfg.n_max)
        return rho, q_scores, sic_order, f_alloc, n_block

    def physical_layer(self, rho, sic_order, n_block):
        gamma1, gamma2 = sic_sinr(self.gain, self.Pk, rho, sic_order, self.noise)
        L1, L2 = stream_bits(self.Lk, rho)
        R1, R2 = L1 / n_block, L2 / n_block
        eps1, eps2 = block_error(gamma1, R1, n_block), block_error(gamma2, R2, n_block)
        return dict(gamma1=gamma1, gamma2=gamma2, L1=L1, L2=L2, R1=R1, R2=R2,
                    eps1=eps1, eps2=eps2)

    def step(self, action_vec):
        cfg = self.cfg
        rho, q_scores, sic_order, f_alloc, n_block = self._decode_action(action_vec)
        phy = self.physical_layer(rho, sic_order, n_block)
        W = cfg.bandwidth_hz
        eps = np.stack([phy["eps1"], phy["eps2"]], axis=1)
        A, stream_dropped = harq_attempts(self.rng_dec, eps, cfg.n_harq)
        t_tx = np.maximum(phy["L1"] / (W * phy["R1"]) + (A[:, 0] - 1) * cfg.t_harq_s,
                          phy["L2"] / (W * phy["R2"]) + (A[:, 1] - 1) * cfg.t_harq_s)
        t_comp = self.Lk * self.ck / f_alloc
        dropped = stream_dropped.any(axis=1)
        T = np.where(dropped, t_tx, t_tx + t_comp)
        dl = cfg.deadline_s
        latencies = np.minimum(T, cfg.latency_cap_mult * dl)
        violated = dropped | (T > dl)
        viol_rate = float(violated.mean())
        reward = -float(latencies.mean() / dl + cfg.kappa_violation * viol_rate)

        self._recent_lat.extend(latencies.tolist())
        self._recent_lat = self._recent_lat[-256:]
        self._recent_drop = float(dropped.mean())
        self._last_lat = latencies.copy()
        self.gain = self.large_scale * self.fading.step(self.rng)

        eps_msg = 1.0 - (1.0 - phy["eps1"]) * (1.0 - phy["eps2"])
        info = {
            "latencies": latencies,
            "dropped": dropped,
            "violated": violated,
            "violation_rate": viol_rate,
            "drop_rate": float(dropped.mean()),
            "viol_delivered": float(np.mean(T[~dropped] > dl)) if (~dropped).any() else 0.0,
            "mean_eps": float(np.mean(eps_msg)),
            "n_block": n_block,
            "attempts": A,
            "t_tx": t_tx,
            "t_comp": t_comp,
            "f_alloc": f_alloc,
            "rho": rho,
            "sic_order": sic_order,
            **phy,
        }
        return self._state(), reward, False, info


if __name__ == "__main__":
    from config import full_config, describe
    from rl import heuristic_action
    cfg = full_config()
    env = AerialEdgeEnv(cfg, abs_id=0, seed=0)
    env.reset()
    print(describe(cfg))
    print("devices:", env.n_dev, " packet bits:", env.Lk.astype(int).tolist())
    for _ in range(5):
        s2, r, d, info = env.step(heuristic_action(env, cfg))
        print(f"reward {r:+.3f}  latency ms {np.round(info['latencies'] * 1e3, 3).tolist()}  "
              f"dropped {int(info['dropped'].sum())}  mean eps {info['mean_eps']:.2e}")
