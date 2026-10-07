from dataclasses import dataclass, asdict, replace
from typing import Tuple
import os


@dataclass
class Config:
    name: str = "full"

    n_abs: int = 5
    devices_per_abs: Tuple[int, int] = (4, 8)
    max_devices: int = 8

    fc_hz: float = 2e9
    bandwidth_hz: float = 10e6
    abs_height_m: float = 100.0
    horiz_dist_m: Tuple[float, float] = (50.0, 200.0)
    plos_a: float = 9.61
    plos_b: float = 0.16
    eta_los_db: float = 1.0
    eta_nlos_db: float = 20.0
    shadow_sigma_db: float = 4.0
    channel_rho: float = 0.9
    noise_psd_dbm_hz: float = -174.0
    pmax_dbm: float = 23.0

    n_min: int = 150
    n_max: int = 900
    blocklength: int = 300
    packet_bits: Tuple[int, int] = (200, 400)
    n_harq: int = 4
    t_harq_s: float = 0.125e-3

    fmax_hz: float = 8e9
    cpu_floor: float = 0.20
    comp_intensity: Tuple[float, float] = (500.0, 1500.0)
    ctrl_interval_s: float = 5e-3
    deadline_s: float = 1e-3
    latency_cap_mult: float = 3.0

    freeze_blocklength: float = 0.0
    freeze_rho: float = 0.0
    drop_blocklength_head: bool = False
    adaptive_alpha: bool = False
    alpha_min: float = 0.50
    alpha_max: float = 0.995
    # alpha_source, bootstrap_timelimit, tail_source, fixed_eta and clients_per_round keep
    # their defaults in every reported run.
    # alpha_source: "rollout" = previous round's rollout violation rate (adaptive-alpha schedule);
    #               "greedy"  = total violation rate of the greedy global model on the
    #                           validation environment after the previous round.
    #               "client"  = per client: the total violation rate of the greedy global
    #                           model on validation episodes of that client's environment.
    alpha_source: str = "rollout"
    client_val_episodes: int = 6
    # bootstrap_timelimit: an episode end is a truncation, not a terminal state
    # (Pardo et al., ICML 2018): bootstrap with the critic at s_{t+1} and restart the
    # GAE recursion at each episode boundary.
    bootstrap_timelimit: bool = False
    # tail_source: "return"    = u_t from the return tail, Eq. (20);
    #              "violation" = u_t from the share of violating tasks in the interval,
    #                            the task-level CVaR tail at the aligned level alpha*.
    tail_source: str = "return"
    # fixed_eta: rescale lambda*beta in every update by (1 - cvar_alpha) / mean(tail), so the
    # mean weight stays 1 + lambda*beta*(1 - cvar_alpha) and eta = 1/6 for any alpha_r.
    fixed_eta: bool = False
    # clients_per_round: 0 = full participation; S < n_abs samples S clients per round
    # uniformly without replacement.
    clients_per_round: int = 0

    # kappa_violation: weight of the share of violating tasks subtracted from the reward;
    # 0 except in the violation-penalty runs (run_ablations.py kappa).
    kappa_violation: float = 0.0
    agg_rule: str = "fedavg"

    cvar_alpha: float = 0.95
    lambda_cvar: float = 1.0
    cvar_boost: float = 4.0

    n_quantiles: int = 200
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    hidden: int = 256
    local_epochs: int = 4
    minibatch: int = 256
    episodes_per_round: int = 24
    steps_per_episode: int = 24
    fed_rounds: int = 150
    weight_clip: float = 3.0

    warm_start_epochs: int = 0

    seeds: int = 10
    eval_episodes: int = 150
    device: str = "cpu"

    def as_dict(self):
        return asdict(self)


def smoke_config() -> Config:
    return apply_env(Config(
        name="smoke",
        n_abs=3,
        devices_per_abs=(4, 4),
        max_devices=4,
        n_quantiles=50,
        hidden=64,
        local_epochs=3,
        episodes_per_round=6,
        fed_rounds=40,
        seeds=2,
        eval_episodes=60,
    ))


def full_config() -> Config:
    cfg = Config(name="full")
    if os.environ.get("FEDPPO_QUICK", "0") == "1":
        cfg = replace(cfg, name="quick", fed_rounds=2, episodes_per_round=2,
                      seeds=3, eval_episodes=2)
    return apply_env(cfg)


def apply_env(cfg: Config) -> Config:
    if os.environ.get("FEDPPO_T_HARQ_MS"):
        cfg = replace(cfg, t_harq_s=float(os.environ["FEDPPO_T_HARQ_MS"]) * 1e-3)
    check(cfg)
    return cfg


def worst_case_latency(cfg: Config) -> float:
    t_tx = cfg.n_max / cfg.bandwidth_hz + (cfg.n_harq - 1) * cfg.t_harq_s
    f_min = cfg.cpu_floor * cfg.fmax_hz / cfg.devices_per_abs[1]
    t_comp = cfg.packet_bits[1] * cfg.comp_intensity[1] / f_min
    return t_tx + t_comp


def check(cfg: Config) -> None:
    if cfg.alpha_source not in ("rollout", "greedy", "client"):
        raise ValueError(f"alpha_source must be 'rollout', 'greedy' or 'client', got {cfg.alpha_source!r}")
    if cfg.tail_source not in ("return", "violation"):
        raise ValueError(f"tail_source must be 'return' or 'violation', got {cfg.tail_source!r}")
    if not 0 <= cfg.clients_per_round <= cfg.n_abs:
        raise ValueError("clients_per_round must lie in [0, n_abs]")
    if cfg.t_harq_s < cfg.n_max / cfg.bandwidth_hz:
        raise ValueError(
            f"t_HARQ = {cfg.t_harq_s * 1e3:.4f} ms is shorter than the longest "
            f"codeword n_max / W = {cfg.n_max / cfg.bandwidth_hz * 1e3:.4f} ms")
    if worst_case_latency(cfg) >= cfg.ctrl_interval_s:
        raise ValueError(
            f"a task can last {worst_case_latency(cfg) * 1e3:.3f} ms, longer than the "
            f"control interval T_c = {cfg.ctrl_interval_s * 1e3:.1f} ms")


def describe(cfg: Config) -> str:
    return (f"Section III model  t_HARQ={cfg.t_harq_s * 1e3:g} ms  N_HARQ={cfg.n_harq}  "
            f"P_k={cfg.pmax_dbm:g} dBm  T_max={cfg.deadline_s * 1e3:g} ms  "
            f"T_c={cfg.ctrl_interval_s * 1e3:g} ms  kappa={cfg.kappa_violation:g}")


if __name__ == "__main__":
    import json
    c = full_config()
    print(describe(c))
    print(f"longest possible task latency {worst_case_latency(c) * 1e3:.3f} ms")
    print(json.dumps(c.as_dict(), indent=2, default=str))
