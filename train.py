from __future__ import annotations
import numpy as np
import copy, json, time
from config import Config, smoke_config, full_config
from sim_core import AerialEdgeEnv
from rl import (CVaRPPOAgent, DDQNAgent, federated_average,
                heuristic_action, random_action)


KEYS = ['s', 'df', 'm', 'a', 'logp', 'r', 'done', 'v', 's2', 'vf']


def collect_ppo(agent, env, cfg):
    batch = {k: [] for k in KEYS}
    ep_rewards, viol = [], []
    for _ in range(cfg.episodes_per_round):
        s = env.reset(); df, mk = env.device_features()
        ep_r = 0.0
        for t in range(cfg.steps_per_episode):
            a, logp, v = agent.act(s, df, mk)
            s2, r, d, info = env.step(a)
            df2, mk2 = env.device_features()
            done = float(t == cfg.steps_per_episode - 1)
            for k, val in zip(KEYS, [s, df, mk, a, logp, r, done, v, s2,
                                     info['violation_rate']]):
                batch[k].append(val)
            viol.append(info['violated'])
            ep_r += r
            s, df, mk = s2, df2, mk2
        ep_rewards.append(ep_r / cfg.steps_per_episode)
    return batch, ep_rewards, np.concatenate(viol)


def eval_policy(policy_fn, cfg, seeds, n_episodes, with_total=False):
    lat, vio = [], []
    for sd in seeds:
        env = AerialEdgeEnv(cfg, 0, 10_000 + sd)
        for _ in range(n_episodes):
            s = env.reset(); df, _ = env.device_features()
            for t in range(cfg.steps_per_episode):
                a = policy_fn(env, s, df)
                s, r, d, info = env.step(a)
                df, _ = env.device_features()
                lat.append(info['latencies']); vio.append(info['violated'])
    lat = np.concatenate(lat)
    viol = float(np.mean(lat > cfg.deadline_s))
    if with_total:
        # total violation rate: drops plus late deliveries, the rate in alpha* = 1 - v_tot
        return lat, viol, float(np.concatenate(vio).mean())
    return lat, viol


def client_val_rate(policy_fn, cfg, seed, b, n_episodes):
    """Total violation rate of policy_fn on validation episodes of client b's
    environment: the device population of training client b (same construction
    seed), with channel and decoding draws from separate generators, so the
    training stream is untouched. The same episodes are used in every round."""
    env = AerialEdgeEnv(cfg, b, seed * 100 + b)
    env.rng = np.random.default_rng(np.random.SeedSequence([seed, b, 31337]))
    env.rng_dec = np.random.default_rng(np.random.SeedSequence([seed, b, 31338]))
    vio = []
    for _ in range(n_episodes):
        s = env.reset(); df, _ = env.device_features()
        for t in range(cfg.steps_per_episode):
            s, r, d, info = env.step(policy_fn(env, s, df))
            df, _ = env.device_features()
            vio.append(info['violated'])
    return float(np.concatenate(vio).mean())


def cvar(x, alpha):
    from evaluate_arms import cvar_ru
    return cvar_ru(x, alpha)


def bootstrap_ci(x, stat_fn, n_boot=500, ci=95, rng=None):
    rng = rng or np.random.default_rng(0)
    n = len(x)
    stats = [stat_fn(x[rng.integers(0, n, n)]) for _ in range(n_boot)]
    lo, hi = np.percentile(stats, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(stat_fn(x)), float(lo), float(hi)


def train_ppo(cfg, seed, federated=True, mode=None, return_all=False):
    if mode is None:
        mode = "fed" if federated else "pooled"
    assert mode in ("fed", "pooled", "local", "local_shared"), mode
    federated = (mode == "fed")
    np.random.seed(seed)
    envs = [AerialEdgeEnv(cfg, b, seed * 100 + b) for b in range(cfg.n_abs)]
    agents = [CVaRPPOAgent(cfg, seed * 100 + b) for b in range(cfg.n_abs)]
    if mode == "pooled":
        agents = [agents[0]]

    if mode in ("fed", "local_shared") and len(agents) > 1:
        pa, pc = agents[0].get_params()
        for ag in agents[1:]:
            ag.set_params(pa, pc)
    reward_curve, cvar_curve, episodes_x = [], [], []
    cum_ep = 0
    alpha_sched, viol_trace, val_trace = [], [], []
    _src = getattr(cfg, 'alpha_source', 'rollout')
    _S = int(getattr(cfg, 'clients_per_round', 0) or 0)
    _part_rng = np.random.default_rng(np.random.SeedSequence([seed, 4242]))
    part_log, alpha_client, client_val = [], [], []
    _nval = int(getattr(cfg, 'client_val_episodes', 6))
    for rnd in range(cfg.fed_rounds):

        if _src == 'client':
            if client_val:
                _ai = [float(np.clip(1.0 - v, cfg.alpha_min, cfg.alpha_max)) for v in client_val[-1]]
            else:
                _ai = [float(cfg.cvar_alpha)] * len(agents)
            _a = float(np.mean(_ai))
        else:
            _trace = val_trace if _src == 'greedy' else viol_trace
            _a = 1.0 - _trace[-1] if rnd > 0 else cfg.cvar_alpha
            _a = float(np.clip(_a, cfg.alpha_min, cfg.alpha_max))
            _ai = [_a] * len(agents)
        alpha_sched.append(_a)
        alpha_client.append(_ai)
        if getattr(cfg, 'adaptive_alpha', False):
            for _ag, _x in zip(agents, _ai):
                _ag._alpha_r = _x
        round_rewards, n_samps, params_a, params_c, round_viol = [], [], [], [], []
        if mode in ("fed", "local", "local_shared"):
            if mode == "fed" and 0 < _S < len(agents):
                _sel = np.sort(_part_rng.choice(len(agents), size=_S, replace=False))
            else:
                _sel = np.arange(len(agents))
            part_log.append(_sel.tolist())
            for env, ag in [(envs[i], agents[i]) for i in _sel]:
                batch, eprew, vio = collect_ppo(ag, env, cfg)
                round_viol.append(vio)
                ag.update(batch)

                _tf = getattr(ag, "_last_tail_frac", None)
                if _tf is not None and (rnd < 3 or rnd % 25 == 0):
                    print(f"    [audit] round {rnd} tail_frac={_tf:.4f} "
                          f"(target {1.0 - cfg.cvar_alpha:.3f})", flush=True)
                round_rewards.extend(eprew)
                n_samps.append(len(batch['r']))
                pa, pc = ag.get_params(); params_a.append(pa); params_c.append(pc)
            if mode == "fed":
                _rule = getattr(cfg, "agg_rule", "fedavg")
                agg_a = federated_average(params_a, n_samps, cfg.weight_clip, cfg.n_abs, _rule)
                agg_c = federated_average(params_c, n_samps, cfg.weight_clip, cfg.n_abs, _rule)
                for ag in agents:
                    ag.set_params(agg_a, agg_c)
        else:
            ag = agents[0]
            pooled = {k: [] for k in KEYS}
            for env in envs:
                batch, eprew, vio = collect_ppo(ag, env, cfg)
                round_viol.append(vio)
                for k in KEYS:
                    pooled[k].extend(batch[k])
                round_rewards.extend(eprew)
            ag.update(pooled)
        viol_trace.append(float(np.concatenate(round_viol).mean()))
        cum_ep += cfg.episodes_per_round * cfg.n_abs
        reward_curve.append(np.mean(round_rewards)); episodes_x.append(cum_ep)

        if rnd % max(1, cfg.fed_rounds // 20) == 0:
            print(f"      round {rnd + 1}/{cfg.fed_rounds}  reward {reward_curve[-1]:.3f}",
                  flush=True)

        lat, _, _vt = eval_policy(lambda e, s, df: agents[0].act_greedy(s, df),
                                  cfg, seeds=[seed], n_episodes=max(4, cfg.eval_episodes // 10),
                                  with_total=True)
        val_trace.append(_vt)
        if _src == 'client' and mode == "fed":
            client_val.append([client_val_rate(lambda e, s, df: agents[0].act_greedy(s, df),
                                               cfg, seed, b, _nval) for b in range(len(agents))])
        cvar_curve.append(cvar(lat, cfg.cvar_alpha) * 1e3)
    for _ag in agents:
        _ag._alpha_sched = alpha_sched
        _ag._viol_trace = viol_trace
        _ag._val_trace = val_trace
        _ag._part_log = part_log
        _ag._alpha_client = alpha_client
        _ag._client_val = client_val
    head = agents if return_all else agents[0]
    return head, np.array(reward_curve), np.array(cvar_curve), np.array(episodes_x)


def train_ddqn(cfg, seed):
    np.random.seed(seed)
    envs = [AerialEdgeEnv(cfg, b, seed * 100 + b) for b in range(cfg.n_abs)]
    agents = [DDQNAgent(cfg, seed * 100 + b) for b in range(cfg.n_abs)]
    reward_curve, cvar_curve, episodes_x = [], [], []
    cum_ep = 0
    for rnd in range(cfg.fed_rounds):
        round_rewards, n_samps, params = [], [], []
        for env, ag in zip(envs, agents):
            for _ in range(cfg.episodes_per_round):
                s = env.reset(); df, _ = env.device_features(); ep_r = 0.0
                for t in range(cfg.steps_per_episode):
                    a, m = ag.act(s, env)
                    s2, r, d, info = env.step(a)
                    done = float(t == cfg.steps_per_episode - 1)
                    ag.remember(s, m, r, s2, done); ag.update()
                    s = s2; ep_r += r
                round_rewards.append(ep_r / cfg.steps_per_episode)
            ag.eps = max(0.02, ag.eps * 0.99)
            n_samps.append(cfg.episodes_per_round); params.append(ag.get_params())
        agg = federated_average(params, n_samps, cfg.weight_clip, cfg.n_abs,
                                getattr(cfg, "agg_rule", "fedavg"))
        for ag in agents:
            ag.set_params(agg)
        cum_ep += cfg.episodes_per_round * cfg.n_abs
        reward_curve.append(np.mean(round_rewards)); episodes_x.append(cum_ep)

        if rnd % max(1, cfg.fed_rounds // 20) == 0:
            print(f"      round {rnd + 1}/{cfg.fed_rounds}  reward {reward_curve[-1]:.3f}",
                  flush=True)
        lat, _ = eval_policy(lambda e, s, df: agents[0].act(s, e)[0],
                             cfg, seeds=[seed], n_episodes=max(4, cfg.eval_episodes // 10))
        cvar_curve.append(cvar(lat, cfg.cvar_alpha) * 1e3)
    return agents[0], np.array(reward_curve), np.array(cvar_curve), np.array(episodes_x)


def sic_heatmap(agent, cfg, n=40):
    import torch
    gain_ratio = np.linspace(0.5, 3.5, n)
    demand_ratio = np.linspace(0.5, 3.5, n)
    Z = np.zeros((n, n))
    base_g = -8.5
    base_w = np.log10(np.mean(cfg.packet_bits) * np.mean(cfg.comp_intensity))
    s = torch.tensor([[base_g, base_g, base_w, 0.5, 0.8, 0.05, 1.0, 2.0 / cfg.max_devices, 0.0]],
                     dtype=torch.float32)
    with torch.no_grad():
        ctx = agent.actor.body(s)
        for i, wr in enumerate(demand_ratio):
            for j, gr in enumerate(gain_ratio):
                df = torch.zeros(1, cfg.max_devices, 4)
                df[0, 0] = torch.tensor([base_g + 0.5 * np.log10(gr),
                                         base_w + 0.5 * np.log10(wr), 0.75, 0.5])
                df[0, 1] = torch.tensor([base_g - 0.5 * np.log10(gr),
                                         base_w - 0.5 * np.log10(wr), 0.75, 0.5])
                sc = agent.actor.sic_scores(ctx, df)[0]
                Z[i, j] = float(sc[0] - sc[1])
    return gain_ratio, demand_ratio, Z


def _clone(cfg, **kw):
    import dataclasses
    return dataclasses.replace(cfg, **kw)
