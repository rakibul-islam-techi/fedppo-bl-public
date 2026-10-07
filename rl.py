from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


def action_dim(cfg: Config) -> int:
    return n_action_channels(cfg) * cfg.max_devices


def n_action_channels(cfg: Config) -> int:
    return 3 if cfg.drop_blocklength_head else 4


class GaussianActor(nn.Module):
    DEV_FEAT = 4

    def __init__(self, state_dim, max_devices, hidden, n_channels=4):
        super().__init__()

        self.C = n_channels
        self.D = max_devices
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.dev_head = nn.Sequential(
            nn.Linear(hidden + self.DEV_FEAT, hidden), nn.Tanh(),
            nn.Linear(hidden, n_channels),
        )

        ls = torch.full((n_channels * max_devices,), -1.2)
        if n_channels == 4:
            ls[3 * max_devices:] = -0.8
        self.log_std = nn.Parameter(ls)

    def sic_scores(self, ctx, dev_feats):
        B, D, _ = dev_feats.shape
        ctx_exp = ctx.unsqueeze(1).expand(B, D, ctx.shape[-1])
        x = torch.cat([ctx_exp, dev_feats], dim=-1)
        return self.dev_head(x)[..., 1]

    def forward(self, s, dev_feats):
        h = self.body(s)
        B, D, _ = dev_feats.shape
        ctx = h.unsqueeze(1).expand(B, D, h.shape[-1])
        out = self.dev_head(torch.cat([ctx, dev_feats], dim=-1))

        mu = torch.cat([out[..., c] for c in range(self.C)], dim=-1)
        return mu, self.log_std.clamp(-4, 2).exp()

    def dist(self, s, dev_feats):
        mu, std = self.forward(s, dev_feats)
        return torch.distributions.Normal(mu, std)


class QRCritic(nn.Module):
    def __init__(self, state_dim, hidden, n_quantiles):
        super().__init__()
        self.M = n_quantiles
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_quantiles),
        )
        taus = (torch.arange(n_quantiles, dtype=torch.float32) + 0.5) / n_quantiles
        self.register_buffer("taus", taus)

    def forward(self, s):
        return self.net(s)

    def value(self, s):
        return self.net(s).mean(dim=-1, keepdim=True)


def quantile_huber_loss(pred, target, taus, kappa=1.0):
    delta = target - pred
    huber = torch.where(delta.abs() <= kappa,
                        0.5 * delta ** 2,
                        kappa * (delta.abs() - 0.5 * kappa))
    weight = (taus.unsqueeze(0) - (delta.detach() < 0).float()).abs()
    return (weight * huber).mean()


class CVaRPPOAgent:
    def __init__(self, cfg: Config, seed: int):
        self.cfg = cfg
        torch.manual_seed(seed)
        from sim_core import AerialEdgeEnv
        sd = AerialEdgeEnv.STATE_DIM
        self.actor = GaussianActor(sd, cfg.max_devices, cfg.hidden,
                                   n_action_channels(cfg))
        self.critic = QRCritic(sd, cfg.hidden, cfg.n_quantiles)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)

    def _dim_mask(self, mask):
        m = torch.as_tensor(mask, dtype=torch.float32)
        if m.dim() == 1:
            m = m.unsqueeze(0)
        return m.repeat(1, self.actor.C)

    @torch.no_grad()
    def act(self, state, dev_feats, mask):
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        df = torch.as_tensor(dev_feats, dtype=torch.float32).unsqueeze(0)
        d = self.actor.dist(s, df)
        a = d.sample()
        logp = (d.log_prob(a) * self._dim_mask(mask)).sum(-1)
        return a.squeeze(0).numpy(), float(logp), float(self.critic.value(s))

    @torch.no_grad()
    def act_greedy(self, state, dev_feats):
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        df = torch.as_tensor(dev_feats, dtype=torch.float32).unsqueeze(0)
        mu, _ = self.actor(s, df)
        return mu.squeeze(0).numpy()

    def _gae(self, rews, vals, dones, v_next=None):
        """GAE, Eq. (19). With v_next (critic value at s_{t+1}), an episode end is a
        time-limit truncation: delta_t bootstraps with V(s_{t+1}) and the recursion
        restarts at the boundary, so no return crosses into the next episode."""
        cfg = self.cfg
        adv = np.zeros_like(rews)
        last = 0.0
        for t in reversed(range(len(rews))):
            nonterm = 1.0 - dones[t]
            if v_next is None:
                nextv = vals[t + 1] if t + 1 < len(vals) else 0.0
                delta = rews[t] + cfg.gamma * nextv * nonterm - vals[t]
            else:
                nextv = v_next[t] if dones[t] else vals[t + 1]
                delta = rews[t] + cfg.gamma * nextv - vals[t]
            last = delta + cfg.gamma * cfg.gae_lambda * nonterm * last
            adv[t] = last
        ret = adv + vals[:len(rews)]
        return adv, ret

    def update(self, batch):
        cfg = self.cfg
        S = torch.as_tensor(np.array(batch["s"]), dtype=torch.float32)
        DF = torch.as_tensor(np.array(batch["df"]), dtype=torch.float32)
        MK = self._dim_mask(np.array(batch["m"]))
        A = torch.as_tensor(np.array(batch["a"]), dtype=torch.float32)
        LP = torch.as_tensor(np.array(batch["logp"]), dtype=torch.float32)
        R = np.array(batch["r"], dtype=np.float32)
        D = np.array(batch["done"], dtype=np.float32)
        V = np.array(batch["v"] + [0.0], dtype=np.float32)

        Sn = torch.as_tensor(np.array(batch["s2"]), dtype=torch.float32)
        boot = bool(getattr(cfg, "bootstrap_timelimit", False))
        if boot:
            with torch.no_grad():
                v_next = self.critic.value(Sn).squeeze(-1).numpy()
            adv, ret = self._gae(R, V, D, v_next)
        else:
            adv, ret = self._gae(R, V, D)
        adv_t = torch.as_tensor(adv, dtype=torch.float32)
        ret_t = torch.as_tensor(ret, dtype=torch.float32)

        if cfg.lambda_cvar > 0 and cfg.n_quantiles > 1:
            if getattr(cfg, "tail_source", "return") == "violation":
                # task-level tail at the aligned level alpha* = 1 - v_tot: the CVaR tail
                # of the task latencies is the violation set, so each interval is
                # weighted by its share of violating tasks
                tail = torch.as_tensor(np.array(batch["vf"]), dtype=torch.float32)
            else:
                with torch.no_grad():
                    _alpha = float(getattr(self, '_alpha_r', cfg.cvar_alpha))
                    var_b = torch.quantile(ret_t, 1.0 - _alpha)
                tail = (ret_t <= var_b).float()

            self._last_tail_frac = float(tail.mean())
            k = cfg.lambda_cvar * cfg.cvar_boost
            if getattr(cfg, "fixed_eta", False):
                # keep the total extra weight at its alpha = 0.95 value, so that
                # eta = 1/6 and only the choice of weighted transitions changes
                k = k * (1.0 - cfg.cvar_alpha) / max(float(tail.mean()), 1e-8)
            w = 1.0 + k * tail
            self._last_mean_w = float(w.mean())
        else:
            w = torch.ones_like(ret_t)

        adv_n = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        eff_adv = (w * adv_n).detach()

        idx = np.arange(len(R))
        for _ in range(cfg.local_epochs):
            np.random.shuffle(idx)
            for start in range(0, len(idx), cfg.minibatch):
                mb = idx[start:start + cfg.minibatch]
                mbt = torch.as_tensor(mb)

                d = self.actor.dist(S[mbt], DF[mbt])
                logp = (d.log_prob(A[mbt]) * MK[mbt]).sum(-1)
                ratio = (logp - LP[mbt]).exp()
                a1 = ratio * eff_adv[mbt]
                a2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * eff_adv[mbt]
                loss_a = -torch.min(a1, a2).mean() - 0.001 * (d.entropy() * MK[mbt]).sum(-1).mean()
                self.opt_a.zero_grad(); loss_a.backward(); self.opt_a.step()

                with torch.no_grad():
                    zp = self.critic(Sn[mbt])
                    r_mb = torch.as_tensor(R[mb], dtype=torch.float32).unsqueeze(1)
                    nonterm = torch.as_tensor(1 - D[mb] if not boot else np.ones(len(mb)),
                                              dtype=torch.float32).unsqueeze(1)
                    y = r_mb + cfg.gamma * nonterm * zp
                z = self.critic(S[mbt])
                if cfg.n_quantiles > 1:
                    loss_c = quantile_huber_loss(z, y, self.critic.taus)
                else:
                    loss_c = F.mse_loss(z, y)
                self.opt_c.zero_grad(); loss_c.backward(); self.opt_c.step()

    def get_params(self):
        return ([p.detach().clone() for p in self.actor.parameters()],
                [p.detach().clone() for p in self.critic.parameters()])

    def set_params(self, actor_p, critic_p):
        with torch.no_grad():
            for p, q in zip(self.actor.parameters(), actor_p):
                p.copy_(q)
            for p, q in zip(self.critic.parameters(), critic_p):
                p.copy_(q)


class DiscreteActionBook:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.modes = []
        for sic in ("gain", "revgain"):
            for rho in (-2.0, 0.0, 2.0):
                for cpu in ("equal", "prop"):
                    for nb in (-1.5, 0.0, 1.5):
                        self.modes.append((sic, rho, cpu, nb))

    def __len__(self):
        return len(self.modes)

    def to_action(self, mode_idx, env):
        sic, rho, cpu, nb = self.modes[mode_idx]
        D = self.cfg.max_devices; n = env.n_dev
        a = np.zeros(4 * D)
        a[:D][:n] = rho
        sign = -1.0 if sic == "gain" else 1.0
        a[D:2 * D][:n] = sign * env.gain[:n] * 1e9
        a[2 * D:3 * D][:n] = 0.0 if cpu == "equal" else np.log(env.workload[:n])
        a[3 * D:4 * D][:n] = nb
        return a


class QNet(nn.Module):
    def __init__(self, state_dim, n_actions, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, s):
        return self.net(s)


class DDQNAgent:
    def __init__(self, cfg: Config, seed: int):
        self.cfg = cfg
        torch.manual_seed(seed)
        self.book = DiscreteActionBook(cfg)
        from sim_core import AerialEdgeEnv
        self.q = QNet(AerialEdgeEnv.STATE_DIM, len(self.book), cfg.hidden)
        self.qt = QNet(AerialEdgeEnv.STATE_DIM, len(self.book), cfg.hidden)
        self.qt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr_critic)
        self.eps = 0.2
        self.buf = []

    @torch.no_grad()
    def act(self, state, env):
        if np.random.random() < self.eps:
            m = np.random.randint(len(self.book))
        else:
            s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            m = int(self.q(s).argmax())
        return self.book.to_action(m, env), m

    def remember(self, s, m, r, s2, d):
        self.buf.append((s, m, r, s2, d))
        if len(self.buf) > 20000:
            self.buf = self.buf[-20000:]

    def update(self):
        if len(self.buf) < self.cfg.minibatch:
            return
        idx = np.random.randint(0, len(self.buf), size=self.cfg.minibatch)
        s, m, r, s2, d = zip(*[self.buf[i] for i in idx])
        S = torch.as_tensor(np.array(s), dtype=torch.float32)
        M = torch.as_tensor(np.array(m), dtype=torch.long).unsqueeze(1)
        Rr = torch.as_tensor(np.array(r), dtype=torch.float32).unsqueeze(1)
        S2 = torch.as_tensor(np.array(s2), dtype=torch.float32)
        Dd = torch.as_tensor(np.array(d), dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            a2 = self.q(S2).argmax(1, keepdim=True)
            tq = self.qt(S2).gather(1, a2)
            y = Rr + self.cfg.gamma * (1 - Dd) * tq
        q = self.q(S).gather(1, M)
        loss = F.smooth_l1_loss(q, y)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        for tp, p in zip(self.qt.parameters(), self.q.parameters()):
            tp.data.mul_(0.99).add_(0.01 * p.data)

    def get_params(self):
        return [p.detach().clone() for p in self.q.parameters()]

    def set_params(self, params):
        with torch.no_grad():
            for p, q in zip(self.q.parameters(), params):
                p.copy_(q)


def federated_average(param_lists, n_samples, weight_clip, n_clients,
                      rule="fedavg"):
    if rule == "fedavg":
        w = np.array(n_samples, dtype=np.float64)
        w = w / w.sum()
        w = np.minimum(w, weight_clip / n_clients)
        w = w / w.sum()
        agg = []
        for layer in zip(*param_lists):
            stacked = torch.stack([wi * p for wi, p in zip(w, layer)], dim=0)
            agg.append(stacked.sum(dim=0))
        return agg

    agg = []
    for layer in zip(*param_lists):
        stacked = torch.stack(list(layer), dim=0)
        B = stacked.shape[0]
        if rule == "median":
            agg.append(stacked.median(dim=0).values)
        elif rule == "trimmed":
            s, _ = torch.sort(stacked, dim=0)
            c = max(1, int(0.2 * B))
            hi = B - c
            agg.append(s[c:hi].mean(dim=0) if hi > c else stacked.mean(dim=0))
        else:
            raise ValueError(f"unknown aggregation rule: {rule}")
    return agg


def heuristic_action(env, cfg: Config):
    D = cfg.max_devices; n = env.n_dev
    a = np.zeros(n_action_channels(cfg) * D)
    lg = np.log10(env.gain[:n] + 1e-30)
    a[D:2 * D][:n] = -(lg - lg.mean())
    ld = np.log(env.workload[:n])
    a[2 * D:3 * D][:n] = ld - ld.mean()
    return a


def blocklength_logit(cfg: Config, n):
    p = np.clip((np.asarray(n, dtype=float) - cfg.n_min) / (cfg.n_max - cfg.n_min), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def heuristic_fixed_n_action(env, cfg: Config, n):
    a = heuristic_action(env, cfg)
    D = cfg.max_devices
    a[3 * D:4 * D] = blocklength_logit(cfg, n)
    return a


def heuristic_adaptive_n_action(env, cfg: Config, eps_target: float = 1e-2):
    from sim_core import block_error, sic_sinr, stream_bits
    a = heuristic_action(env, cfg)
    D, n = cfg.max_devices, env.n_dev
    rho, _, sic_order, _, _ = env._decode_action(a)
    g1, g2 = sic_sinr(env.gain, env.Pk, rho, sic_order, env.noise)
    L1, L2 = stream_bits(env.Lk, rho)

    def err(nb):
        return 1.0 - (1.0 - block_error(g1, L1 / nb, nb)) * (1.0 - block_error(g2, L2 / nb, nb))

    lo = np.full(n, float(cfg.n_min))
    hi = np.full(n, float(cfg.n_max))
    while np.any(lo < hi):
        mid = np.floor((lo + hi) / 2.0)
        ok = err(mid) <= eps_target
        hi = np.where(ok & (lo < hi), mid, hi)
        lo = np.where(~ok & (lo < hi), mid + 1.0, lo)
    sig = np.clip((lo - cfg.n_min) / (cfg.n_max - cfg.n_min), 1e-6, 1 - 1e-6)
    a[3 * D:4 * D][:n] = np.log(sig / (1 - sig))
    return a


def random_action(env, cfg: Config):
    return np.random.randn(n_action_channels(cfg) * cfg.max_devices)
