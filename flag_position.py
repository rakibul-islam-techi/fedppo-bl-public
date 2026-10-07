"""Where does the return-tail weight u_t of Eq. (20) fall within an episode?

Runs one round of rollouts (behavior policy, fresh environments) for each
trained FedPPO-QR checkpoint, recomputes the GAE returns and the empirical
(1 - alpha) threshold exactly as in rl.CVaRPPOAgent.update, and reports the
share of flagged transitions by interval index within the episode.

    set FEDPPO_T_HARQ_MS=0.125 & python flag_position.py --ckpt ckpt --seeds 10
"""
import argparse
import numpy as np
import torch
from config import full_config
from sim_core import AerialEdgeEnv
from rl import CVaRPPOAgent
from train import collect_ppo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt"); ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    cfg = full_config(); T = cfg.steps_per_episode
    by_t = np.zeros(T); n_flag = 0; r_flag, r_same = [], []
    for sd in range(a.seeds):
        ag = CVaRPPOAgent(cfg, sd)
        ck = torch.load(f"{a.ckpt}/agent_FedPPO-QR_{sd}.pt", map_location="cpu", weights_only=False)
        ag.actor.load_state_dict(ck["actor"]); ag.critic.load_state_dict(ck["critic"])
        for b in range(cfg.n_abs):
            env = AerialEdgeEnv(cfg, b, 900_000 + sd * 100 + b)
            batch, _, _ = collect_ppo(ag, env, cfg)
            R = np.array(batch["r"], np.float32); D = np.array(batch["done"], np.float32)
            V = np.array(batch["v"] + [0.0], np.float32)
            _, ret = ag._gae(R, V, D)
            flag = ret <= np.quantile(ret, 1 - cfg.cvar_alpha)
            t_idx = np.arange(len(R)) % T
            by_t += np.bincount(t_idx[flag], minlength=T); n_flag += flag.sum()
            for t in np.unique(t_idx[flag]):
                sel = t_idx == t
                r_flag.append(R[sel & flag].mean())
                if (sel & ~flag).any():
                    r_same.append(R[sel & ~flag].mean())
    share = 100 * by_t / n_flag
    print("share of flagged transitions by interval index (%):",
          " ".join(f"t{t}:{s:.1f}" for t, s in enumerate(share) if s >= 0.05))
    print(f"first three intervals: {share[:3].sum():.1f}% of flagged transitions "
          f"(they are {100 * 3 / T:.1f}% of all transitions)")
    print(f"mean reward, flagged vs unflagged at the same index: {np.mean(r_flag):.3f} vs {np.mean(r_same):.3f}")


if __name__ == "__main__":
    main()
