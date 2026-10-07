# FedPPO-BL

Simulator, training code and analysis scripts for
*Blocklength-Aware Federated Control for URLLC-Critical Aerial Edge IoT Networks*
(IEEE Internet of Things Journal, under review).

The run behind every table and figure of the paper, with all trained checkpoints and
per-seed results, is in `runs/sec3_tharq0.125ms/`.

## Setup

```
pip install -r requirements.txt
```

The archived run used Python 3.10.9, PyTorch 2.9.1 (CPU), NumPy 2.2.6, SciPy 1.15.3 and
Matplotlib 3.10.7 on Windows. The code also runs on Linux with Python 3.11 and PyTorch 2.14.


| Paper | Code (`sim_core.py`) |
|---|---|
| Eqs. (1)-(2): x_k = sqrt(p_k1) s_k1 + sqrt(p_k2) s_k2, p_k1 = rho_k P_k, p_k2 = (1 - rho_k) P_k | `sic_sinr`, with P_k = 23 dBm for every device |
| Eq. (3): all devices of M_i transmit concurrently over W_i, noise N_0 W_i | every device of M_i transmits in every interval |
| Eqs. (4)-(5): SINRs under ideal SIC in the order pi_i, s_k1 before s_k2 | `sic_sinr` |
| n_min <= n_k <= n_max, per-device control | `_decode_action` |
| Bit split: L_k1 = rho_k L_k rounded to the nearest integer in [1, L_k - 1], L_k2 = L_k - L_k1 | `stream_bits` |
| Eqs. (6)-(7): block-error probability of each stream from its SINR, dispersion and coding rate | `block_error`, `dispersion`; stream s carries L_ks bits in the n_k channel uses shared by both streams |
| Eq. (8): rate at a prescribed error probability | used by the adaptive-blocklength heuristic through Eq. (6) |
| Eq. (9): A_ks = min(G_ks, N_HARQ), G_ks ~ Geom(1 - eps_ks), attempts independent | `harq_attempts` |
| Eq. (10): T_tx = max over s of (airtime of one attempt + (A_ks - 1) t_HARQ) | `step`; the airtime of one attempt is n_k / W_i |
| a stream is dropped if G_ks > N_HARQ | the task is dropped if either stream is dropped |
| Eq. (11): sum_k f_ki <= F_i^max | `_decode_action` (the allocation sums to F_i^max) |
| Eq. (12): T_comp = L_k c_k / f_ki | `step` |
| Eq. (13): T_k = T_tx + T_comp | `step` |
| Problem (14): controls rho_k, pi_i, f_ki, n_k | the action of Eq. (15): (rho, q, f, n) |

All parameters are the defaults of `config.py` and follow Table I, including
t_HARQ = 0.125 ms and the CPU floor omega F_i^max / |M_i| with omega = 0.2. The channel
h_k combines air-to-ground path loss, with the excess loss averaged over the LoS
probability, log-normal shadowing and Rayleigh block fading that is Gauss-Markov
across intervals. Its parameters (ABS altitude 100 m, LoS parameters a = 9.61 and
b = 0.16, excess losses of 1 and 20 dB, shadowing standard deviation 4 dB, fading
correlation 0.9 per interval) are in `config.py`.

Bits and blocklengths are integers. A dropped task is recorded with its airtime T_tx and
is not computed. Every task ends before the next interval starts: `config.check`
verifies this for the worst case, 3.465 ms against T_c = 5 ms, so no task waits for
another.

`python check_model.py` recomputes Eqs. (4)-(13) independently for every task of
several full episodes, compares them with the simulator, and tests the distribution
of Eq. (9) and the inversion of Eq. (8).

## Reproducing the results

`python rerun_all.py --workers 5` trains and evaluates everything and writes the
tables and figures. `RERUN.md` lists the steps, the running times, the files that hold
each table and figure, and how to regenerate them from the archived run without
training.

## Code

| File | Contents |
|---|---|
| `config.py` | simulation and training parameters (Table I) |
| `sim_core.py` | the Section III environment |
| `rl.py` | PPO with quantile and mean critics, FedAvg-DDQN, the static and adaptive-blocklength heuristics |
| `train.py` | training loops and policy evaluation |
| `run_chunked.py` | resumable training of the principal methods, pooled evaluation and the SIC priority surface of Fig. 7 |
| `evaluate_arms.py` | seed-level evaluation of the six principal methods (Table II) |
| `run_ablations.py` | fixed blocklength (head kept and head removed), independent local learning (separate and shared initialization), median and trimmed-mean aggregation, fixed power split, B = 10, quantile resolution M, non-federated learning-rate sweep, violation penalty in the reward (`kappa`) |
| `run_adaptive_alpha.py` | adaptive risk level, Eq. (25) |
| `run_extra_seeds.py` | FedPPO-QR and the non-federated reference (`Centralized-QR`) with 30 training seeds |
| `run_budget.py` | FedPPO-QR with 2 and 4 times the training rounds |
| `sweep_grid.py`, `fig3_dual.py` | blocklength sweep and every method at three transmit powers, three CPU capacities and two CPU allocations |
| `fig_conditional_nk.py` | selected blocklength against post-SIC SINR and computation demand |
| `pooled_latencies.py` | recorded latency of every evaluated task, per method and seed |
| `cvar_recompute.py` | Table III and the CVaR column of Table II |
| `flag_position.py` | position of the return-tail flags of Eq. (20) within an episode (Section V-E) |
| `make_figures.py`, `paper_style.py`, `fig_budget.py` | Figs. 2 to 7 |
| `paper_stats.py` | every other table entry and in-text statistic |
| `kappa_stats.py` | paired statistics of the violation-penalty runs |
| `sens_eval.py` | evaluation-only checks on trained checkpoints: imperfect SIC with error propagation, and the channel correlation of shorter control intervals |
| `check_model.py` | verification of Eqs. (4)-(13) |
| `rerun_all.py` | runs the pipeline in order |

## Seeds

Training seeds are 0 to 9 (0 to 29 for FedPPO-QR and the non-federated reference in
`run_extra_seeds.py`, 0 to 2 in `run_budget.py` and for the quantile-resolution arms).
For training seed s, every evaluation, including the conditional-blocklength figure and
each agent of the independent-local arms, uses environment seed 10100 + s. Channels and
HARQ outcomes use separate random streams, so all methods see the same channel
realizations on the same seed. The blocklength sweep and the power and CPU grid use
6000 + s for s = 0 to 19 (5 episodes each), so the nominal column of the grid reproduces
the sweep exactly. Training seeds NumPy and PyTorch, so a run is reproducible on the
same machine with the same number of workers.
