# Rerunning the experiments

All commands work in Windows PowerShell and in a Linux or macOS shell. Run them from
the repository folder unless a step says otherwise.

## 1. Install

```
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
source .venv/bin/activate         # Linux or macOS
pip install -r requirements.txt
```

If PowerShell refuses to run `Activate.ps1`, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

## 2. Short test

```
python rerun_all.py --quick --workers 2
```

Every step runs with 3 seeds and 2 training rounds and writes to `runs/quick/`. The
numbers mean nothing; the test only shows that the pipeline runs on your machine. The
first step prints `49 of 49 checks passed`, and the run ends with
`all requested steps finished`.

## 3. Full run

```
python rerun_all.py --workers 5
```

Output goes to `runs/sec3_tharq0.125ms/`. Set `--workers` to the number of physical
cores; each worker trains one seed at a time. t_HARQ = 0.125 ms and every other value
of Table I are defaults, so no flag is needed.

The two heuristics need no training, so any run must give exactly these total
violation rates (10 seeds, 150 episodes each): static heuristic 5.38%, adaptive
heuristic 0.08%.

The archived run took about 28 hours with 5 workers on a Windows PC with 8 logical cores:

| Step | Contents | Wall time |
|---|---|---|
| `check` | verification of Eqs. (4)-(13) | under 1 min |
| `train_main` | FedPPO-QR, FedPPO-Mean, non-federated reference, FedAvg-DDQN, 10 seeds each | 6.7 h |
| `eval_main` | seed-level and pooled evaluation | 8 min |
| `figs` | conditional blocklength, sweep and power/CPU grid, recorded latencies, figures, statistics | about 10 min |
| `ablations` | fixed blocklength, independent local learning, robust aggregation, fixed power split, B = 10 | 9.0 h |
| `adaptive` | adaptive risk level, 10 seeds | 1.0 h |
| `quantile` | M = 20, 50, 100, 500, seeds 0 to 2 | 1.2 h |
| `seeds30` | FedPPO-QR and the non-federated reference, seeds 10 to 29 | 3.6 h |
| `budget` | FedPPO-QR at 300 and 600 rounds, seeds 0 to 2 | 2.4 h |
| `extra` | shared-initialization local learning; non-federated reference at learning rates 1e-4 and 1e-3 | 3.5 h |
| `final` | figures and statistics with every result in place | about 10 min |

**Stopping and resuming.** The run can be stopped at any time. Running the same command
again continues it and skips finished seeds. The folder records its settings and the
hashes of the simulator and training code, and `rerun_all.py` refuses to resume a folder
started with different ones; use a new `--out` folder in that case.

**Running part of it.** `--only train_main eval_main figs` gives the principal comparison
first, and `--from ablations` continues from a step. The steps, in order:
`check train_main eval_main figs ablations adaptive quantile seeds30 budget extra final`.

## 4. Where the results are

Everything is in `runs/sec3_tharq0.125ms/`.

| Paper | File |
|---|---|
| Table I | defaults of `config.py` (`python config.py` prints them) |
| Table II | `paper_stats.txt`, "Seed-level statistics"; the CVaR column is in the last section |
| Table III | `paper_stats.txt`, last section (`cvar_recompute.py`) |
| Table IV | `paper_stats.txt`, "Paired comparison against FedPPO-QR" |
| Figs. 2 to 7 | `figs/fig2_reward.pdf`, `fig3_sweep_grid.pdf`, `fig4_blocklength.pdf`, `fig5_budget.pdf`, `fig6_ccdf.pdf`, `fig7_sic_priority.pdf` |
| other numbers in Section V | `paper_stats.txt`, under the heading of each analysis |
| per-seed results | `main_comparison.json`, `results/*.json`, `results/rows/` (one file per arm and seed), `results.json` and `results.npz` (training curves, pooled evaluation, SIC surface), `sweep_grid.json`, `fig3_dual.json`, `fig_conditional_nk_*.json` |
| recorded latency of every evaluated task | `results/latencies_per_seed.pkl` |
| trained models and training curves | `ckpt/agent_<arm>_<seed>.pt`, `ckpt/curve_<arm>_<seed>.json` |
| settings, package versions and code hashes | `manifest.json` |

The other PDF files in the run folder are diagnostic plots, not figures of the paper.

## 5. Further checks

These are not part of `rerun_all.py`. Run them from the run folder after the full run:

```
cd runs/sec3_tharq0.125ms
python ../../run_ablations.py kappa --workers 5 --out results/kappa.json
python ../../kappa_stats.py
python ../../sens_eval.py sic --ckpt ckpt --workers 5 --xi 0 0.001 0.01 0.05 0.1 --out results/sic_sensitivity.json
python ../../sens_eval.py zeta --ckpt ckpt --workers 5 --out results/zeta_sensitivity.json
python ../../flag_position.py --ckpt ckpt --seeds 10
```

* `kappa` trains FedPPO-QR and FedPPO-Mean with a violation term in the reward:
  the reward of (16) minus `kappa_violation` times the share of the interval's tasks
  that violate the deadline, with `kappa_violation` = 1 and 10 seeds per arm.
  `kappa_stats.py` prints the paired comparisons.
* `sens_eval.py sic` evaluates the trained policies and both heuristics, without
  retraining, under imperfect SIC: a dropped device remains interference for every
  device decoded after it, and each decoded stream leaves a residual `xi` of its
  received power. It also replaces only the decoding order of FedPPO-QR with the
  descending- and ascending-gain orders.
* `sens_eval.py zeta` evaluates the same policies at the fading correlations of 2 and
  1 ms control intervals under Clarke's model, 0.9837 and 0.9959, without retraining.
* `flag_position.py` gives the position of the return-tail flags within an episode
  (Section V-E).

The results of the archived run are stored as `results/kappa.json`,
`results/sic_sensitivity.json` and `results/zeta_sensitivity.json`.

## 6. Tables and figures from the archived run, without training

```
cd runs/sec3_tharq0.125ms
python ../../paper_stats.py
python ../../cvar_recompute.py
python ../../make_figures.py --run . --out figs
python ../../pooled_latencies.py --ckpt ckpt --seeds 10 --workers 5
```

The last command evaluates the stored checkpoints again and rewrites
`results/latencies_per_seed.pkl` with identical values.

The archived run was started before the violation penalty and the options that no
reported experiment uses were added to `config.py`, `rl.py`, `train.py` and
`evaluate_arms.py`. At their defaults, these options leave training unchanged
(identical weights in a test run) and evaluation unchanged except for the CVaR of
delivered tasks, which now splits ties at the 95th percentile by probability mass, as
in Problem (14). The CVaR column of Table II comes from `cvar_recompute.py`, which
applies this definition to the stored latencies. Because the hashes of these four files
differ from those in `manifest.json`, `rerun_all.py` will not resume the archived folder;
use the commands above, or start a new run.

## 7. If something fails

* The script stops and names the failed step and its log file in `logs/`. Fix the cause
  and run the same command again; finished work is kept.
* Out of memory: use fewer workers.
* Keep the computer from sleeping during a long run.
