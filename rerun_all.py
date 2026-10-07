from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
STEPS = ["check", "train_main", "eval_main", "figs", "ablations", "adaptive",
         "quantile", "seeds30", "budget", "extra", "final"]


def run(cmd, cwd, env, log):
    line = "$ python " + " ".join(str(c) for c in cmd[1:])
    print(line, flush=True)
    log.write(line + "\n"); log.flush()
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    out = []
    for ln in p.stdout:
        sys.stdout.write(ln); sys.stdout.flush()
        log.write(ln); log.flush()
        out.append(ln)
    p.wait()
    if p.returncode != 0:
        raise SystemExit(f"\nstep failed (exit code {p.returncode}): {line}\nsee {log.name}")
    return "".join(out)


def py(script, *args):
    return [sys.executable, str(CODE / script), *[str(a) for a in args]]


def figures(o, env, log, a):
    for arm, tag in (("FedPPO-QR", "qr"), ("FedPPO-Mean", "mean")):
        run(py("fig_conditional_nk.py", "--ckpt", "ckpt", "--arm", arm,
               "--seeds", a.seeds, "--out", f"fig_conditional_nk_{tag}"), o, env, log)
    lims = [json.load(open(o / f"fig_conditional_nk_{t}.json"))["ylim"] for t in ("qr", "mean")]
    lo, hi = min(l[0] for l in lims), max(l[1] for l in lims)
    for tag in ("qr", "mean"):
        run(py("fig_conditional_nk.py", "--replot", "--ylim", lo, hi,
               "--out", f"fig_conditional_nk_{tag}"), o, env, log)
    run(py("fig3_dual.py"), o, env, log)
    run(py("sweep_grid.py", "--workers", a.workers), o, env, log)
    run(py("pooled_latencies.py", "--ckpt", "ckpt", "--seeds", a.seeds, "--workers", a.workers), o, env, log)
    run(py("make_figures.py", "--run", ".", "--out", "figs"), o, env, log)
    text = run(py("paper_stats.py"), o, env, log)
    text += "\n" + run(py("cvar_recompute.py"), o, env, log)
    (o / "paper_stats.txt").write_text(text, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-harq-ms", type=float, default=0.125)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--from", dest="start", choices=STEPS, default="check")
    ap.add_argument("--only", choices=STEPS, nargs="+", default=None)
    a = ap.parse_args()

    if a.out is None:
        a.out = "runs/quick" if a.quick else f"runs/sec3_tharq{a.t_harq_ms:g}ms"
    o = Path(a.out).resolve()
    (o / "logs").mkdir(parents=True, exist_ok=True)
    (o / "results").mkdir(exist_ok=True)

    env = dict(os.environ)
    env.update(FEDPPO_T_HARQ_MS=repr(a.t_harq_ms), FEDPPO_QUICK="1" if a.quick else "0",
               PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    sys.path.insert(0, str(CODE))
    os.environ.update({k: env[k] for k in ("FEDPPO_T_HARQ_MS", "FEDPPO_QUICK")})
    from config import full_config, describe
    cfg = full_config()
    a.seeds, a.episodes = cfg.seeds, cfg.eval_episodes

    core = {f: hashlib.sha256((CODE / f).read_bytes()).hexdigest()[:16]
            for f in ("config.py", "sim_core.py", "rl.py", "train.py", "evaluate_arms.py")}
    settings = dict(model="section3", t_harq_ms=a.t_harq_ms,
                    quick=a.quick, seeds=cfg.seeds, fed_rounds=cfg.fed_rounds,
                    eval_episodes=cfg.eval_episodes, kappa_violation=cfg.kappa_violation,
                    core_code=core)
    mf = o / "manifest.json"
    if mf.exists():
        old = json.load(open(mf))["settings"]
        if old != settings:
            raise SystemExit(f"{o} was started with different settings or a different version of the "
                             f"simulator or training code:\n  {old}\nnow:\n  {settings}\n"
                             f"use a new --out folder")
    else:
        import numpy, scipy, torch, matplotlib
        code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(CODE.glob("*.py"))}
        json.dump(dict(settings=settings, started=dt.datetime.now().isoformat(timespec="seconds"),
                       workers=a.workers, cpu_count=os.cpu_count(), python=platform.python_version(),
                       platform=platform.platform(), torch=torch.__version__, numpy=numpy.__version__,
                       scipy=scipy.__version__, matplotlib=matplotlib.__version__, code_sha256=code,
                       config=cfg.as_dict()), open(mf, "w"), indent=2)

    print(describe(cfg))
    print(f"output folder: {o}")
    print(f"seeds {cfg.seeds}, rounds {cfg.fed_rounds}, eval episodes {cfg.eval_episodes}, workers {a.workers}\n",
          flush=True)

    W = a.workers
    todo = a.only or STEPS[STEPS.index(a.start):]
    t_all = time.time()
    for i, step in enumerate(STEPS):
        if step not in todo:
            continue
        t0 = time.time()
        print(f"\n===== [{i + 1}/{len(STEPS)}] {step}  {dt.datetime.now():%Y-%m-%d %H:%M} =====", flush=True)
        with open(o / "logs" / f"{i + 1:02d}_{step}.log", "a", encoding="utf-8") as log:
            if step == "check":
                run(py("check_model.py"), o, env, log)
            elif step == "train_main":
                run(py("run_chunked.py", "main", "full", "--workers", W), o, env, log)
            elif step == "eval_main":
                run(py("run_chunked.py", "evals", "full"), o, env, log)
                run(py("run_chunked.py", "assemble", "full"), o, env, log)
                run(py("evaluate_arms.py", "--ckpt", "ckpt", "--seeds", a.seeds,
                       "--episodes", a.episodes), o, env, log)
            elif step == "figs":
                figures(o, env, log, a)
            elif step == "ablations":
                run(py("run_ablations.py", "all", "--seeds", a.seeds, "--workers", W), o, env, log)
            elif step == "adaptive":
                run(py("run_adaptive_alpha.py", "--seeds", a.seeds, "--workers", W), o, env, log)
            elif step == "quantile":
                run(py("run_ablations.py", "quantile", "--seeds", min(3, a.seeds), "--workers", W,
                       "--out", "results/quantile.json"), o, env, log)
            elif step == "seeds30":
                run(py("run_extra_seeds.py", "--total", 30, "--workers", W), o, env, log)
            elif step == "budget":
                run(py("run_budget.py", "--seeds", 3, "--mult", 2, 4, "--workers", W), o, env, log)
            elif step == "extra":
                run(py("run_ablations.py", "extra", "--seeds", a.seeds, "--workers", W), o, env, log)
            elif step == "final":
                figures(o, env, log, a)
        print(f"----- {step} finished in {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"\nall requested steps finished in {(time.time() - t_all) / 3600:.2f} h")
    print(f"tables and in-text numbers: {o / 'paper_stats.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
