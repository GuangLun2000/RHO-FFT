# P0: golden values, regression and the E0 reproduction

`p0.py` runs the GPU steps of milestone P0 and writes a feedback report. `notebooks/P0_colab.ipynb` drives it on
Colab; the same commands work on any machine with a CUDA GPU.

```bash
python experiments/p0/p0.py env tests golden regress --root outputs/p0          # about 10 minutes of GPU time
python experiments/p0/p0.py e0 --root outputs/p0 --plan lite --budget-hours 5   # E0, seed 42, within a time budget
python experiments/p0/p0.py feedback --root outputs/p0                          # feedback_P0.md, .json and a zip
python experiments/p0/p0.py env golden regress e0 feedback --tiny --root outputs/p0-tiny   # CPU dry run, 2 minutes
```

| Stage | What it does | Result |
| --- | --- | --- |
| `env` | Records interpreter, packages, GPU, driver and git state | `env.json` |
| `tests` | Runs the CPU test suite (`--gpu`: the GPU-marked tests) | `pytest.xml`, `pytest.log` |
| `golden` | Three smoke configurations on the upstream revision and, twice, on the instrumented revision; `ho-fl-replay` | `golden/<rev>-<gpu>/golden.json` |
| `regress` | The same configurations and replay on the current tree, compared with the golden values (I1, I2) | `regress/<rev>-<gpu>/verdicts.json` |
| `e0` | 3-round preflight of every E0 configuration, then the E0 queue; runs that do not fit the budget are skipped | `e0/queue.json`, `e0/timings.json`, `e0/runs/` |
| `feedback` | Condenses everything into one report with a gate G0 checklist | `feedback/` |

- `revisions.json` names the two reference commits: the verbatim upstream import and the instrumentation commit.
  The pipeline extracts them with `git archive`, so it needs a clone with history.
- Every run is `python -m ho_fl.train` in a subprocess with the tree on `PYTHONPATH`; nothing is installed.
- Re-running a stage reuses finished runs and repeats unfinished ones (the old attempt is kept as `<run>.stale-N`).
  Golden and regression results are stored per GPU model and torch version and are never reused across hardware.
- E0 plans: `opt` (5 runs, seed 42), `lite` (9 runs: OPT and Qwen, seed 42), `full` (37 runs: 5 seeds for OPT,
  3 for Qwen). Reference accuracies in `p0.py` are the values the brief quotes from the HO-FL paper.
- With `--tiny` the model and data are synthetic and the device is the CPU. It validates the pipeline, not the
  method.
