# P0: golden values, regression and the E0 reproduction

`p0.py` runs the GPU steps of milestone P0 and writes a feedback report. `notebooks/P0_colab.ipynb` drives it on
Colab; the same commands work on any machine with a CUDA GPU.

```bash
python experiments/p0/p0.py env tests golden regress --root outputs/p0          # about 10 minutes of GPU time
python experiments/p0/p0.py e0 --root outputs/p0 --plan full --budget-hours 5   # as much of E0 as fits in the budget
python experiments/p0/p0.py feedback --root outputs/p0                          # feedback_P0.md, .json and a zip
python experiments/p0/p0.py env golden regress e0 feedback --tiny --root outputs/p0-tiny   # CPU dry run, 2 minutes
```

| Stage | What it does | Result |
| --- | --- | --- |
| `env` | Records interpreter, packages, GPU, driver and git state | `env.json` |
| `tests` | Runs the CPU test suite (`--gpu`: the GPU-marked tests) | `pytest.xml`, `pytest.log` |
| `golden` | Three smoke configurations on the upstream revision and, twice, on the instrumented revision; `ho-fl-replay` | `golden/<rev>-<gpu>/golden.json` |
| `regress` | The same configurations and replay on the current tree, compared with the golden values (I1, I2) | `regress/<rev>-<gpu>/verdicts.json` |
| `e0` | The E0 queue in priority order; each configuration is probed for 3 rounds before its first run, and runs that do not fit the budget are skipped | `e0/queue.json`, `e0/timings-<gpu>.json`, `e0/runs/` |
| `feedback` | Condenses everything into one report: gate G0 checklist, E0 table, first-order supply seen in the ledgers, timings | `feedback/` |

- `revisions.json` names the two reference commits: the verbatim upstream import and the instrumentation commit.
  The pipeline extracts them with `git archive`, so it needs a clone with history.
- Every run is `python -m ho_fl.train` in a subprocess with the tree on `PYTHONPATH`; nothing is installed.
- Re-running a stage reuses finished runs and repeats unfinished ones (the old attempt is kept as `<run>.stale-N`).
  Golden and regression results are stored per GPU model and torch version and are never reused across hardware.
- E0 plans: `opt` (5 runs, seed 42), `qwen` (4 runs: Qwen2.5-1.5B on AG News, seed 42), `lite` (9 runs: OPT and Qwen, seed 42), `full` (37 runs: 5 seeds for OPT,
  3 for Qwen, in the order OPT seed 42, Qwen seed 42, remaining seeds). Reference accuracies in `p0.py` are the
  values the brief quotes from the HO-FL paper. Only directories named by the plan count towards the statistics.
- The smoke runs of each source tree use their own cache directory (`<cache>-baseline`, `-instrumented`, `-head`),
  so tokenisation and partitioning are part of the comparison. Timings are stored per GPU model.
- With `--tiny` the model and data are synthetic and the device is the CPU. It validates the pipeline, not the
  method.
