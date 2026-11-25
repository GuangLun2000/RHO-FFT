# E0: the first experiment of the satellite scenario (gate G0)

`e0.py` runs everything gate G0 of the brief asks for, plus three diagnostics that are cheap now and expensive to
learn later, and writes one feedback report. `notebooks/RHO-FFT_colab.ipynb` drives it on Colab; the same commands
work on any machine with a CUDA GPU.

```bash
python experiments/sat/e0/e0.py env baseline tests trace data --root outputs/sat/e0   # the baseline needs the GPU, the rest does not
python experiments/sat/e0/e0.py selfcheck memory smoke pilot --root outputs/sat/e0    # about 30 minutes of GPU time
python experiments/sat/e0/e0.py feedback --root outputs/sat/e0                        # feedback_E0.md, .json and a zip
python experiments/sat/e0/e0.py all --tiny --root outputs/sat/e0-tiny                 # CPU dry run, about 4 minutes
```

## What the session is for

Every stage exists for one of six questions. Cheap stages that can invalidate the rest come first.

| Question | Stages | What the answer decides |
| --- | --- | --- |
| Is upstream HO-FL unchanged? | `baseline`, `tests` | Everything on the GPU is skipped when it is not |
| Is the satellite code right on the real models? | `trace`, `data`, `selfcheck` | The training runs are skipped when it is not |
| How much memory does each cut need? | `memory` | The memory states of the system model (brief 7.6) |
| What does a run cost? | timings of `smoke` and `pilot` | The GPU budget of E1 to E4; whether the replay needs optimising first |
| Does the model learn, is the zeroth-order estimate faithful? | `smoke`, the diagnostic of `selfcheck` | Whether learning rate and perturbation radius are sane on the new tasks |
| Does EuroSAT saturate, is a day the right horizon? | `pilot` | Dataset or main metric of the vision task (gate G3), before its experiments run |

A seventh question, how much the zeroth-order part matters in upstream HO-FL (E0-b), was answered by the stage `t0`
on 2026-11-24 (results in the milestone report P-S3). The stage stays callable but is no longer part of `all` or of
the notebook.

## Stages

| Stage | What it does | Item of E0 |
| --- | --- | --- |
| `env` | Interpreter, packages, GPU and git state | |
| `baseline` | `experiments/p0/p0.py env golden regress feedback`: HO-FL, FedAvg and DeComFL on this code against the upstream commit, and `ho-fl-replay`. Not repeated while `verified.json` covers the code: it records the commit, GPU and PyTorch the regression passed on, and the stage runs again only when a file outside `ho_fl/sat/` that upstream HO-FL executes has changed since, or the GPU or PyTorch differ | E0-a 1 |
| `tests` | The CPU test suite: upstream tests and `tests/sat` | E0-a 1 |
| `trace` | Builds the S1 trace (`ho-fl-sat-trace`) and compares 19 statistics with appendix A of the brief | E0-a 6 |
| `data` | Downloads HumAID (per-event release) and EuroSAT, builds the partitions of seeds 42, 43, 44 in the base and the forward-only ("zonly") variant, prints the statistics of seed 42 and all partition hashes | |
| `selfcheck` | `ho_fl.sat.selfcheck` on ViT / EuroSAT and Qwen / HumAID, and against upstream `train_client` on Qwen / AG News. Its diagnostic compares the zeroth-order scalars of training with exact directional derivatives at three perturbation radii | E0-a 2 to 5 |
| `memory` | `ho_fl.sat.calibrate`: peak GPU memory of a participation at every cut, Qwen and ViT | section 7.6 |
| `smoke` | Two simulated hours of the full method on both tasks and of the three other implemented methods on EuroSAT | E0-a 7 |
| `pilot` | A whole simulated day of the full method and of the centralised reference on EuroSAT. Not a result (server step 1, one seed, validation split): it shows whether EuroSAT saturates, what a full run costs and, because its first twelve rounds repeat a `smoke` run, whether a run reproduces on the GPU | diagnostic |
| `t0` | `ho-fl-train` on Qwen / AG News with the zeroth-order part on and off, seeds 43 and 44 | E0-b |
| `feedback` | The report of `digest.py`: state of the gate, answers to the questions, findings with the file to change or the decision to take, the numbers, the end of every failed log, what to run next | |

A stage does not start when a stage it builds on ran and did not pass (`REQUIRES` in `e0.py`; `--force` overrides):
`selfcheck` and `memory` need `baseline` and `tests`; `smoke` and `pilot` also need `trace`, `data` and `selfcheck`;
`pilot` needs the two-hour run of the same method to have passed; `t0` needs `baseline` only. A stage that never ran
blocks nothing. Outcomes and wall times are in `<root>/stages.json`.

## The feedback report

`digest.py` turns the collected results into `feedback/feedback_E0.md`, in Chinese like the milestone reports. It
only reads the dictionary `collect` builds, so its rules are tested without running anything
(`tests/sat/test_sat_e0_pipeline.py`). A finding has a level: `block` (a gate item failed), `fix` (code or pipeline
should change; the text names the file), `decide` (a choice that affects the experiments, for the research lead),
`note` and `good`. The thresholds behind the findings are the constants at the top of `digest.py`; each cites the
section of the brief it comes from.

## What a stage prints

The page shows where the session is at any moment, and conclusions rather than raw logs:

- Every stage starts with a line saying what it is about to do and ends with its outcome and duration.
- A training run prints one line per round (participants, training loss, seconds, time elapsed and left) and one
  per evaluation; the checks and the memory calibration print a line per item or measurement as it completes; the
  tests print the progress lines of pytest and the names of failed tests; the data stage prints one line per partition.
- A command that prints nothing for 30 seconds (a download, a model being loaded) gets a line from the pipeline with
  the time so far and the command's last line, so a slow command can be told from a stuck one.
- The baseline regression says why it runs when it does (which file changed, or which GPU or PyTorch differs).

Tracebacks, tables of passing checks and library output stay in the logs under `<root>`, which the result bundle
holds; the report quotes the end of the log of whatever failed.

## Notes

- `expected.json` holds the trace and partition hashes recorded on the development machine and quoted in the
  milestone reports. The pipeline says whether what it builds has the same hashes; a difference is reported as
  something to fix in that file, not as a failure (floating-point results may differ in the last bit between
  platforms, and a trace hash depends on whether the land mask is installed).
- The training runs evaluate on the validation split: no server step has been selected yet, so nothing in this
  session looks at the test split.
- The seed-42 pair of E0-b is read from `experiments/p1/references/v1_1121_qwen.json` (runs `A-b0.75` and `B-b0.75`).
  The latter spelled the averaging rule as "first-order reports only, divided by K"; with the zeroth-order part off
  that is arithmetically the default rule, and gives the same parameters (checked on the tiny model).
- Re-running a stage reuses finished runs. An unfinished run is moved aside to `<run>.stale-N` and repeated.
- `pilot` and `t0` start a run only if its expected duration fits the remaining budget (`--budget-hours` or
  `--deadline`); `pilot` takes the expectation from the timings of the `smoke` run of the same method. A run that
  takes several times what it should is stopped, so that the session reaches its report.
- With `--tiny` the models and data are synthetic and the device is the CPU: it validates the pipeline, not the method.
