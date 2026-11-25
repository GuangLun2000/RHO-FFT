# E0: the first experiment of the satellite scenario (gate G0)

`e0.py` runs everything gate G0 of the brief asks for, plus three diagnostics that are cheap now and expensive to
learn later, and writes one feedback report. The same commands work on any machine with a CUDA GPU.

On Colab, `notebooks/RHO-FFT_colab.ipynb` calls `env`, then `memory`, `smoke` and `pilot` as its three steps, then
`feedback` (`COLAB` in `e0.py`). The check stages (`baseline`, `tests`, `trace`, `data`, `selfcheck`) were taken out
of the notebook on 2026-11-25 by the research lead, after they had passed there, to keep GPU time for measurements
and runs. They stay callable, and `all` still runs them. Without `trace` and `data` a training run builds the trace
and its partition itself on first use; the report compares the hashes each run records with `expected.json`. A
report of a session without checks has no checklist and judges no gate: the checklist of gate G0 is kept in the
milestone report.

```bash
python experiments/sat/e0/e0.py env baseline tests trace data --root outputs/sat/e0   # the baseline needs the GPU, the rest does not
python experiments/sat/e0/e0.py selfcheck memory smoke pilot --root outputs/sat/e0    # about 30 minutes of GPU time
python experiments/sat/e0/e0.py feedback --root outputs/sat/e0                        # feedback_E0.md, .json and a zip
python experiments/sat/e0/e0.py all --tiny --root outputs/sat/e0-tiny                 # CPU dry run, about 4 minutes
```

## What the stages are for

Every stage exists for one of six questions. Cheap stages that can invalidate the rest come first. The first two
questions belong to the check stages; a Colab session answers the other four (without the diagnostic of `selfcheck`).

| Question | Stages | What the answer decides |
| --- | --- | --- |
| Is upstream HO-FL unchanged? | `baseline`, `tests` | Everything on the GPU is skipped when it is not |
| Is the satellite code right on the real models? | `trace`, `data`, `selfcheck` | The training runs are skipped when it is not |
| How much memory does each cut need? | `memory` | The memory states of the system model (brief 7.6) |
| What does a run cost? | timings of `smoke` and `pilot` | The GPU budget of E1 to E4; whether the replay needs optimising first |
| Does the model learn, is the zeroth-order estimate faithful? | `smoke`, the diagnostic of `selfcheck` | Whether learning rate and perturbation radius are sane on the new tasks |
| Does EuroSAT saturate, is a day the right horizon? | `pilot` | Dataset or main metric of the vision task (gate G3), before its experiments run |

Since the session of 2026-11-25 two more are asked of the runs, because after two simulated hours the full method
trailed the method that trains the backbone satellites alone (server step 1 for both, one seed):

| Question | Stages | What the answer decides |
| --- | --- | --- |
| Where do the methods stand after a day, and on the text task? | `pilot` (all four methods), `smoke` (three methods on HumAID) | Whether that order is an effect of the short horizon or of the vision task; what to expect of the main comparison |
| How do the two channels share an update? | `channels` of every round of `smoke` and `pilot` | Whether the divisor dilutes the first-order step of the lower blocks and whether the zeroth-order updates carry a direction; the case for selecting the server step per method (brief 10.2) |

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
| `smoke` | Two simulated hours of every implemented method on EuroSAT (full method, centralised reference, backbone satellites only, zeroth-order flooding) and of the first three on HumAID | E0-a 7 |
| `pilot` | A whole simulated day of every implemented method on EuroSAT. Not a result (server step 1, one seed, validation split): it shows where the methods stand once a run is long enough to tell them apart, whether EuroSAT saturates, what a full run costs and, because its first twelve rounds repeat a `smoke` run, whether a run reproduces on the GPU. A day-long run starts only if the `smoke` run of its own method passed | diagnostic |
| `t0` | `ho-fl-train` on Qwen / AG News with the zeroth-order part on and off, seeds 43 and 44 | E0-b |
| `feedback` | The report of `digest.py`: state of the gate, answers to the questions, findings with the file to change or the decision to take, the numbers, the end of every failed log, what to run next | |

A stage does not start when a stage it builds on ran and did not pass (`REQUIRES` in `e0.py`; `--force` overrides):
`selfcheck` and `memory` need `baseline` and `tests`; `smoke` and `pilot` also need `trace`, `data` and `selfcheck`;
each run of `pilot` needs the two-hour run of its own method to have passed (the stage as a whole is held back only
when none did); `t0` needs `baseline` only. A stage that never ran blocks nothing. Outcomes and wall times are in `<root>/stages.json`; a stage is marked `running` when it starts, so one
that was stopped by hand reads as interrupted in the report, and a run it was in the middle of as `INTERRUPTED`.

## The feedback report

`digest.py` turns the collected results into `feedback/feedback_E0.md`, in Chinese like the milestone reports. It
only reads the dictionary `collect` builds, so its rules are tested without running anything
(`tests/sat/test_sat_e0_pipeline.py`). A finding has a level: `block` (a gate item failed), `fix` (code or pipeline
should change; the text names the file), `decide` (a choice that affects the experiments, for the research lead),
`note` and `good`. The thresholds behind the findings are the constants at the top of `digest.py`; each cites the
section of the brief it comes from.

Next to the report the stage writes `feedback/summary_card.md`, the table the notebook shows first (where the
session ran, how far each step got, one row per run), and the result bundle. `plots.py` draws the figures of the
session from `feedback/feedback_E0.json` into `feedback/figures/`: the validation accuracy of the methods over
simulated time, the training loss of the day-long runs, the two channels block by block, the memory per cut. It
runs in the notebook's own Python (matplotlib and the standard library only) and from the command line:
`python experiments/sat/e0/plots.py <root>/feedback/feedback_E0.json`.

## The two channels of an update

Every round logs `channels` in `rounds.jsonl` (`ho_fl.sat.aggregate.channel_statistics`): per block, how many of
the counted participants sent a first-order and how many a zeroth-order update, the norms of the two parts of the
mean update, and the cosine between the two sums. The update divides by all participants, so a block to which only
the backbone satellites send a first-order update takes a first-order step of a third of the size it has when they
train alone; the statistics say how large that effect is in a run and whether the zeroth-order updates of the other
satellites point the way the first-order ones do. It is a diagnostic of the first replica group, computed from
the terms of the aggregation with accumulators of its own: nothing of a run depends on it. The report averages it
over the rounds by segment of blocks (below the lowest hybrid cut, between the two, above, the head).

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
- Nothing is limited in time unless the caller sets a budget (`--budget-hours` or `--deadline`); the notebook sets
  none. With a budget, `smoke` starts no new run once it is used up, `pilot` and `t0` start a run only if its
  expected duration fits what is left (`pilot` takes the expectation from the timings of the `smoke` run of the same
  method), and a run that takes several times what it should is stopped.
- The estimates of the brief (section 10.4) were measured on Colab's G4; on any other GPU the report gives the
  measured times with the name of the GPU and derives no budget from them.
- A stage or a run that was stopped from outside (the stop button of the notebook, a lost runtime) is reported as
  interrupted, with the runs that had finished; it is not a failure of the code.
- With `--tiny` the models and data are synthetic and the device is the CPU: it validates the pipeline, not the method.
