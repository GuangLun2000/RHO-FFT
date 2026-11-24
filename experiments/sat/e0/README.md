# E0: correctness checks, memory calibration and alignment with HO-FL

`e0.py` runs everything gate G0 of the brief asks for and writes one feedback report. Stage S of
`notebooks/RHO-FFT_colab.ipynb` drives it on Colab; the same commands work on any machine with a CUDA GPU.

```bash
python experiments/sat/e0/e0.py env tests trace data --root outputs/sat/e0        # no GPU needed
python experiments/sat/e0/e0.py selfcheck memory smoke --root outputs/sat/e0      # about 25 minutes of GPU time
python experiments/sat/e0/e0.py t0 --root outputs/sat/e0 --budget-hours 1         # four HO-FL runs, about 40 minutes
python experiments/sat/e0/e0.py feedback --root outputs/sat/e0                    # feedback_E0.md, .json and a zip
python experiments/sat/e0/e0.py all --tiny --root outputs/sat/e0-tiny             # CPU dry run, about 3 minutes
```

| Stage | What it does | Item of E0 |
| --- | --- | --- |
| `env` | Interpreter, packages, GPU and git state | |
| `tests` | The CPU test suite: upstream tests and `tests/sat` | E0-a 1 |
| `trace` | Builds the S1 trace (`ho-fl-sat-trace`) and compares 19 statistics with appendix A of the brief | E0-a 6 |
| `data` | Downloads HumAID (per-event release) and EuroSAT, builds the partitions of seeds 42, 43, 44 in the base and the forward-only ("zonly") variant, prints the statistics of seed 42 and all partition hashes | |
| `selfcheck` | `ho_fl.sat.selfcheck` on ViT / EuroSAT and Qwen / HumAID, and against upstream `train_client` on Qwen / AG News | E0-a 2 to 5 |
| `memory` | `ho_fl.sat.calibrate`: peak GPU memory of a participation at every cut, Qwen and ViT | section 7.6 |
| `smoke` | Two simulated hours of the full method on both tasks and of the three other implemented methods on EuroSAT | E0-a 7 |
| `t0` | `ho-fl-train` on Qwen / AG News with the zeroth-order part on and off, seeds 43 and 44 | E0-b |
| `feedback` | The E0-a checklist, hashes, memory tables, smoke timings and the three-seed estimate of E0-b | |

- `expected.json` holds the trace and partition hashes recorded on the development machine and quoted in the
  milestone reports. The pipeline says whether what it builds has the same hashes; a difference is reported, not an
  error (a trace hash depends on whether the land mask is installed).
- The seed-42 pair of E0-b is read from `experiments/p1/references/v1_1121_qwen.json` (runs `A-b0.75` and `B-b0.75`).
  The latter spelled the averaging rule as "first-order reports only, divided by K"; with the zeroth-order part off
  that is arithmetically the default rule, and gives the same parameters (checked on the tiny model).
- Re-running a stage reuses finished runs. An unfinished run is moved aside to `<run>.stale-N` and repeated.
- `t0` starts a run only if its expected duration fits the remaining budget (`--budget-hours` or `--deadline`).
- With `--tiny` the models and data are synthetic and the device is the CPU: it validates the pipeline, not the method.
