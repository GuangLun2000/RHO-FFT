# P1: measurement (E1a, E1b, E1c)

`p1.py` runs the experiments of milestone P1 and writes a feedback report; `notebooks/P1_colab.ipynb` drives it on
Colab together with the baseline regression of `experiments/p0/p0.py`.

```bash
python experiments/p1/p1.py supply --root outputs/p1                       # E1a supply map, CPU, seconds
python experiments/p1/p1.py queue --root outputs/p1 --blocks 1 --jobs 2 --budget-hours 3
python experiments/p1/p1.py feedback --root outputs/p1                     # feedback_P1.md, CSV files, bundle
python experiments/p1/p1.py all --tiny --jobs 2 --root outputs/p1-tiny     # CPU dry run, about a minute
```

| Stage | What it does | Result |
| --- | --- | --- |
| `supply` | Supply map of the E1a scheduler settings next to appendix A of the brief; no training | `supply/` |
| `queue` | The arms of `arms.json` in priority order under a time budget | `runs/`, `queue.json`, `timings-<gpu>.json` |
| `feedback` | Accuracy table, the H1 and H1a computations, probe and layer tables, realised supply, timings | `feedback/` |

- `arms.json` defines the arms (A, B, C, D, A x2, A x3 and the FedAvg / DeComFL bounds) in three blocks: block 1
  decides H1a, block 2 adds H1 and the bounds, block 3 completes the ladder. Every arm keeps the HO-FL defaults.
- All HO-FL arms record layer statistics (`layers.jsonl`); the A arms also run the counterfactual first-order probe
  at rounds 1, 20, 40, 80 and 160 (`probe.jsonl`). Both are read-only: the arms reach the parameters they would
  reach without them.
- Each cost profile is probed with a three-round run before the queue starts. `--jobs N` lets N runs share the
  GPU, each in its own process; the number is lowered when the measured memory of one run does not allow it.
  Until a shared run has been timed, estimates assume that sharing brings no speed-up.
- Nothing persists between sessions. A queue reuses finished runs under the same root only.
- Tables for several roots (for example unpacked bundles of several sessions):
  `python -m ho_fl.analysis ROOT... --output DIR`.
