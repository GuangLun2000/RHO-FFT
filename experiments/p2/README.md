# P2, first part: the dose-response experiment (E2)

> Earlier phase (brief v1.1, git history only), not part of the satellite work; "the brief" below is that one.
> Its end-to-end comparison is suspended (brief v2, header table).

`p2.py` runs experiment E2 of the brief (hypothesis H2, the quorum crossover) and writes a feedback report.
`notebooks/RHO-FFT_colab.ipynb` drives it on Colab after the baseline regression of `experiments/p0`.

```bash
python experiments/p2/p2.py base dose feedback --root outputs/p2 --jobs 2 --budget-hours 4
python experiments/p2/p2.py all --tiny --jobs 2 --root outputs/p2-tiny     # CPU dry run, about two minutes
python experiments/p2/plot_dose.py outputs/p2/feedback/feedback_P2.json outputs/p2/feedback
```

| Stage | What it does | Result |
| --- | --- | --- |
| `base` | 40 rounds of HO-FL with the default settings; its last state is the common starting point | `runs/<model>-s<seed>-dose-base/` |
| `dose` | Writes the schedule and fault files, then runs the arms in the order m = 2, 1, 3, 0, 4, 5 under the time budget | `inputs/`, `runs/` |
| `feedback` | Loss, damage, accuracy and influence tables per fault strength, the H2 computation, queue and timing | `feedback/` |

- The arms are defined in `ho_fl/dose.py`: dose m (honest first-order reports on the scarce segment next to the
  unreliable one) x six aggregation rules x {no fault, `flip20`, `flip1`}, 20 rounds each from the same state on
  the same cohorts. `quorum_robust` coincides by construction with `fo_robust` (m >= 2) or `robust_all` (m <= 1)
  and is filled in from them. `quorum_hold` and `quorum_dilute` run without zeroth-order updates.
- Runs that hit non-finite values end as `DIVERGED`; that is a result and is counted as the worst outcome.
- `influence.jsonl` of a run with faults records, per round and layer, how far the faulty report moved the
  aggregate relative to what it moves the equal-weight mean.
- The end-to-end resilience experiment (E3) is not implemented yet; its configuration depends on these results.
