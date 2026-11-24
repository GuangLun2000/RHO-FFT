# Fault specifications

> Earlier phase (brief v1.1, git history only): fault files of the suspended resilience comparison. The faults of
> the satellite scenario (brief v2, section 6.3 and experiment E3) are not defined here.

Fault files of the brief's E3 (section 7) for one unreliable device, read by `ho_fl/faults.py` (format in its
docstring). Paths in `experiments/p1/arms.json` are relative to the repository root.

| file | fault | device | rounds |
|---|---|---|---|
| `e3_lr_scale20_preferred_t1.json` | local learning rate x20 (`lr_scale`) | the best-ranked T1 device of the preferred set | 41 to 80 |
| `e3_label_shift_preferred_t1.json` | every local label replaced by the next class (`label_noise`, rate 1, `shift`) | the same device | 41 to 80 |
