# RHO-FFT

**Resilient Hybrid-Order Federated Fine-Tuning.** Research code built on [HO-FL](https://github.com/HKU-WILL-Lab/ho-fl), a
federated fine-tuning method in which every device trains the layers above its memory cut with backpropagation
(first-order) and the layers below it with forward-only gradient estimates (zeroth-order). RHO-FFT asks where each
layer's effective update actually comes from, and how scheduling, aggregation and uplink order should follow from that
when some devices are unreliable.

## Research questions

| | Question | Hypotheses |
| --- | --- | --- |
| RQ1 (measurement) | Which reports does each layer's update come from? How much do zeroth-order and first-order reports contribute? | H1: zeroth-order reports carry almost no information per step. H1a: the gain of biased sampling comes from less dilution of first-order reports. |
| RQ2 (resilience) | How does the damage of one unreliable first-order source depend on the first-order supply of its layers? | H2: robust averaging over first-order reports only works from about two honest reports on; below that, order-agnostic robust averaging (zeroth-order reports as ballast) is better. H3: biased sampling and uniform sampling fail in two different ways. |
| RQ3 (design) | Can supply-aware scheduling, aggregation and uplink order improve efficiency and resilience with the same device resources? | H4: deadlines thin the scarce layers first; sending first-order components bottom-up preserves them. |

The proposed configuration combines a tier quota with rotation, layer-wise robust aggregation that switches on a
first-order quorum, and scarcity-first uplink.

## Status

| Milestone | State |
| --- | --- |
| P0: baseline, instrumentation, regression | Done. The instrumented code reproduces upstream bit for bit on the Colab G4 GPU. |
| P1: measurement (E1a supply map, E1b probes, E1c ablation ladder) | Session 1 done (Qwen2.5-1.5B / AG News, seed 42, 12 runs). Preliminary: switching the zeroth-order updates off changes accuracy by less than 0.2 points; averaging only first-order reports gains 1.8 points, but doubling the server step alone gains 1.7. |
| P2: faults and resilience (E2 dose-response, E3 end-to-end) | Code for E2 done (counterfactual design). Session 2 is prepared: zeroth-order checks under the HO-FL authors' settings, E2, a numerical check of the estimator, and an E3 pilot. The E3 grid comes next. |
| P3: deadlines (E4) | Not started. |

Milestone reports and the implementation plan are in `docs/reports/` and `docs/plans/` (in Chinese); the reports
brought back from Colab are in `colab_results_feedback/`.

## Running the experiments on Colab

Open [`notebooks/RHO-FFT_colab.ipynb`](https://colab.research.google.com/github/GuangLun2000/RHO-FFT/blob/main/notebooks/RHO-FFT_colab.ipynb)
in Colab, choose a GPU runtime and run all cells. It clones this repository (no token needed), builds the pinned
environment, runs a CPU dry run and the baseline regression, then the experiments of the session within a time budget,
and ends with the feedback reports, a downloadable result bundle and the release of the GPU. This is the only notebook
to run; `notebooks/backup/` keeps those of earlier sessions for the record.

## Repository layout

| Path | Contents |
| --- | --- |
| `ho_fl/` | Training code: upstream HO-FL plus scheduling, aggregation (robust statistics, quorum), fault injection, probes, dose-response, supply metrics, the zeroth-order estimator check (`zo_check.py`) and analysis |
| `experiments/p0/` | Golden values and the baseline regression (invariants I1, I2) |
| `experiments/p1/` | Measurement pipeline: arms (`arms.json`), session-1 results (`references/`), figures |
| `experiments/p2/` | Dose-response pipeline (E2) and its figure |
| `experiments/faults/` | Fault specifications |
| `notebooks/` | The Colab notebook |
| `tests/` | CPU tests, including equivalence with the upstream training loop |

## Local development

```bash
python3 -m pip install uv
uv venv --python 3.11
uv pip install --torch-backend cu124 -e ".[dev]"
source .venv/bin/activate

python -m pytest -q                                              # CPU tests (GPU tests: -m gpu)
python experiments/p0/p0.py env golden regress --tiny            # dry runs on a tiny synthetic model
python experiments/p1/p1.py queue feedback --tiny --blocks 5,7,4
python experiments/p2/p2.py all --tiny
```

## Relation to HO-FL

`ho_fl/`, `pyproject.toml` and the section below were imported from
[HKU-WILL-Lab/ho-fl](https://github.com/HKU-WILL-Lab/ho-fl) at commit `ca74bed900b1af133bea2412bfdeba45447116dc`
(2026-09-26). Every option added here defaults to the upstream behaviour, so `ho-fl-train` with default settings
reproduces the upstream results; the tests compare the training loop with a frozen copy of the upstream one.

---

The rest of this file is the upstream HO-FL README.

# HO-FL

## Install

Python 3.11 and a CUDA GPU supporting BF16 (PyTorch 2.6.0 / CUDA 12.4):

```bash
python3 -m pip install uv
uv venv --python 3.11
uv pip install --torch-backend cu124 .
source .venv/bin/activate
```

## Train

```bash
ho-fl-train --model opt --dataset sst2 --method hofl --beta 0.75 --seed 42 --output outputs/opt-sst2
ho-fl-train --model qwen --dataset ag_news --partition dirichlet_05 --method hofl --beta 0.5 --seed 43 --output outputs/qwen-agnews
ho-fl-train --help
```

| Option | Values / Default |
| --- | --- |
| `--model` | `opt`, `qwen`, `smollm3` |
| `--dataset` | `sst2`, `boolq`, `sciq`, `ag_news`, `squad` |
| `--method` | `fedavg`, `fedprox`, `decomfl`, `hofl` |
| `--beta` | 0 to 1; default 0.75 for `hofl` |
| `--seed` | Base random seed; default 42 |
| `--partition` | `iid`; AG News also accepts `dirichlet_1`, `dirichlet_05`, `dirichlet_01` |
| `--k`, `--E`, `--q`, `--batch-size`, `--rounds` | Defaults: 6, 5, 2, 16, 160 |
| `--lr`, `--mu`, `--prox-lambda` | Defaults: 1e-5, 0.001, 0.001 |
| `--output`, `--cache-dir` | Output and download/cache directories |
| `--model-path`, `--data-path` | Optional local model and Datasets `save_to_disk` directories |
| `--manifest`, `--token-cache` | Optional local partition and tokenized-data files |
| `--device`, `--local-files-only` | Device selection and offline model loading |

Without local paths, models and datasets download automatically and client partitions are generated from `--seed`. For E=1/5/10 with equal processed-sample budgets, set `--rounds 800/160/80`, respectively.

## Other Commands

```bash
ho-fl-profile --method ho_small --length 512 --output outputs/profile.json
ho-fl-profile --help
ho-fl-memory --profiles profiles.csv --source-config outputs/qwen-seed42/config.json --output outputs/memory
ho-fl-memory --help
ho-fl-replay --output outputs/replay
ho-fl-replay --help
```
