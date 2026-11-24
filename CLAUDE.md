# RHO-FFT: orientation for coding agents

Research code for **hybrid-order federated fine-tuning over a satellite constellation**: 120 satellites fine-tune one
foundation model in orbit without a ground server. Each satellite trains the blocks above its cut with
backpropagation (first-order) and the blocks below it forward-only (zeroth-order). The repository is upstream
[HO-FL](https://github.com/HKU-WILL-Lab/ho-fl) plus the satellite package `ho_fl/sat/`.

## Read first

1. `docs/RHO-FFT-agent-brief-v2.md` (Chinese, long): what to build, how it is accepted, where to stop. Sections 0 to 4
   and 11 to 13 give the frame; after that read only the section of the module you touch. Appendix A holds the
   reference numbers the tests check, appendix B the defaults, appendix C the numbering shared by all documents.
2. The latest `docs/reports/P-S{n}.md`: what has run, what has not, deviations from the brief and open decisions.
   This is the status of the project; nothing else records it.
3. `docs/RHO-FFT-research-document-v4.md` and `docs/RHO-FFT-magazine-storyline-v1.md`: only for the reason behind a
   design. They do not specify code.

When the brief and the code disagree, the code wins; say so in the report.

## Map

| Path | Role |
| --- | --- |
| `ho_fl/sat/config.py` | `SatConfig`, task specifications, satellite classes, pinned dataset and model revisions |
| `ho_fl/sat/roles.py` | Satellite numbering, classes, plane roots, in-plane hops, cut table |
| `ho_fl/sat/scenario.py` | Orbits, eclipse, imaging, ground visibility, link geometry; builds and loads the trace |
| `ho_fl/sat/links.py`, `energy.py` | Link rates and delays; compute time, power, batteries |
| `ho_fl/sat/policy.py` | Per-round decision of every satellite: take part or not, cut, expected arrival |
| `ho_fl/sat/network.py` | Delivery of a round: in-plane sums, laser ring, broadcast, flooding, deadline, bytes |
| `ho_fl/sat/aggregate.py` | Replica groups (copy-on-write) and the dual-channel aggregation |
| `ho_fl/sat/methods.py` | The compared methods as sets of switches |
| `ho_fl/sat/backend.py`, `client.py`, `vision.py` | Model, data and one participation of a satellite; the ViT task |
| `ho_fl/sat/datasets.py` | HumAID, EuroSAT, the two-level partition and its manifest |
| `ho_fl/sat/engine.py` | Round loop, evaluation, run directory |
| `ho_fl/sat/selfcheck.py`, `calibrate.py` | Correctness checks on a loaded model; GPU memory per cut |
| `ho_fl/sat/cli.py` | Entry points `ho-fl-sat-*` (`tune`, `topology`, `figures` are placeholders) |
| `ho_fl/sat/testing.py` | Tiny synthetic assets for CPU tests and `--tiny` dry runs |
| `experiments/sat/e0/` | Pipeline of gate G0 (`e0.py`) and its feedback report (`digest.py`); `expected.json` pins the trace and partition hashes |
| `experiments/p0/` | Baseline regression against the upstream commit (still in use) |
| `tests/sat/` | Tests of the satellite code; `tests/` holds the upstream equivalence tests |
| `notebooks/` | The one Colab notebook and `build_notebook.py`, which generates it |

Not written yet (brief section 4.1): `protect.py`, `metrics.py`, `topology.py`, `figures.py`, and the methods beyond
the four in `methods.py`.

Files directly in `ho_fl/` fall into two groups:

- **Used by the satellite code**: `algorithms`, `aggregation`, `replay`, `reports`, `randomness`, `model`, `split`,
  `batches`, `config`, `costs`, `io`, `grid`, `harness`, `testing`.
- **Earlier phase, not used by it**: `scheduling`, `faults`, `probes`, `dose`, `supply`, `states`, `zo_check`,
  `analysis`, `live`, and the pipelines in `experiments/p1/`, `p2/`, `faults/`. Leave them alone and do not read them
  unless a task names them. Two exceptions: `experiments/p1/references/` is the reference of the alignment runs, and
  `tests/test_grid_and_analysis.py` loads `experiments/p1/p1.py`.

## Rules

- **Baseline equivalence.** With every new option at its default, the `ho-fl-*` entry points and the existing tests
  behave as before. New code goes into `ho_fl/sat/`. Existing files change only as listed in the brief, section 4.4,
  through keyword arguments with defaults; do not reorder or reformat them. Each such change needs a test showing
  that the default behaviour is unchanged.
- **Paired comparison.** Under one seed all methods share the trace, the partition, the classification head
  initialisation, the lottery values, each satellite's batch sequence, the zeroth-order directions and the fault
  draws. All randomness in `ho_fl/sat/` comes from `randomness.stable_seed`; no global generators.
- **No tuning for a winner.** Only the server step is selected, by the protocol of the brief, section 10.2.
  Unexpected results are reported as they are.
- **Traceable numbers.** Every number in a report comes from a logged run and names its run directory. What did not
  run is written as "未运行", never estimated. Failed and diverged runs are reported.
- **Pinned hashes.** A change that shifts a trace or partition hash in `experiments/sat/e0/expected.json` has to be
  deliberate and named in the report.
- **Milestones and gates.** Work proceeds by the milestones of the brief, section 11. At a gate (G0 to G3) write the
  report and stop until the research lead confirms.
- **Git.** Branch `sat`. Small commits; commit messages, code and comments in English. Do not push or rewrite
  commits without the research lead's say-so. Commit files by path: the working tree holds untracked local documents.
- **Reports.** One per milestone, `docs/reports/P-S{n}.md`, in Chinese, in the format of the brief, section 12.
  Describe configurations in plain words; a code such as M7 or E2 is a label, not an explanation.

## Running things

Experiments run on Colab with a GPU. The notebook clones the code from GitHub, so a change reaches Colab only after
the branch is pushed. Locally, run only CPU checks:

```bash
python -m pytest -q                                      # all CPU tests, about 3 minutes
python -m pytest tests/sat -q                            # satellite tests only
python experiments/sat/e0/e0.py all --tiny               # dry run of the G0 pipeline on tiny models
python experiments/p0/p0.py env golden regress --tiny    # dry run of the baseline regression
ho-fl-sat-trace --tables                                 # trace statistics against appendix A of the brief
```

`pyproject.toml` pins the Colab versions (Python 3.11); a local environment may differ, so name the versions a local
result was obtained with. The test on the real land mask needs the optional group `sat`. Colab runs Python 3.11:
no f-string features of 3.12 (the same quote inside a nested f-string, a backslash inside an expression).

## Colab sessions and the feedback loop

Code, then a Colab session, then the feedback report, then code again (`experiments/sat/README.md`):

- **Planning a session.** List the questions it answers and, for each run, the decision that depends on the answer.
  Leave out runs nobody would act on. Put cheap checks that can invalidate the rest first and let the pipeline skip
  what builds on a failed stage (`REQUIRES` in `experiments/sat/e0/e0.py`). Measure cost before spending it: time
  per participation, round and evaluation is logged by the engine and projected onto the remaining plan.
- **The report.** A pipeline ends with a report written for the research lead, in Chinese, by rules that are tested
  without running anything (`experiments/sat/e0/digest.py`): state of the gate, answers, findings (what to fix and in
  which file, what the lead has to decide, observations), the numbers, the end of failed logs, what to run next.
- **When the lead pastes a report back.** Save it under `colab_results_feedback/` (local, untracked) with the date,
  replace the "未运行" entries of the milestone report with its numbers, act on its findings, and adjust pipeline and
  notebook for the next session.
- **The notebook.** `notebooks/RHO-FFT_colab.ipynb` is generated: edit `notebooks/build_notebook.py`, run it, and
  let `tests/sat/test_sat_notebook.py` execute the cells with every external effect stubbed. Never edit the `.ipynb` by
  hand. The notebook must always reach its last cell: steps record failures instead of raising, and the GPU is
  released at the end and after any error.

## Do not spend tokens on

- `outputs/`, `cache/`, `colab_results_feedback/`, `ho_fl_paper/`: local, untracked runs, datasets, results and PDFs.
  Open a file there only when a task points at it.
- The second half of `README.md`: the upstream HO-FL README, kept verbatim.
- The documents of the earlier phase: removed from the working tree, in the git history of `main` if ever needed.
