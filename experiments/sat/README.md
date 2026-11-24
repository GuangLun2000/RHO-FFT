# Satellite scenario experiments

One directory per experiment of the brief (`docs/RHO-FFT-agent-brief-v2.md`, section 10): its configuration files and
the pipeline script that runs it. Results go to `outputs/sat/` (or the `--root` given), traces and partitions to
`cache/sat/`; neither is tracked.

| Directory | Experiment | State |
| --- | --- | --- |
| `e0/` | Correctness checks, memory calibration, alignment with HO-FL | ready to run on Colab (stage S of the notebook) |
| `e1/` ... `e4/` | Main comparison, contributions, faults, topology | follow after gate G0 |
