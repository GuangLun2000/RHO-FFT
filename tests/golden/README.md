# Golden values

Files in this directory are `golden.json` files written by `experiments/p0/p0.py golden` on a target GPU, renamed to
describe the hardware (for example `a100-torch2.6.0.json`). Each holds, for the three smoke configurations of the
brief (section 8), the per-round `task_loss`, the evaluation rows and the final parameter hash of the instrumented
revision, together with the GPU model, driver and library versions they were recorded with.

Golden values are hardware-dependent: a file is only meaningful on the GPU model and torch version it names.
`tests/test_gpu_golden.py` (`pytest -m gpu`) uses the file that matches the current device and skips otherwise.

No file has been recorded yet.
