"""GPU regression test: the smoke configurations must reproduce the golden values recorded for this hardware.

Golden values depend on the GPU model and the library versions, so the test only runs when ``tests/golden/`` holds
a file recorded on the same GPU with the same torch version; otherwise it is skipped with the reason. Run with
``pytest -m gpu``. The pipeline in ``experiments/p0/p0.py`` performs the same comparison against values recorded in
the same session and does not need a committed file.
"""
import json
import os
from pathlib import Path

import pytest

from ho_fl import harness

REPO = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).parent / "golden"
pytestmark = pytest.mark.gpu


def matching_golden():
    import torch

    if not torch.cuda.is_available():
        return None, "no CUDA device"
    name = torch.cuda.get_device_name(0)
    for path in sorted(GOLDEN.glob("*.json")):
        golden = json.loads(path.read_text())
        environment = golden.get("environment", {})
        if not golden.get("tiny") and golden.get("complete") and environment.get("gpu_name") == name \
                and environment.get("torch") == torch.__version__:
            return golden, None
    return None, f"no golden values recorded for {name} with torch {torch.__version__}"


@pytest.mark.parametrize("name", ["hofl", "fedavg", "decomfl"])
def test_smoke_configuration_reproduces_the_golden_values(name, tmp_path):
    golden, reason = matching_golden()
    if golden is None:
        pytest.skip(reason)
    if golden["mode"] != "bitwise":
        pytest.skip("golden values were recorded on a device that is not bitwise reproducible; use `p0.py regress`")
    fields = {**golden["configs"][name]["fields"], "device": "cuda", "cache_dir": os.environ.get("HO_FL_CACHE_DIR", str(REPO / "cache"))}
    record = harness.run_training(fields, tmp_path / name, tree=REPO)
    assert record["status"] == "COMPLETED", record.get("log_tail")
    comparison = harness.compare_fingerprints(golden["configs"][name], harness.fingerprint(tmp_path / name))
    assert comparison["bitwise"] and comparison["hash_equal"], comparison["differences"]
