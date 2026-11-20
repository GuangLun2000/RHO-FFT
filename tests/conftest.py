"""Shared fixtures. Everything here runs on CPU with the synthetic tiny assets from ``ho_fl.testing``."""
import pytest

from ho_fl.config import Config
from ho_fl.testing import build_tiny_assets


@pytest.fixture(scope="session")
def tiny_root(tmp_path_factory):
    return tmp_path_factory.mktemp("tiny-assets")


@pytest.fixture
def tiny_config(tiny_root, tmp_path):
    """Factory for resolved configs that train the tiny model: ``tiny_config("sst2", method="fedavg", ...)``."""
    counter = iter(range(1000))

    def make(dataset="sst2", **fields):
        overrides = build_tiny_assets(tiny_root, datasets=(dataset,))[dataset]
        values = {**overrides, "rounds": 2, "evaluate_every": 1, "output": str(tmp_path / f"run{next(counter)}")}
        values.update(fields)
        return Config(**values).resolve()

    return make
