"""Contracts of the entry points that are not exercised by training (invariant I2).

``ho-fl-profile`` needs a CUDA device and ``ho-fl-memory`` needs its measurements, so here they are checked through
the pieces that can run on CPU: the simulation in ``memory.py`` on synthetic profiles, and the imports.
"""
import numpy as np
import pandas as pd

from ho_fl import memory, profile
from ho_fl.sampling import DepRoundSampler


def qwen_like_config(seed=42):
    cuts = np.random.default_rng(seed).permutation(np.repeat([8, 16, 24], 10))
    dimensions = {"8": 311296, "16": 622592, "24": 933888}
    sampler = DepRoundSampler([dimensions[str(c)] for c in cuts], 2, 6, 0.75, seed)
    return {"actual_cuts": cuts.tolist(), "bottom_dimensions": dimensions, "sampler_metadata": sampler.metadata()}


def test_memory_simulation_runs_on_the_config_layout_written_by_training():
    peaks = {"ho_small": 30e9, "ho_medium": 20e9, "ho_large": 10e9}
    profiles = pd.DataFrame([{"input_length": length, "method": method, "peak_allocated_bytes_max": peak * length / 512,
                              "peak_reserved_bytes_max": 1.1 * peak * length / 512}
                             for length in (512, 1024) for method, peak in peaks.items()])
    rounds, means, selections = memory.simulate(qwen_like_config(), profiles)
    assert len(means) == 4 * 2 and len(rounds) == 4 * 2 * 160 and len(selections) == 4 * 160
    default = means[(means.beta == 0.75) & (means.length == 512)].iloc[0]
    assert np.isclose(default.cut8_expected_count, 5.0) and np.isclose(default.cut24_expected_count, 0.5)


def test_profile_module_imports_and_keeps_its_entry_point():
    assert callable(profile.main) and callable(profile.measure) and callable(memory.main)
