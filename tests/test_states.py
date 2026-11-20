"""Saving of global states."""
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from ho_fl.io import tensor_hash
from ho_fl.states import save_state, state_path


def make_state():
    generator = torch.Generator().manual_seed(0)
    return {f"m.layers.{i}.lora_{ab}.weight": torch.randn(3, 4, generator=generator) for i in range(3) for ab in "AB"}


def test_state_path_layout(tmp_path):
    assert state_path(tmp_path, 40) == tmp_path / "states" / "round_0040.safetensors"


def test_save_state_is_exact_and_leaves_the_state_untouched(tmp_path):
    state = make_state()
    before = {name: tensor.clone() for name, tensor in state.items()}
    digest = save_state(state_path(tmp_path, 3), state, {"round": 3, "seed": 42})
    assert digest == tensor_hash(state)
    assert all(torch.equal(state[name], before[name]) for name in state)
    loaded = load_file(str(state_path(tmp_path, 3)))
    assert set(loaded) == set(state) and all(torch.equal(loaded[name], state[name]) for name in state)
    with safe_open(str(state_path(tmp_path, 3)), framework="pt") as handle:
        assert handle.metadata() == {"round": "3", "seed": "42", "parameter_sha256": digest}
