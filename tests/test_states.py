"""Saving and loading of global states."""
import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from ho_fl.io import tensor_hash
from ho_fl.states import load_state, save_state, state_path


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
        assert handle.metadata() == {"round": "3", "seed": "42", "parameter_sha256": digest,
                                     "tensor_names": json.dumps(list(state))}


def test_load_state_restores_the_exact_tensors_in_reference_order(tmp_path):
    state = make_state()
    digest = save_state(tmp_path / "s.safetensors", state, {"round": 3})
    reference = {name: torch.zeros_like(tensor) for name, tensor in reversed(list(state.items()))}
    loaded, record = load_state(tmp_path / "s.safetensors", reference)
    assert list(loaded) == list(reference)
    assert all(torch.equal(loaded[name], state[name]) for name in state)
    # The hash is taken in the order the state was saved in, whatever the order of the reference.
    assert record == {"parameter_sha256": digest, "metadata": {"round": "3", "parameter_sha256": digest}}
    assert load_state(tmp_path / "s.safetensors", state)[1] == record


def test_load_state_rejects_files_that_do_not_fit(tmp_path):
    state = make_state()
    save_state(tmp_path / "s.safetensors", state)
    names = list(state)
    with pytest.raises(ValueError, match="tensor names differ"):
        load_state(tmp_path / "s.safetensors", {name: state[name] for name in names[:-1]})
    with pytest.raises(ValueError, match="expected"):
        load_state(tmp_path / "s.safetensors", {**state, names[0]: torch.zeros(2, 2)})
    save_file({name: tensor + 1 for name, tensor in state.items()}, str(tmp_path / "bad.safetensors"),
              metadata={"parameter_sha256": "0" * 64})
    with pytest.raises(ValueError, match="parameter_sha256"):
        load_state(tmp_path / "bad.safetensors", state)
