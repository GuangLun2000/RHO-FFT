"""Saving and loading of the global trainable state (the LoRA tensors held by the server).

A saved state is a safetensors file under ``<run>/states/``. Its metadata records where the state came from, the
``tensor_hash`` of the tensors and the order of the names the hash was taken in, so a file can be checked against
the hash reported by the run that wrote it.
"""
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .io import tensor_hash


def state_path(out, round_id):
    return Path(out) / "states" / f"round_{round_id:04d}.safetensors"


def save_state(path, state, metadata=None):
    """Write ``state`` (name -> tensor) to ``path`` and return its ``tensor_hash``. Does not modify ``state``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {name: tensor.detach().cpu().contiguous() for name, tensor in state.items()}
    meta = {str(key): str(value) for key, value in (metadata or {}).items()}
    meta["parameter_sha256"] = tensor_hash(tensors)
    meta["tensor_names"] = json.dumps(list(tensors))
    save_file(tensors, str(path), metadata=meta)
    return meta["parameter_sha256"]


def load_state(path, reference):
    """Read a saved state as ``name -> tensor``, ordered, shaped and placed like ``reference``.

    Returns the state and a record of what was loaded; ``parameter_sha256`` in the record is taken in the order
    the state was saved in, so it equals the hash reported by the run that saved it. Raises ``ValueError`` when the
    file does not fit ``reference`` or when its tensors do not match the hash stored in its metadata.
    """
    loaded = load_file(str(path))
    if set(loaded) != set(reference):
        raise ValueError(f"{path}: tensor names differ from the model "
                         f"(missing {sorted(set(reference) - set(loaded))[:3]}, unexpected {sorted(set(loaded) - set(reference))[:3]})")
    state = {}
    for name, expected in reference.items():
        tensor = loaded[name]
        if tensor.shape != expected.shape or tensor.dtype != expected.dtype:
            raise ValueError(f"{path}: {name} is {tuple(tensor.shape)} {tensor.dtype}, expected {tuple(expected.shape)} {expected.dtype}")
        state[name] = tensor.to(expected.device)
    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata() or {}
    order = json.loads(metadata.pop("tensor_names")) if "tensor_names" in metadata else list(reference)
    digest = tensor_hash({name: state[name] for name in order})
    if metadata.get("parameter_sha256", digest) != digest:
        raise ValueError(f"{path}: tensors do not match the stored parameter_sha256")
    return state, {"parameter_sha256": digest, "metadata": metadata}
