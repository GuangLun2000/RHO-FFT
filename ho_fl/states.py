"""Saving of the global trainable state (the LoRA tensors held by the server).

A saved state is a safetensors file under ``<run>/states/``. Its metadata records where the state came from and
the ``tensor_hash`` of the tensors, so a file can be checked against the hash reported by the run that wrote it.
"""
from pathlib import Path

from safetensors.torch import save_file

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
    save_file(tensors, str(path), metadata=meta)
    return meta["parameter_sha256"]
