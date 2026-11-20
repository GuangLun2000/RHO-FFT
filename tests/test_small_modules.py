"""Characterisation of small upstream helpers: seeds, directions, data streams, collation, hashing, uplink bytes."""
import json

import pytest
import torch

from ho_fl.batches import ClientStream, collate
from ho_fl.config import Config
from ho_fl.io import append_jsonl, tensor_hash, write_json
from ho_fl.model import block_index, partition_parameters
from ho_fl.randomness import RoundDirections, client_directions, stable_seed
from ho_fl.train import logical_uplink


# --- randomness -------------------------------------------------------------------------------------------------

def test_stable_seed_values_are_frozen():
    # Literal values computed from HKU-WILL-Lab/ho-fl@ca74bed9. They fix every data stream and direction.
    assert stable_seed(42, "opt", 0, "data") == 3776837127
    assert stable_seed(42, "opt", 1, 0, "zo_master") == 1240567913
    assert stable_seed(42, "qwen", 160, 29, "zo_master") == 582584357
    assert stable_seed(7) == 989699928


def test_stable_seed_depends_on_every_part():
    base = stable_seed(42, "opt", 1, 0, "zo_master")
    variants = [stable_seed(43, "opt", 1, 0, "zo_master"), stable_seed(42, "qwen", 1, 0, "zo_master"),
                stable_seed(42, "opt", 2, 0, "zo_master"), stable_seed(42, "opt", 1, 1, "zo_master"),
                stable_seed(42, "opt", 1, 0, "data")]
    assert base not in variants and len(set(variants)) == len(variants)
    assert all(0 <= value < 2 ** 32 for value in [base, *variants])


def test_round_directions_are_deterministic_and_cached():
    params = {"m.layers.0.a": torch.zeros(3, 4), "m.layers.1.a": torch.zeros(3, 4)}
    first, second = client_directions(42, "opt", 5, 17), client_directions(42, "opt", 5, 17)
    assert first.master_seed == stable_seed(42, "opt", 5, 17, "zo_master")
    u = first.get(params, 0, 1)
    assert u["m.layers.0.a"] is first.get(params, 0, 1)["m.layers.0.a"]
    assert all(torch.equal(u[name], second.get(params, 0, 1)[name]) for name in params)
    assert not torch.equal(u["m.layers.0.a"], u["m.layers.1.a"])
    assert not torch.equal(u["m.layers.0.a"], first.get(params, 0, 0)["m.layers.0.a"])
    assert not torch.equal(u["m.layers.0.a"], first.get(params, 1, 1)["m.layers.0.a"])
    assert not torch.equal(u["m.layers.0.a"], RoundDirections(first.master_seed + 1).get(params, 0, 1)["m.layers.0.a"])
    assert first.used_seeds[0, 1] == stable_seed(first.master_seed, 0, 1)


# --- batches ----------------------------------------------------------------------------------------------------

def test_client_stream_is_reproducible_and_reshuffles_between_epochs():
    client = {"indices": list(range(100, 110)), "data_seed": 123}
    first, second = ClientStream(client, 4), ClientStream(client, 4)
    batches = [first.next() for _ in range(6)]
    assert batches == [second.next() for _ in range(6)]
    assert all(len(batch) == 4 and set(batch) <= set(client["indices"]) for batch in batches)
    # Ten examples give two full batches per epoch; the remainder is dropped and the order is redrawn.
    for epoch in range(3):
        seen = batches[2 * epoch] + batches[2 * epoch + 1]
        assert len(set(seen)) == 8
    assert first.batches_seen == 6
    assert [ClientStream({**client, "data_seed": 124}, 4).next() for _ in range(6)] != batches


def test_collate_pads_on_the_left():
    data = {"input_ids": [[5, 6, 7], [8]], "labels": [1, 0]}
    batch = collate(data, [0, 1], 1, "cpu")
    assert batch["input_ids"].tolist() == [[5, 6, 7], [1, 1, 8]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [0, 0, 1]]
    assert batch["labels"].tolist() == [1, 0]


# --- io ---------------------------------------------------------------------------------------------------------

def test_tensor_hash_depends_on_names_values_and_order():
    a, b = torch.arange(6.0).reshape(2, 3), torch.ones(4)
    reference = tensor_hash({"x": a, "y": b})
    assert reference == tensor_hash({"x": a.clone(), "y": b.clone()})
    assert reference == tensor_hash({"x": a.double(), "y": b})  # hashed as float32
    assert reference != tensor_hash({"y": b, "x": a})
    assert reference != tensor_hash({"x": a, "z": b})
    assert reference != tensor_hash({"x": a + 1e-6, "y": b})


def test_json_writers_reject_non_finite_values(tmp_path):
    write_json(tmp_path / "ok.json", {"b": 1, "a": 2.5})
    assert (tmp_path / "ok.json").read_text() == '{\n  "a": 2.5,\n  "b": 1\n}\n'
    append_jsonl(tmp_path / "rows.jsonl", {"b": 1, "a": None})
    append_jsonl(tmp_path / "rows.jsonl", {"c": 3})
    assert [json.loads(line) for line in (tmp_path / "rows.jsonl").read_text().splitlines()] == [{"a": None, "b": 1}, {"c": 3}]
    with pytest.raises(ValueError):
        write_json(tmp_path / "bad.json", {"a": float("nan")})
    with pytest.raises(ValueError):
        append_jsonl(tmp_path / "rows.jsonl", {"a": float("inf")})


# --- model helpers ----------------------------------------------------------------------------------------------

def test_block_index_and_partition_parameters():
    names = [f"base_model.model.model.decoder.layers.{i}.self_attn.{proj}.lora_{ab}.default.weight"
             for i in range(12) for proj in ("q_proj", "v_proj") for ab in "AB"]
    params = {name: torch.zeros(1) for name in names}
    assert block_index(names[0]) == 0 and block_index(names[-1]) == 11
    assert block_index("base_model.model.model.layers.27.self_attn.q_proj.lora_A.default.weight") == 27
    bottom, top = partition_parameters(params, 3)
    assert len(bottom) == 12 and len(top) == 36
    assert all(block_index(n) < 3 for n in bottom) and all(block_index(n) >= 3 for n in top)
    assert partition_parameters(params, 0)[0] == {} and partition_parameters(params, 12)[1] == {}
    with pytest.raises(AssertionError):
        block_index("lm_head.weight")


# --- logical uplink ---------------------------------------------------------------------------------------------

def test_logical_uplink_matches_the_reference_payloads():
    total = 294912  # trainable LoRA parameters of OPT-125M
    hofl = logical_uplink(Config().resolve(), total, 73728)
    assert (hofl["header_bytes"], hofl["zo_scalar_bytes"], hofl["fo_delta_bytes"]) == (18, 40, 884736)
    assert hofl["logical_uplink_bytes"] == 18 + 40 + 884736
    assert logical_uplink(Config().resolve(), total, 221184)["fo_delta_bytes"] == 294912
    fedavg = logical_uplink(Config(method="fedavg").resolve(), total, 0)
    assert (fedavg["zo_scalar_bytes"], fedavg["fo_delta_bytes"]) == (0, 4 * total)
    decomfl = logical_uplink(Config(method="decomfl").resolve(), total, total)
    assert (decomfl["zo_scalar_bytes"], decomfl["fo_delta_bytes"], decomfl["logical_uplink_bytes"]) == (40, 0, 58)
