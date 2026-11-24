"""The two-level partition and the data helpers of the satellite scenario (brief section 7).

The partition is checked on synthetic label and event arrays with the sizes of the real datasets, so the numbers
of the brief (446 tweets or 135 images per satellite, 70% main events) can be verified without downloading them.
"""
import json
from dataclasses import replace

import numpy as np
import pytest

from ho_fl.batches import ClientStream
from ho_fl.randomness import stable_seed
from ho_fl.sat import datasets
from ho_fl.sat.config import TASKS, SatConfig
from ho_fl.sat.datasets import (HUMAID_EVENTS, allocate_counts, build_manifest, dirichlet, fixed_subset, humaid_prompt,
                                largest_remainder, load_manifest, load_task_data, manifest_statistics, verbalizers)
from ho_fl.sat.roles import Roles
from ho_fl.sat.testing import build_tiny_assets

# Training examples per event of QCRI/HumAID-events at the pinned revision, in event name order (53,531 in all).
HUMAID_TRAIN = (5163, 1569, 2753, 1094, 1060, 5329, 4384, 6378, 6579, 5094, 1157, 840, 1536, 5588, 519, 1316, 1370, 1410, 392)


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    return build_tiny_assets(tmp_path_factory.mktemp("sat-data"))


def humaid_shaped(seed=0, classes=10):
    """Labels and events with the event sizes of HumAID; the class mix differs from event to event."""
    rng = np.random.default_rng(seed)
    events = np.repeat(np.arange(19), HUMAID_TRAIN)
    mix = rng.dirichlet(np.full(classes, 0.6), size=19)
    labels = np.concatenate([rng.choice(classes, size=count, p=mix[event]) for event, count in enumerate(HUMAID_TRAIN)])
    split = {"labels": labels.tolist(), "events": events.tolist()}
    small = {"labels": labels[::20].tolist(), "events": events[::20].tolist()}
    return {"classes": [f"c{i}" for i in range(classes)], "events": list(HUMAID_EVENTS), "train": split, "validation": small,
            "test": small, "meta": {"sha256": f"synthetic-{seed}"}}


def eurosat_shaped(seed=0):
    """Labels with the size of the EuroSAT training split (16,200) and unequal classes."""
    counts = np.array([1800, 1800, 1800, 1500, 1500, 1200, 1500, 1800, 1500, 1800])
    labels = np.random.default_rng(seed).permutation(np.repeat(np.arange(10), counts))
    split = {"labels": labels.tolist()}
    return {"classes": [f"c{i}" for i in range(10)], "train": split, "validation": {"labels": labels[::3].tolist()},
            "test": {"labels": labels[1::3].tolist()}, "meta": {"sha256": f"synthetic-{seed}"}}


# --- helpers -----------------------------------------------------------------------------------------------------------

def test_largest_remainder_and_allocation():
    assert largest_remainder([1, 1, 1], 10).tolist() == [4, 3, 3]
    assert largest_remainder([0.5, 0.25, 0.25], 135).tolist() == [67, 34, 34]
    assert largest_remainder([3, 0, 1], 0).tolist() == [0, 0, 0]
    assert allocate_counts([0.5, 0.5, 0.0], 10, [5, 5, 5]).tolist() == [5, 5, 0]
    # A class that runs out is made up from the others in proportion to their weights ...
    assert allocate_counts([0.8, 0.1, 0.1], 10, [2, 50, 50]).tolist() == [2, 4, 4]
    assert allocate_counts([0.6, 0.3, 0.1], 20, [3, 50, 50]).tolist() == [3, 13, 4]
    # ... and from whatever is in stock when no weight is left there.
    assert allocate_counts([1.0, 0.0, 0.0], 6, [2, 3, 1]).tolist() == [2, 3, 1]
    with pytest.raises(ValueError):
        allocate_counts([1.0, 1.0], 7, [3, 3])
    for seed in range(20):
        rng = np.random.default_rng(seed)
        stock = rng.integers(0, 40, size=6)
        total = int(rng.integers(0, stock.sum() + 1))
        counts = allocate_counts(rng.random(6), total, stock)
        assert counts.sum() == total and (counts <= stock).all() and (counts >= 0).all()


def test_dirichlet_draws():
    draw = dirichlet(np.random.default_rng(1), [2.0, 0.0, 3.0, 0.0])
    assert draw.sum() == pytest.approx(1.0) and draw[1] == draw[3] == 0 and (draw[[0, 2]] > 0).all()
    samples = np.array([dirichlet(np.random.default_rng(seed), [5.0, 15.0, 30.0]) for seed in range(400)])
    assert np.allclose(samples.mean(0), [0.1, 0.3, 0.6], atol=0.01)


def test_fixed_subset_depends_on_its_name_only():
    subset = fixed_subset(15160, 4000, "t1-test-subset")
    assert len(subset) == len(set(subset)) == 4000 and subset == sorted(subset) and max(subset) < 15160
    assert subset == fixed_subset(15160, 4000, "t1-test-subset") and subset != fixed_subset(15160, 4000, "t1-validation-subset")
    assert fixed_subset(300, 4000, "t1-test-subset") == list(range(300))


def test_prompt_and_verbalizers():
    classes = ["caution_and_advice", "not_humanitarian", "sympathy_and_support"]
    assert humaid_prompt("Roads are closed", classes) == ("Tweet: Roads are closed\n"
                                                           "Category (A caution and advice, B not humanitarian, C sympathy and support):")
    assert verbalizers(10) == [" A", " B", " C", " D", " E", " F", " G", " H", " I", " J"]
    assert len(HUMAID_EVENTS) == 19 and list(HUMAID_EVENTS) == sorted(HUMAID_EVENTS) and sum(HUMAID_TRAIN) == 53531


# --- level one and two on arrays of the real sizes ----------------------------------------------------------------------

def test_humaid_partition_has_the_sizes_of_the_brief():
    data = humaid_shaped()
    manifest = build_manifest(TASKS["t1"], data, 42)
    labels, events = np.array(data["train"]["labels"]), np.array(data["train"]["events"])
    assert manifest["examples_per_satellite"] == 446 and len(manifest["satellites"]) == 120
    assert len(manifest["unused_indices"]) == 53531 - 120 * 446 == 11
    # 70% of a plane (3,122 of 4,460) come from its main events; planes whose events are smaller take them whole.
    main = {plane: sum(HUMAID_TRAIN[e] for e in range(19) if e % 12 == plane) for plane in range(12)}
    for entry in manifest["plane_summary"]:
        plane = entry["plane"]
        assert entry["main_events"] == [e for e in range(19) if e % 12 == plane] and entry["main_available"] == main[plane]
        assert entry["main_taken"] == min(3122, main[plane])
        assert sum(entry["event_counts"]) == sum(entry["label_counts"]) == 4460
        assert sum(entry["event_counts"][e] for e in entry["main_events"]) >= entry["main_taken"]
    assert [entry["main_taken"] for entry in manifest["plane_summary"]] == [3122, 3122, 3122, 2410, 2430, 3122, 3122, 3122, 3122, 3122, 1157, 840]
    for record in manifest["satellites"]:
        assert len(record["indices"]) == 446 and record["held_out_examples"] == 0
        assert record["label_counts"] == np.bincount(labels[record["indices"]], minlength=10).tolist()
        assert record["data_seed"] == stable_seed(42, "sat-data", record["sid"])
        assert (record["plane"], record["slot"], record["kind"]) == (record["sid"] // 10, record["sid"] % 10, Roles(12, 10).kind(record["sid"]))
    # Every satellite of a plane draws from that plane's examples only.
    for plane in range(12):
        members = [index for record in manifest["satellites"] if record["plane"] == plane for index in record["indices"]]
        assert np.bincount(events[members], minlength=19).tolist() == manifest["plane_summary"][plane]["event_counts"]
    stats = manifest_statistics(manifest)
    assert stats["share_of_data_by_class"] == pytest.approx({"B": 0.1, "H": 0.4, "Z": 0.5})
    assert stats["main_share_by_plane"][0] == pytest.approx(0.7) and stats["main_share_by_plane"][11] == pytest.approx(840 / 4460, abs=1e-4)
    assert 0.05 < stats["tv_satellite_to_plane"] < stats["tv_plane_to_overall"] + 0.3


def test_eurosat_partition_has_the_sizes_of_the_brief():
    data = eurosat_shaped()
    manifest = build_manifest(TASKS["t2"], data, 42)
    labels = np.array(data["train"]["labels"])
    assert manifest["examples_per_satellite"] == 135 and manifest["unused_indices"] == []
    flat = [index for record in manifest["satellites"] for index in record["indices"]]
    assert sorted(flat) == list(range(16200))
    for entry in manifest["plane_summary"]:
        assert sum(entry["label_counts"]) == 1350 and sum(entry["class_prior"]) == pytest.approx(1.0)
    # The first plane follows its Dirichlet(0.5) prior exactly (nothing has run out yet); planes are skewed.
    first = manifest["plane_summary"][0]
    assert first["label_counts"] == largest_remainder(first["class_prior"], 1350).tolist()
    stats = manifest_statistics(manifest)
    assert stats["tv_plane_to_overall"] > 0.3 and stats["classes_per_satellite_min"] >= 1
    assert all(len(record["indices"]) == 135 for record in manifest["satellites"])
    assert sum(np.bincount(labels[flat], minlength=10)) == 16200


def test_partition_is_deterministic_and_depends_on_the_seed():
    data = humaid_shaped()
    first, second, other = (build_manifest(TASKS["t1"], data, seed) for seed in (42, 42, 43))
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["partition_sha256"] == second["partition_sha256"] != other["partition_sha256"]
    # The partition hash covers the examples of every satellite and nothing that depends on the environment.
    assert datasets.partition_hash({**first, "software": {"numpy": "0"}, "data_sha256": "x"}) == first["partition_sha256"]
    assert [r["indices"] for r in first["satellites"]] != [r["indices"] for r in other["satellites"]]
    with pytest.raises(ValueError):
        build_manifest(TASKS["t1"], data, 42, variant="iid")


def test_forward_only_variant_of_humaid():
    data = humaid_shaped()
    manifest = build_manifest(TASKS["t1"], data, 42, variant="zonly")
    events = np.array(data["train"]["events"])
    held = manifest["held_out"]
    assert held["names"] == ["puebla_mexico_earthquake_2017", "srilanka_floods_2017"] and held["ids"] == [17, 18]
    assert held["train_examples"] == 1410 + 392 and held["per_forward_satellite"] == 30
    assert held["test_indices"] == np.flatnonzero(np.isin(np.array(data["test"]["events"]), [17, 18])).tolist()
    for record in manifest["satellites"]:
        own = int(np.isin(events[record["indices"]], [17, 18]).sum())
        assert len(record["indices"]) == 446  # the same amount of data as in the base partition
        assert own == record["held_out_examples"] == (30 if record["kind"] == "Z" else 0)
    # Main planes are recomputed over the 17 remaining events; 70% of the 4,310 regular examples of a plane.
    assert [entry["main_events"] for entry in manifest["plane_summary"][:6]] == [[0, 12], [1, 13], [2, 14], [3, 15], [4, 16], [5]]
    assert manifest["plane_summary"][0]["main_taken"] == 3017 and sum(manifest["plane_summary"][0]["label_counts"]) == 4310
    assert len(manifest["unused_indices"]) == 53531 - 120 * 446
    assert manifest_statistics(manifest)["held_out"]["per_forward_satellite"] == 30


def test_forward_only_variant_of_eurosat():
    data = eurosat_shaped()
    manifest = build_manifest(TASKS["t2"], data, 42, variant="zonly")
    labels = np.array(data["train"]["labels"])
    held = manifest["held_out"]
    assert (held["kind"], held["ids"], held["train_examples"], held["per_forward_satellite"]) == ("class", [8], 1500, 25)
    for record in manifest["satellites"]:
        own = int((labels[record["indices"]] == 8).sum())
        assert len(record["indices"]) == 135 and own == (25 if record["kind"] == "Z" else 0)
    assert all(entry["class_prior"][8] == 0 for entry in manifest["plane_summary"])
    assert manifest["unused_indices"] == []


# --- the tiny assets through the loaders --------------------------------------------------------------------------------

@pytest.mark.parametrize("task", ["t1", "t2"])
def test_manifest_file_of_a_run(task, assets):
    cfg = SatConfig(**assets[task]).resolve()
    data = load_task_data(cfg)
    manifest, manifest_id, raw = load_manifest(cfg, data)
    assert cfg.manifest_path.exists() and json.loads(raw) == manifest and len(manifest_id) == 64
    assert load_manifest(cfg, data)[1] == manifest_id  # read back, not rebuilt
    assert manifest["examples_per_satellite"] == 21 and manifest["data_sha256"] == data["meta"]["sha256"]
    assert manifest["labels_sha256"] == datasets.labels_hash(data) and manifest["partition_sha256"] == datasets.partition_hash(manifest)
    streams = [ClientStream(record, cfg.batch_size) for record in manifest["satellites"]]
    assert streams[5].next() != streams[6].next() and set(streams[5].next()) <= set(manifest["satellites"][5]["indices"])
    zonly = replace(cfg, variant="zonly").resolve()
    other, other_id, _ = load_manifest(zonly, data)
    assert other_id != manifest_id and other["held_out"]["per_forward_satellite"] >= 1
    # The held-out examples that cannot be dealt out evenly are left unused, which may cost one example per satellite.
    assert len({len(record["indices"]) for record in other["satellites"]}) == 1 and other["examples_per_satellite"] in (20, 21)
    assert all(record["held_out_examples"] == (other["held_out"]["per_forward_satellite"] if record["kind"] == "Z" else 0)
               for record in other["satellites"])
    # A manifest belongs to the labels it was built from; the content hash of the data (pixels, texts) may differ.
    assert load_manifest(cfg, {**data, "meta": {**data["meta"], "sha256": "another"}})[1] == manifest_id
    with pytest.raises(ValueError):
        load_manifest(cfg, {**data, "train": {**data["train"], "labels": data["train"]["labels"][::-1]}})
    tampered = json.loads(raw)
    tampered["satellites"][0]["indices"] = tampered["satellites"][0]["indices"][::-1]
    cfg.manifest_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError):
        load_manifest(cfg, data)
    cfg.manifest_path.write_bytes(raw)
    with pytest.raises(ValueError):
        load_manifest(replace(cfg, manifest=str(zonly.manifest_path)), data)


def test_cached_data_are_checked_against_their_hash(assets):
    cfg = SatConfig(**assets["t1"]).resolve()
    data = load_task_data(cfg)
    assert len(data["train"]["text"]) == 2520 and len(data["events"]) == 19 and len(data["classes"]) == 4
    path = datasets.data_dir(cfg.cache_dir, "humaid", tiny=True) / "data.json"
    original = path.read_text()
    try:
        path.write_text(original.replace('"labels": [', '"labels": [3, ', 1))
        with pytest.raises(ValueError):
            load_task_data(cfg)
    finally:
        path.write_text(original)
    images = load_task_data(SatConfig(**assets["t2"]).resolve())
    assert images["train"]["images"].shape == (2520, 8, 8, 3) and images["train"]["images"].dtype == np.uint8
    with pytest.raises(FileNotFoundError):
        load_task_data(SatConfig(**{**assets["t1"], "cache_dir": str(path.parent / "nowhere")}).resolve())
