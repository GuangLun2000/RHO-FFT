"""Satellite numbering, classes, cut table and the run configuration (brief sections 1.2, 5.3 and 8.1)."""
import json

import pytest

from ho_fl.config import Config
from ho_fl.sat.config import CLASS_SPECS, TASKS, TINY_TASKS, SatConfig, TrainConfig, parse_sat_config
from ho_fl.sat.roles import Roles, memory_floor, next_cut


def test_numbering_and_classes():
    roles = Roles(12, 10)
    assert roles.count == 120 and roles.sid(3, 7) == 37 and (roles.plane(37), roles.slot(37)) == (3, 7)
    assert [roles.kind(sid) for sid in range(10)] == ["B", "Z", "H", "Z", "H", "Z", "H", "Z", "H", "Z"]
    assert [len(roles.members(kind)) for kind in "BHZ"] == [12, 48, 60]
    assert roles.members("H", plane=2) == [22, 24, 26, 28] and roles.members(plane=11)[0] == 110
    assert [roles.tier(sid) for sid in (0, 2, 1)] == [0, 1, 2]
    assert roles.root(4) == 40 and roles.root(4, root_slot=2) == 42
    assert sorted(roles.grid_neighbours(0)) == [1, 9, 10, 110] and sorted(roles.grid_neighbours(115)) == [5, 105, 114, 116]


def test_hops_and_paths_to_the_plane_root():
    roles = Roles(12, 10)
    assert [roles.hops_to_root(sid) for sid in range(20, 30)] == [0, 1, 2, 3, 4, 5, 4, 3, 2, 1]
    assert roles.max_hops == 5
    assert roles.path_to_root(20) == [20] and roles.path_to_root(24) == [24, 23, 22, 21, 20]
    assert roles.path_to_root(26) == [26, 27, 28, 29, 20] and roles.path_to_root(25) == [25, 24, 23, 22, 21, 20]
    for sid in range(120):
        path = roles.path_to_root(sid)
        assert len(path) - 1 == roles.hops_to_root(sid) and path[-1] == roles.root(roles.plane(sid))
        assert all(b in roles.grid_neighbours(a) for a, b in zip(path, path[1:]))
    # After a handover the root is another slot of the plane.
    assert roles.hops_to_root(20, root_slot=2) == 2 and roles.path_to_root(29, root_slot=2) == [29, 20, 21, 22]


def test_cut_table():
    t1, t2 = TASKS["t1"], TASKS["t2"]
    assert [memory_floor(t1, kind, imaging) for kind, imaging in (("B", False), ("H", False), ("H", True), ("Z", False))] == [0, 8, 16, 28]
    assert [memory_floor(t2, kind, imaging) for kind, imaging in (("B", True), ("H", False), ("H", True), ("Z", True))] == [0, 3, 6, 12]
    assert [next_cut(t1, cut) for cut in (0, 8, 16, 24, 28)] == [8, 16, 24, 28, None]
    assert [next_cut(t2, cut) for cut in (3, 6, 9, 12)] == [6, 9, 12, None]
    assert (t1.depth, t1.model, t1.kind, t2.depth, t2.model, t2.kind) == (28, "qwen", "text", 12, "vit", "vision")
    assert TINY_TASKS["t1"].ladder == TINY_TASKS["t2"].ladder == (3, 6, 9, 12) and TINY_TASKS["t1"].model == "opt"


def test_defaults_are_those_of_the_brief():
    cfg = SatConfig().resolve()
    assert (cfg.task, cfg.method, cfg.variant, cfg.scenario, cfg.seed, cfg.hours) == ("t2", "m7", "", "s1", 42, 24.0)
    assert (cfg.round_s, cfg.slot_s, cfg.deadline_s, cfg.E, cfg.q, cfg.batch_size) == (600, 30, 480.0, 5, 2, 16)
    assert cfg.pi == {"B": 0.5, "H": 0.125, "Z": 0.1} and cfg.slowdown == {"B": 1.0, "H": 30.0, "Z": 150.0}
    assert (cfg.mu, cfg.lr, cfg.server_lr, cfg.t_fwd) == (1e-2, 1e-4, 1.0, 0.01)
    assert (cfg.rf_gain_dbi, cfg.laser_gbps, cfg.gs_sites, cfg.eval_every, cfg.protect, cfg.faults) == (20.0, 10.0, ["cambridge"], 6, True, "")
    assert (cfg.rounds, cfg.slots_per_round, cfg.satellites) == (144, 20, 120)
    t1 = SatConfig(task="t1").resolve()
    assert (t1.lr, t1.t_fwd, t1.spec.depth) == (1e-5, 0.04, 28)
    assert SatConfig(hours=2).resolve().rounds == 12
    # Expected participants per round: about six of each class.
    assert [cfg.pi[kind] * count for kind, count in (("B", 12), ("H", 48), ("Z", 60))] == [6.0, 6.0, 6.0]
    assert {kind: (spec.power_sun_w, spec.power_eclipse_w, spec.battery_wh, spec.solar_w, spec.bus_w) for kind, spec in CLASS_SPECS.items()} == {
        "B": (1000.0, 300.0, 0.0, 0.0, 0.0), "H": (40.0, 10.0, 100.0, 80.0, 30.0), "Z": (10.0, 0.0, 40.0, 25.0, 10.0)}


def test_paths_follow_the_cache_directory():
    cfg = SatConfig(task="t1", seed=43, cache_dir="/c").resolve()
    assert str(cfg.trace_dir) == "/c/sat/traces/s1" and str(cfg.manifest_path) == "/c/sat/manifests/t1-base-s43.json"
    zonly = SatConfig(variant="zonly", cache_dir="/c", tiny=True).resolve()
    assert zonly.data_variant == "zonly" and str(zonly.manifest_path) == "/c/sat/manifests/t2-tiny-zonly-s42.json"
    assert str(SatConfig(trace="/t", manifest="/m.json").resolve().trace_dir) == "/t"


@pytest.mark.parametrize("fields", [
    {"task": "t0"}, {"task": "t3"}, {"method": "m1"}, {"method": "fedavg"}, {"variant": "rf13"}, {"scenario": "s2"},
    {"scenario": "s9"}, {"hours": 0}, {"hours": 0.1}, {"mu": 0.0}, {"server_lr": 0.0}, {"E": 0}, {"eval_every": 0},
    {"round_s": 610}, {"deadline_s": 700}, {"pi": {"B": 0.5}}, {"pi": {"B": 0.5, "H": 2.0, "Z": 0.1}},
    {"slowdown": {"B": 1, "H": 0, "Z": 150}}, {"gs_sites": []}, {"gs_sites": ["oslo"]}, {"eval_split": "train"},
    {"faults": "faults.json"},
])
def test_resolve_rejects_invalid_settings(fields):
    with pytest.raises(ValueError):
        SatConfig(**fields).resolve()


def test_train_config_is_the_upstream_config_of_the_task():
    cfg = SatConfig(task="t1", seed=44, server_lr=3.0, model_path="/m", device="cpu").resolve()
    train = cfg.train_config(k=18, verbalizers=[" A", " B"])
    assert isinstance(train, Config) and isinstance(train, TrainConfig)
    assert (train.n, train.k, train.scenario, train.model, train.dataset, train.seed) == (120, 18, "s1", "qwen", "humaid", 44)
    assert (train.E, train.q, train.batch_size, train.mu, train.lr, train.server_lr) == (5, 2, 16, 1e-2, 1e-5, 3.0)
    assert (train.weight_decay, train.eps, train.betas) == (0.0005, 1e-8, (0.9, 0.999))  # as upstream HO-FL
    assert train.verbalizers == [" A", " B"] and train.max_length == 192 and train.zo_mode == "on"
    assert train.aggregation_name == "mean_all" and (train.model_path, train.device) == ("/m", "cpu")
    with pytest.raises(RuntimeError):
        train.resolve()
    assert SatConfig().resolve().train_config().model == "vit"
    assert SatConfig(task="t1", tiny=True).resolve().train_config().model == "opt"


def test_parse_sat_config(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"task": "t1", "method": "m4", "seed": 43, "pi": {"B": 1.0, "H": 0.5, "Z": 0.25}}))
    cfg = parse_sat_config(["--config", str(path), "--seed", "44", "--hours", "2", "--server-lr", "3", "--no-protect",
                            "--gs-sites", '["cambridge", "punta_arenas"]', "--output", "o"])
    assert (cfg.task, cfg.method, cfg.seed, cfg.hours, cfg.server_lr, cfg.protect, cfg.output) == ("t1", "m4", 44, 2.0, 3.0, False, "o")
    assert cfg.pi == {"B": 1.0, "H": 0.5, "Z": 0.25} and cfg.gs_sites == ["cambridge", "punta_arenas"] and cfg.lr == 1e-5
    assert parse_sat_config([], tiny=True, device="cpu").tiny is True
    path.write_text(json.dumps({"no_such_field": 1}))
    with pytest.raises(ValueError):
        parse_sat_config(["--config", str(path)])
    assert set(cfg.to_dict()) >= {"task", "method", "pi", "slowdown", "trace", "manifest", "tiny"}
