"""The budgeted run queue, the analysis tables and the H1 / H1a decision rules."""
import importlib.util
import json
import threading
import time
from pathlib import Path

import pytest

from ho_fl import analysis, grid, harness
from ho_fl.config import Config

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("p1", REPO / "experiments" / "p1" / "p1.py")
p1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p1)


# --- queue --------------------------------------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def no_gpu_memory(monkeypatch):
    """The queue asks nvidia-smi how much memory the GPU has; these tests describe a machine where it cannot tell."""
    monkeypatch.setattr(grid, "gpu_memory", lambda: None)


def make_specs(count, key="m/hofl"):
    return [{"name": f"run{index}", "arm": f"arm{index}", "timing_key": key if index != 2 else "m/fedavg",
             "fields": {"rounds": 160, "method": "hofl"}} for index in range(count)]


class FakeRunner:
    """Stands in for harness.run_training: records calls, writes the status files and can fail or take time."""

    def __init__(self, duration=0.0, failing=()):
        self.calls, self.duration, self.failing = [], duration, set(failing)
        self.active, self.max_active, self.lock = 0, 0, threading.Lock()

    def __call__(self, fields, output, tree=None, label=None, revision=None, echo=None, extra_meta=None, stop_at=None):
        self.calls.append((label, fields["rounds"], stop_at))
        if label.startswith("preflight"):
            if any(name in label for name in self.failing):
                return {"status": "FAILED", "error": "CUDA out of memory", "timing": None}
            return {"status": "COMPLETED", "timing": {"round_s": 10.0, "eval_s": 5.0, "startup_s": 20.0}}
        if harness.run_status(output) == "COMPLETED":
            return {"status": "COMPLETED", "reused": True, "wall_s": 1.0}
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        harness.write_json(Path(output) / "config.json", {})  # written once data preparation is over: the next run may start
        time.sleep(self.duration)
        harness.write_json(Path(output) / "status.json", {"status": "COMPLETED"})
        with self.lock:
            self.active -= 1
        return {"status": "COMPLETED", "reused": False, "wall_s": 1900.0, "timing": {"round_s": 11.0, "eval_s": 5.0, "startup_s": 20.0}}


def queue(tmp_path, runner, specs, **options):
    return grid.run_queue(specs, root=tmp_path, tree=".", hardware="gpu-x", revision="abc", common=lambda fields: {"device": "cpu"},
                          echo=lambda text="": None, run=runner, **options)


def test_queue_probes_each_cost_profile_once_and_respects_the_budget(tmp_path):
    runner = FakeRunner(failing=("fedavg",))
    # A run is estimated at 1.25 * (20 + 17 * 5 + 160 * 10) = 2131 s: two fit into 4500 s, the third does not.
    record = queue(tmp_path, runner, make_specs(4), deadline=time.time() + 4500)
    assert [label for label, _, _ in runner.calls[:2]] == ["preflight m/hofl", "preflight m/fedavg"] and runner.calls[0][1] == 3
    statuses = {entry["name"]: entry["status"] for entry in record["queue"]}
    assert statuses == {"run0": "COMPLETED", "run1": "COMPLETED", "run2": "SKIPPED_PREFLIGHT", "run3": "COMPLETED"}
    assert record["queue"][2]["reason"] == "CUDA out of memory" and record["preflight_failures"] == {"m/fedavg": "CUDA out of memory"}
    assert all(stop is not None for label, _, stop in runner.calls if not label.startswith("preflight"))
    timings = harness.read_json(tmp_path / "timings-gpu-x.json")
    assert timings["m/hofl"]["source"] == "run3" and timings["m/hofl"]["round_s"] == 11.0 and "m/fedavg" not in timings

    tight = queue(tmp_path / "tight", FakeRunner(), make_specs(2), deadline=time.time() + 2500)
    assert [entry["status"] for entry in tight["queue"]] == ["COMPLETED", "COMPLETED"]
    none = queue(tmp_path / "none", FakeRunner(), make_specs(2), deadline=time.time() + 1000)
    assert [entry["status"] for entry in none["queue"]] == ["SKIPPED_BUDGET", "SKIPPED_BUDGET"] and none["queue"][0]["estimated_s"] == 2131


def test_queue_reuses_finished_runs_without_probing(tmp_path):
    first = FakeRunner()
    queue(tmp_path, first, make_specs(2))
    second = FakeRunner()
    record = queue(tmp_path, second, make_specs(2))
    assert [entry["reused"] for entry in record["queue"]] == [True, True] and not any(label.startswith("preflight") for label, _, _ in second.calls)


def test_queue_runs_several_jobs_at_a_time_and_budgets_for_the_worst_case(tmp_path):
    runner = FakeRunner(duration=1.5)  # longer than the half-second poll that hands the start lock on
    record = queue(tmp_path, runner, make_specs(2) + [dict(make_specs(4)[3])], jobs=2)
    assert runner.max_active == 2 and record["jobs"] == 2 and [entry["status"] for entry in record["queue"]] == ["COMPLETED"] * 3
    assert harness.read_json(tmp_path / "timings-gpu-x.json")["m/hofl"]["shared"]["jobs"] == 2
    # Until a shared run has been measured, two runs at a time are assumed to take twice as long each.
    slow = queue(tmp_path / "slow", FakeRunner(), make_specs(2), jobs=2, deadline=time.time() + 4000)
    assert [entry["status"] for entry in slow["queue"]] == ["SKIPPED_BUDGET", "SKIPPED_BUDGET"] and slow["queue"][0]["estimated_s"] == 4262


def test_memory_limits_the_number_of_jobs():
    timings = {"a": {"peak_mb": 30000.0}, "b": {"peak_mb": 10000.0}}
    assert grid.allowed_jobs(4, timings, {"a", "b"}, memory=(0.0, 96000.0)) == 2
    assert grid.allowed_jobs(4, timings, {"b"}, memory=(0.0, 96000.0)) == 4
    assert grid.allowed_jobs(4, timings, {"a"}, memory=(0.0, 24000.0)) == 1
    assert grid.allowed_jobs(3, {"a": {"peak_mb": None}}, {"a"}, memory=(0.0, 96000.0)) == 1  # a GPU without a measurement
    assert grid.allowed_jobs(3, timings, {"a"}, memory=None) == 3 and grid.allowed_jobs(1, timings, {"a"}, memory=(0.0, 1.0)) == 1


def test_preflight_fields_and_bundle(tmp_path):
    fields = {"rounds": 160, "probe_rounds": "1,20,160", "save_state_rounds": "40", "layer_stats": True}
    assert grid.preflight_fields(fields) == {"rounds": 3, "evaluate_every": 3, "probe_rounds": "1", "layer_stats": True}
    assert "probe_rounds" not in grid.preflight_fields({"rounds": 160}) and fields["rounds"] == 160
    for name in ("runs/a/ledger.jsonl", "runs/a/states/round_0040.safetensors", "runs/a/manifest.json.gz", "assets/model.bin", "queue.json"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("x")
    import zipfile
    with zipfile.ZipFile(grid.write_bundle(tmp_path, tmp_path / "feedback" / "bundle.zip")) as bundle:
        assert sorted(bundle.namelist()) == ["queue.json", "runs/a/ledger.jsonl"]


def test_a_run_past_its_stop_time_is_terminated(tmp_path):
    import sys
    progress = harness.launch([sys.executable, "-c", "import time; time.sleep(60)"], tmp_path / "out", tmp_path / "out.log",
                              stop_at=time.time() + 0.5)
    assert progress["timed_out"] and progress["wall_s"] < 30 and progress["returncode"] != 0


# --- decision rules -----------------------------------------------------------------------------------------------

def test_h1a_rule():
    base = {"A-b0": {42: 85.0}, "A-b0.75": {42: 86.0}, "Ax2-b0": {42: 85.2}, "Ax3-b0": {42: 85.3}}
    supported = analysis.decide_h1a({**base, "C-b0": {42: 85.8}})
    assert supported["verdict"] == "SUPPORTED" and supported["preliminary"] and supported["recovered_share"] == pytest.approx(0.8)
    assert supported["per_seed"][0] == pytest.approx({"seed": 42, "g_c": 0.8, "g_beta": 1.0, "g_lr": 0.3})
    assert analysis.decide_h1a({**base, "C-b0": {42: 85.6}})["verdict"] == "NOT_SUPPORTED"                 # 60% of the gain
    assert analysis.decide_h1a({**base, "C-b0": {42: 85.8}, "Ax3-b0": {42: 85.5}})["verdict"] == "NOT_SUPPORTED"   # a larger step does as well
    assert analysis.decide_h1a({**base, "C-b0": {42: 85.8}, "A-b0.75": {42: 84.9}})["verdict"] == "CANNOT_JUDGE"   # no gain to recover
    assert analysis.decide_h1a({"A-b0": {42: 85.0}, "A-b0.75": {42: 86.0}, "C-b0": {42: 85.8}})["verdict"] == "CANNOT_JUDGE"  # no controls
    assert analysis.decide_h1a({"A-b0": {42: 85.0}})["verdict"] == "NOT_RUN"
    three = {"A-b0": {42: 85.0, 43: 85.4, 44: 84.6}, "A-b0.75": {42: 86.0, 43: 86.6, 44: 85.8}, "C-b0": {42: 85.9, 43: 86.3, 44: 85.7, 45: 80.0},
             "Ax2-b0": {42: 85.1, 43: 85.5, 44: 84.8}, "Ax3-b0": {42: 85.2, 43: 85.3, 44: 84.9}}
    full = analysis.decide_h1a(three)
    assert full["seeds"] == [42, 43, 44] and not full["preliminary"] and full["verdict"] == "SUPPORTED" and full["g_c_sd"] is not None


def test_h1_rule():
    result = analysis.decide_h1({"A-b0": {42: 85.0, 43: 85.6, 44: 84.4}, "B-b0": {42: 84.9, 43: 85.5, 44: 84.5}, "A-b0.75": {42: 86.0}, "B-b0.75": {42: 84.0}})
    assert result["b0"]["verdict"] == "SUPPORTED" and result["b0"]["a_minus_b"] == pytest.approx(0.0333, abs=1e-3) and result["b0"]["seed_sd_of_a"] == pytest.approx(0.6)
    assert result["b0.75"]["verdict"] == "CANNOT_JUDGE" and result["b0.75"]["a_minus_b"] == pytest.approx(2.0)
    worse = analysis.decide_h1({"A-b0": {42: 85.0, 43: 85.2}, "B-b0": {42: 83.0, 43: 83.4}})
    assert worse["b0"]["verdict"] == "NOT_SUPPORTED" and worse["b0.75"]["verdict"] == "NOT_RUN"


def test_probe_and_layer_summaries():
    rows = [{"round": 1, "tier": 0, "scope": "pooled", "group": "zo", "cosine": c, "norm_ratio": r, "dimension": 10000} for c, r in ((0.01, 0.8), (0.03, 1.2), (None, None))]
    rows += [{"round": 1, "tier": 0, "scope": "layer", "layer": 3, "group": "zo", "cosine": 0.5, "norm_ratio": 1.0, "dimension": 100}, {"round": 1, "scope": "error", "error": "x"}]
    (entry,) = analysis.probe_summary(rows)
    assert (entry["clients"], entry["cosine_mean"], entry["null_cosine"], entry["norm_ratio_median"]) == (3, pytest.approx(0.02), pytest.approx(0.01), pytest.approx(1.0))
    assert entry["cosine_se"] == pytest.approx(0.01)
    layers = [{"round": r, "layer": layer, "segment": 0 if layer < 3 else 1, "rule": "fo" if layer >= 3 else "thin_all", "n_fo": 2 * (layer >= 3), "n_zo": 6,
               "fo_norm": None if layer < 3 else 1.0, "update_norm": 0.5} for r in (1, 2) for layer in (0, 1, 3)]
    s0, s1 = analysis.layer_summary(layers)
    assert (s0["layers"], s0["rounds"], s0["n_fo"], s0["fo_norm"], s0["rules"]) == (2, 2, 0, None, {"thin_all": 1.0})
    assert (s1["n_fo"], s1["fo_norm"], s1["update_norm"], s1["rules"]) == (2, 1.0, 0.5, {"fo": 1.0})


# --- the P1 plan ---------------------------------------------------------------------------------------------------

class Plan:
    model, seeds, rounds, tiny = "qwen", [42, 43], 160, False

    def common(self, fields):
        return {"device": "cpu"}


def test_p1_arms_resolve_to_the_configurations_of_the_brief():
    specs = p1.specs(Plan())
    # The ladder for both seeds (2 x 12), then the seven controls for the first seed only.
    assert len(specs) == 31 and [s["arm"] for s in specs[:5]] == ["A-b0", "C-b0", "A-b0.75", "Ax2-b0", "Ax3-b0"] and specs[12]["seed"] == 43
    assert len({s["name"] for s in specs}) == 31 and specs[0]["name"] == "qwen-s42-A-b0"
    resolved = {s["arm"]: Config(**s["fields"]).resolve() for s in specs[:12]}
    names = {arm: (cfg.aggregation_name, cfg.zo_mode, cfg.server_lr, cfg.beta) for arm, cfg in resolved.items()}
    assert names["A-b0"] == ("mean_all", "on", 1.0, 0.0) and names["A-b0.75"] == ("mean_all", "on", 1.0, 0.75)
    assert names["B-b0"] == ("fo_diluted", "off", 1.0, 0.0) and names["C-b0"] == ("fo_mean", "on", 1.0, 0.0) and names["D-b0.75"] == ("fo_mean", "off", 1.0, 0.75)
    assert names["Ax2-b0"] == ("mean_all", "on", 2.0, 0.0) and names["Ax3-b0"] == ("mean_all", "on", 3.0, 0.0)
    assert all(cfg.dataset == "ag_news" and cfg.partition == "dirichlet_05" and cfg.rounds == 160 and cfg.model == "qwen" for cfg in resolved.values())
    assert all((cfg.E, cfg.q, cfg.k, cfg.lr, cfg.batch_size, cfg.sampler) == (5, 2, 6, 1e-5, 16, "dep_round_v1") for cfg in resolved.values())  # I5
    assert resolved["A-b0"].probe_rounds == "1,20,40,80,160" and resolved["C-b0"].probe_rounds == "" and resolved["fedavg"].method == "fedavg"
    assert [s["arm"] for s in p1.specs(Plan(), blocks=[2])[:4]] == ["B-b0", "B-b0.75", "fedavg", "decomfl"]
    assert {s["timing_key"] for s in specs[:24]} == {"qwen/hofl", "qwen/hofl-zo-off", "qwen/fedavg", "qwen/decomfl"}
    short = Plan()
    short.rounds = 3
    assert p1.specs(short)[0]["fields"]["probe_rounds"] == "1,3" and p1.specs(short)[0]["name"].endswith("-r3")


def test_control_arms_run_for_the_first_seed_only_and_keep_everything_else_at_the_defaults():
    controls = p1.specs(Plan(), blocks=[4])
    assert [s["arm"] for s in controls] == ["Cx2-b0", "Ax2-b0.75", "fedavg-x2", "A-b0-q8", "C-b0-r480", "A-b0-r480", "Ax3-b0-r480"]
    assert {s["seed"] for s in controls} == {42} and controls[-1]["name"] == "qwen-s42-Ax3-b0-r480"
    resolved = {s["arm"]: Config(**s["fields"]).resolve() for s in controls}
    settings = {arm: (cfg.aggregation_name, cfg.server_lr, cfg.beta, cfg.q, cfg.rounds, cfg.evaluate_every) for arm, cfg in resolved.items()}
    assert settings["Cx2-b0"] == ("fo_mean", 2.0, 0.0, 2, 160, 10) and settings["Ax2-b0.75"] == ("mean_all", 2.0, 0.75, 2, 160, 10)
    assert settings["A-b0-q8"] == ("mean_all", 1.0, 0.0, 8, 160, 10) and resolved["fedavg-x2"].method == "fedavg" and resolved["fedavg-x2"].server_lr == 2.0
    assert settings["C-b0-r480"] == ("fo_mean", 1.0, 0.0, 2, 480, 20) and settings["Ax3-b0-r480"] == ("mean_all", 3.0, 0.0, 2, 480, 20)
    assert all((cfg.E, cfg.k, cfg.lr, cfg.batch_size, cfg.zo_mode) == (5, 6, 1e-5, 16, "on") for cfg in resolved.values())
    assert {s["timing_key"] for s in controls} == {"qwen/hofl", "qwen/fedavg", "qwen/hofl-q8"}
    short = Plan()
    short.rounds = 3
    tiny = {s["arm"]: s for s in p1.specs(short, blocks=[4])}
    assert tiny["A-b0-r480"]["fields"]["rounds"] == 9 and tiny["Cx2-b0"]["fields"]["rounds"] == 3 and tiny["A-b0-r480"]["name"] == "qwen-s42-A-b0-r480-r3"


def curve(*points):
    return [{"round": r, "accuracy_pct": a} for r, a in points]


def test_speed_measures():
    rising = curve((0, 50.0), (10, 70.0), (20, 80.0), (40, 84.0))
    assert analysis.accuracy_at(rising, 10) == 70.0 and analysis.accuracy_at(rising, 30) == pytest.approx(82.0) and analysis.accuracy_at(rising, 50) is None
    assert analysis.rounds_to_reach(rising, 60.0) == pytest.approx(5.0) and analysis.rounds_to_reach(rising, 80.0) == pytest.approx(20.0)
    assert analysis.rounds_to_reach(rising, 50.0) == 0.0 and analysis.rounds_to_reach(rising, 90.0) is None
    # Trapezoids: (50+70)/2*10 + (70+80)/2*10 = 1350 over 20 rounds.
    assert analysis.mean_accuracy(rising, 20) == pytest.approx(67.5) and analysis.mean_accuracy(rising, 60) is None
    assert analysis.mean_accuracy(rising, 30) == pytest.approx((1350 + 810) / 30)


def test_step_size_control():
    first = {"A-b0": {42: 86.0}, "C-b0": {42: 87.8}, "Ax2-b0": {42: 87.7}, "Ax3-b0": {42: 88.6}}
    result = analysis.step_size_control(first)
    assert result["explained"] is True and result["g_c"] == pytest.approx(1.8) and result["g_x3"] == pytest.approx(2.6) and result["matched_x2"] is None
    matched = analysis.step_size_control({**first, "Cx2-b0": {42: 88.9}})
    assert matched["matched_x2"] == pytest.approx(1.2) and matched["per_seed"][0]["matched_x2"] == pytest.approx(1.2)
    assert analysis.step_size_control({"A-b0": {42: 86.0}, "C-b0": {42: 87.8}, "Ax2-b0": {42: 86.2}})["explained"] is False
    assert analysis.step_size_control({"A-b0": {42: 86.0}, "C-b0": {42: 87.8}})["explained"] is None          # no control run
    assert analysis.step_size_control({"A-b0": {42: 86.0}, "C-b0": {42: 85.0}, "Ax2-b0": {42: 87.0}})["explained"] is None   # nothing gained
    assert analysis.step_size_control({"A-b0": {42: 86.0}})["seeds"] == []


def test_sampling_speedup_with_and_without_zeroth_order_updates():
    curves = {"A-b0": {42: curve((0, 55.0), (40, 71.0), (160, 86.0))}, "A-b0.75": {42: curve((0, 55.0), (40, 76.0), (160, 86.0))},
              "B-b0": {42: curve((0, 55.0), (40, 70.8), (160, 85.8))}, "B-b0.75": {42: curve((0, 55.0), (40, 76.3), (160, 85.9))}}
    result = analysis.sampling_speedup(curves, 40)
    assert result["gain_zo_on"] == pytest.approx(5.0) and result["gain_zo_off"] == pytest.approx(5.5) and result["max_gap_on_off"] == pytest.approx(0.3)
    assert analysis.sampling_speedup({"A-b0": curves["A-b0"]}, 40)["per_seed"] == []
