"""The checks of the zeroth-order part: uniform cuts, the attribution probe, the counterfactual dose probe, the
numerical check of the estimator, and the summaries built from them."""
import json
from pathlib import Path

import pytest

from ho_fl import analysis, dose, harness
from ho_fl.config import Config
from ho_fl.train import run
from ho_fl.zo_check import run as zo_check
from test_tiny_training import logs, read_jsonl


def rows(cfg, name):
    return read_jsonl(Path(cfg.output) / name)


# --- configuration ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("fields", [
    {"uniform_cut": 24},                                   # not a tier cut of OPT
    {"uniform_cut": 9, "method": "fedavg"},                # only HO-FL has cuts to make uniform
    {"dose_probe": True},                                  # needs the cohorts of a dose schedule
    {"attribution_every": -1}, {"probe_val_examples": 0},
    {"dataset": "squad", "attribution_every": 4},
])
def test_invalid_settings_are_rejected(fields):
    with pytest.raises(ValueError):
        Config(**fields).resolve()


def test_new_settings_are_named_in_the_method_label():
    assert Config(uniform_cut=9).resolve().method_label == "HO-FL (β=0.75) [uniform cut 9]"
    assert Config(attribution_every=4, eval_initial=False).resolve().method_label == "HO-FL (β=0.75)"


# --- uniform cut and the optional first evaluation ------------------------------------------------------------------------

def test_uniform_cut_keeps_the_cohorts_and_changes_only_the_cuts(tiny_config):
    plain = tiny_config("sst2", rounds=2, beta=0.0, k=3, E=1)
    uniform = tiny_config("sst2", rounds=2, beta=0.0, k=3, E=1, uniform_cut=9)
    run(plain), run(uniform)
    assert [row["client"] for row in rows(plain, "ledger.jsonl")] == [row["client"] for row in rows(uniform, "ledger.jsonl")]
    assert {row["cut"] for row in rows(uniform, "ledger.jsonl")} == {9} and len({row["cut"] for row in rows(plain, "ledger.jsonl")}) > 1
    assert {row["tier"] for row in rows(uniform, "ledger.jsonl")} == {row["tier"] for row in rows(plain, "ledger.jsonl")}
    config = json.loads((Path(uniform.output) / "config.json").read_text())
    assert config["actual_cuts"] == [9] * 30 and sorted(set(config["resource_cuts"])) == [3, 6, 9]


def test_skipping_the_first_evaluation_does_not_change_training(tiny_config):
    full = tiny_config("sst2", rounds=2, evaluate_every=2, k=3, E=1)
    lean = tiny_config("sst2", rounds=2, evaluate_every=2, k=3, E=1, eval_initial=False)
    first, second = run(full), run(lean)
    assert first["final_parameter_sha256"] == second["final_parameter_sha256"] and first["accuracy"] == second["accuracy"]
    assert [row["round"] for row in rows(full, "curves.jsonl")] == [0, 2] and [row["round"] for row in rows(lean, "curves.jsonl")] == [2]


def test_timing_without_an_evaluation_of_round_zero():
    record = harness.timing({"round_times": {"1": 15.0, "2": 18.0, "3": 21.0}, "eval_times": {"3": 23.0}})
    assert (record["round_s"], record["eval_s"], record["startup_s"], record["rounds_measured"]) == (3.0, 2.0, 12.0, 2)
    assert harness.timing({"round_times": {"1": 15.0}, "eval_times": {}}) is None
    # ... and the estimate of such a run counts its evaluations without round 0.
    assert harness.evaluation_count(160, 160, initial=False) == 1 and harness.evaluation_count(160, 160) == 2
    assert harness.estimate_seconds(record, 160, 160, initial=False) == 12.0 + 2.0 + 3.0 * 160


# --- attribution --------------------------------------------------------------------------------------------------------

def test_attribution_splits_the_predicted_decrease_exactly_and_is_read_only(tiny_config):
    plain = tiny_config("sst2", rounds=2, evaluate_every=1, k=3, E=2)
    probed = tiny_config("sst2", rounds=2, evaluate_every=1, k=3, E=2, attribution_every=1, probe_val_examples=96)
    assert run(plain) == run(probed) and logs(plain) == logs(probed)
    attributed = rows(probed, "attribution.jsonl")
    assert len(attributed) == 2 * 4 and {row["round"] for row in attributed} == {1, 2}
    for row in attributed:
        # Exact up to rounding: the parts are summed in float64, the applied update was accumulated in float32.
        assert row["decomposed"] and row["pred_fo"] + row["pred_zo"] == pytest.approx(row["pred_total"], rel=1e-6, abs=1e-12)
    assert all(row["n_fo"] == 0 for row in attributed if row["segment"] == 0)       # nobody trains the lowest segment first-order
    assert len({row["val_loss"] for row in attributed if row["round"] == 1}) == 1
    summary = analysis.attribution_summary(attributed)
    assert [entry["segment"] for entry in summary["segments"]] == [0, 1, 2, 3] and summary["segments"][0]["pred_fo"] == 0
    json.dumps(attributed, allow_nan=False)

    frozen = tiny_config("sst2", rounds=2, evaluate_every=2, k=3, E=2, attribution_every=1, probe_val_examples=96, zo_mode="off",
                         agg_electorate="fo", agg_weight="cohort")
    run(frozen)
    assert all(row["pred_zo"] == 0 and row["n_zo"] == 0 for row in rows(frozen, "attribution.jsonl"))
    robust = tiny_config("sst2", rounds=1, k=3, E=1, attribution_every=1, probe_val_examples=32, agg_stat="geomed")
    run(robust)
    assert all(not row["decomposed"] and row["pred_fo"] is None and row["pred_total"] is not None for row in rows(robust, "attribution.jsonl"))


# --- counterfactual dose probe ------------------------------------------------------------------------------------------

def dose_schedule(tiny_config, tmp_path, rounds=2, m_traj=2):
    probe = tiny_config("sst2", rounds=1, k=6, E=1)
    run(probe)
    config = json.loads((Path(probe.output) / "config.json").read_text())
    inputs = dose.write_inputs(tmp_path / "inputs", config["resource_cuts"], 6, 42, rounds)
    return inputs, {"sampler": "schedule_file", "schedule_file": inputs["schedules"][m_traj], "rounds": rounds, "k": 6, "E": 2}


def test_dose_probe_is_read_only_and_forms_every_cell_from_the_same_round(tiny_config, tmp_path):
    inputs, schedule = dose_schedule(tiny_config, tmp_path)
    plain = tiny_config("sst2", **schedule)
    probed = tiny_config("sst2", **schedule, dose_probe=True, attribution_every=1, probe_val_examples=64)
    assert run(plain) == run(probed) and logs(plain) == logs(probed)
    probe_rows = rows(probed, "dose_probe.jsonl")
    references = [row for row in probe_rows if row["kind"] == "reference"]
    cells = {(row["round"], row["m"], row["rule"], row["fault"]): row for row in probe_rows if row["kind"] == "candidate"}
    assert len(references) == 2 and len(cells) == 2 * 6 * len(dose.COUNTERFACTUAL_RULES) * len(dose.COUNTERFACTUAL_FAULTS)
    assert len(references[0]["pred_fo_reports"]) == 6 and len(references[0]["pred_zo_reports"]) == 5
    for (r, m, rule, fault), row in cells.items():
        if fault == "none":
            assert row["influence"] is None
            continue
        if rule in ("mean_all", "fo_diluted"):
            assert row["influence"] == pytest.approx(1.0, rel=1e-6)            # the faulty report enters with weight 1/K
        if rule == "fo_mean":
            assert row["influence"] == pytest.approx(6 / (m + 1), rel=1e-6)    # ... or 1/(m+1) among the first-order reports
        if rule == "quorum_hold" and m + 1 < dose.QUORUM:
            assert row["norm"] == 0 and row["pred"] == 0 and row["influence"] == 0
    for (r, m, rule, fault), row in cells.items():
        twin = {"quorum_robust": "robust_all" if m + 1 < dose.QUORUM else "fo_robust", "quorum_dilute": "fo_diluted" if m + 1 < dose.QUORUM else "fo_robust",
                "quorum_hold": None if m + 1 < dose.QUORUM else "fo_robust"}.get(rule)
        if twin:
            assert row["pred"] == pytest.approx(cells[(r, m, twin, fault)]["pred"], rel=1e-9, abs=1e-15), (rule, m, fault)
    # The cell of the trajectory's own dose, without fault, under its own rule is the applied update itself: the attribution
    # probe measures the same first-order decrease on segment S1.
    attributed = {row["round"]: row for row in rows(probed, "attribution.jsonl") if row["segment"] == 1}
    for r in (1, 2):
        assert cells[(r, 2, "mean_all", "none")]["pred"] == pytest.approx(attributed[r]["pred_total"], rel=1e-4)
    summary = dose.cf_summary(probe_rows)
    assert summary["rounds"] == 2 and summary["table"][("none", "fo_mean", 5)]["rounds"] == 2
    assert summary["table"][("flip20", "mean_all", 0)]["influence"] == pytest.approx(1.0, rel=1e-6)
    verdict = dose.decide_h2_cf(summary)
    assert set(verdict) == {"flip1", "flip20"} and verdict["flip20"]["verdict"] in ("SUPPORTED", "NOT_SUPPORTED")
    assert set(verdict["flip20"]["fallback_ranking_below_quorum"]) == {0, 1}


def test_counterfactual_summary_and_rule_on_known_rows():
    references = [{"kind": "reference", "round": r, "pred_fo_report": 1.0, "pred_zo_report": z} for r, z in ((1, 0.02), (2, -0.02))]
    candidates = []
    for r in (1, 2):
        for m in range(6):
            for fault in ("none", "flip20"):
                good = 0.6 if m + 1 >= dose.QUORUM else -1.0
                values = {"fo_robust": good, "robust_all": 0.2, "quorum_robust": 0.2 if m + 1 < dose.QUORUM else good,
                          "quorum_hold": 0.0 if m + 1 < dose.QUORUM else good, "quorum_dilute": 0.3 if m + 1 < dose.QUORUM else good}
                for rule, value in values.items():
                    candidates.append({"kind": "candidate", "round": r, "m": m, "rule": rule, "fault": fault, "pred": value,
                                       "cos": 0.1, "influence": None if fault == "none" else 0.5})
    summary = dose.cf_summary(references + candidates)
    assert summary["unit"] == 1.0 and summary["table"][("flip20", "fo_robust", 3)]["progress"] == pytest.approx(0.6)
    assert summary["zo_vs_fo_on_s1"]["ratio"] == pytest.approx(0.0)
    verdict = dose.decide_h2_cf(summary)["flip20"]
    assert verdict["verdict"] == "SUPPORTED" and verdict["fo_robust_better_at"] == [2, 3, 4, 5]
    assert verdict["fallback_ranking_below_quorum"] == {0: ["quorum_dilute", "quorum_robust", "quorum_hold"], 1: ["quorum_dilute", "quorum_robust", "quorum_hold"]}
    spec = dose.cf_spec("qwen", 42, {"schedules": {2: "s2.jsonl"}, "rounds": 20}, "state.safetensors", {"dataset": "ag_news", "partition": "dirichlet_05"})
    cfg = Config(**spec["fields"]).resolve()
    assert (cfg.sampler, cfg.schedule_file, cfg.rounds, cfg.dose_probe, cfg.eval_initial, cfg.aggregation_name, cfg.zo_mode) == \
        ("schedule_file", "s2.jsonl", 20, True, False, "mean_all", "on")
    assert spec["name"] == "qwen-s42-dose-cf-m2"


# --- numerical check of the zeroth-order estimator ----------------------------------------------------------------------

def test_the_estimator_check_finds_the_finite_difference_accurate_in_float32(tiny_config, tmp_path):
    cfg = tiny_config("sst2")
    result = zo_check(cfg, tmp_path / "check", q=8, mus=(1e-3,), batches=1, precisions=("bfloat16", "float32"))
    rows_ = result["rows"]
    assert len(rows_) == 2 * 3 and {row["cut"] for row in rows_} == {3, 6, 9} and all(row["directions"] == 8 for row in rows_)
    for row in rows_:
        if row["precision"] == "float32":
            assert row["correlation"] > 0.99 and row["slope"] == pytest.approx(1.0, abs=0.05), row
        assert [entry["q"] for entry in row["information"]] == [1, 2, 4, 8]
    assert (tmp_path / "check" / "zo_check.md").exists()
    json.dumps(result, allow_nan=False)
