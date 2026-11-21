"""The dose-response experiment: its inputs, its arms and the H2 computation."""
import json
from pathlib import Path

import pytest

from ho_fl import dose, harness
from ho_fl.config import Config
from ho_fl.faults import Faults, role_members
from ho_fl.scheduling import ScheduleFileScheduler

CUTS = [8] * 10 + [16] * 10 + [24] * 10


def test_inputs_share_one_cohort_sequence_and_differ_only_in_the_cut_overrides(tmp_path):
    inputs = dose.write_inputs(tmp_path, CUTS, 6, 42, 20)
    device = role_members("preferred_t1", CUTS, 6, 42)[0]
    assert inputs["device"] == device and inputs["cuts"] == [8, 16] and len(inputs["cohorts"]) == 20
    assert all(cohort[0] == device and len(set(cohort)) == 6 for cohort in inputs["cohorts"])
    for m in range(6):
        scheduler = ScheduleFileScheduler(inputs["schedules"][m], CUTS, 6)
        for round_id, cohort in enumerate(inputs["cohorts"], 1):
            plan = scheduler.sample(round_id)
            assert plan.clients == cohort
            cuts = [plan.cut(cid) for cid in cohort]
            # The unreliable device keeps its own cut; m of the five others are first-order on S1, the rest are not.
            assert cuts[0] == 8 and cuts[1:] == [8] * m + [16] * (5 - m)
    assert dose.write_inputs(tmp_path / "again", CUTS, 6, 42, 20)["cohorts"] == inputs["cohorts"]
    assert dose.write_inputs(tmp_path / "other", CUTS, 6, 43, 20)["cohorts"] != inputs["cohorts"]
    assert inputs["faults"]["none"] is None
    cfg = Config(model="qwen", dataset="ag_news", fault_spec=inputs["faults"]["flip20"]).resolve()
    (entry,) = Faults(cfg, CUTS).entries
    assert entry == {"clients": [device], "first": 1, "last": 20, "kind": "sign_flip", "params": {"scale": 20.0, "layers": "S1"}}


def test_arms_resolve_to_the_intended_configurations(tmp_path):
    inputs = dose.write_inputs(tmp_path, CUTS, 6, 42, 20)
    specs = dose.specs("qwen", 42, inputs, "state.safetensors", {"dataset": "ag_news", "partition": "dirichlet_05"})
    assert len(specs) == 6 * 6 * 3 and len({spec["name"] for spec in specs}) == len(specs)
    assert [spec["m"] for spec in specs[::18]] == [2, 1, 3, 0, 4, 5] and specs[0]["name"] == "qwen-s42-dose-m2-mean_all-none"
    resolved = {(spec["m"], spec["aggregation"], spec["fault"]): Config(**spec["fields"]).resolve() for spec in specs}
    assert all((cfg.sampler, cfg.rounds, cfg.evaluate_every, cfg.on_nonfinite, cfg.init_state, cfg.method) == ("schedule_file", 20, 20, "stop", "state.safetensors", "hofl")
               for cfg in resolved.values())
    assert all((cfg.E, cfg.q, cfg.k, cfg.lr, cfg.batch_size, cfg.server_lr) == (5, 2, 6, 1e-5, 16, 1.0) for cfg in resolved.values())
    names = {aggregation: resolved[(2, aggregation, "none")] for aggregation in dose.AGGREGATIONS}
    assert [names[a].aggregation_name for a in ("mean_all", "fo_mean", "robust_all", "fo_robust")] == ["mean_all", "fo_mean", "robust_all", "fo_robust"]
    assert all(names[a].zo_mode == "on" for a in ("mean_all", "fo_mean", "robust_all", "fo_robust"))
    hold, dilute = names["quorum_hold"], names["quorum_dilute"]
    assert (hold.agg_electorate, hold.agg_stat, hold.fo_quorum, hold.thin_policy, hold.zo_mode) == ("fo", "geomed", 3, "hold", "off")
    assert (dilute.thin_policy, dilute.zo_mode, dilute.fo_quorum) == ("dilute", "off", 3)
    assert resolved[(2, "mean_all", "none")].fault_spec == "" and resolved[(2, "mean_all", "flip1")].fault_spec == inputs["faults"]["flip1"]
    assert resolved[(0, "fo_robust", "flip20")].schedule_file == inputs["schedules"][0]
    assert len(dose.specs("qwen", 42, inputs, "s", {}, doses=(2, 1), faults=("none", "flip20"))) == 2 * 6 * 2


def fake_arm(root, m, aggregation, fault, loss, influence=None, status="COMPLETED"):
    path = root / f"qwen-s42-dose-{dose.arm_name(m, aggregation, fault)}"
    harness.write_json(path / "status.json", {"status": status})
    harness.write_json(path / "summary.json", {"accuracy": 0.8, "task_loss": loss, **({"diverged_round": 7} if status == "DIVERGED" else {})})
    harness.write_json(path / "run_meta.json", {"wall_s": 90.0})
    if influence is not None:
        (path / "influence.jsonl").write_text("".join(json.dumps({"round": r, "layer": 8, "segment": segment, "relative": value}) + "\n"
                                                      for r in (1, 2) for segment, value in ((1, influence), (2, 99.0))))


def test_outcomes_tables_and_the_crossover_rule(tmp_path):
    # robust_all: bounded but slow everywhere. fo_robust: bad below the quorum, better from m = 2 on.
    for m in range(4):
        for aggregation, clean, hit in (("robust_all", 0.50, 0.52), ("fo_robust", 0.45, 0.90 if m < 2 else 0.46)):
            fake_arm(tmp_path, m, aggregation, "none", clean)
            fake_arm(tmp_path, m, aggregation, "flip20", hit, influence=0.1 if aggregation == "robust_all" else (3.0 if m < 2 else 0.2))
    arms = dose.collect(tmp_path, "qwen", 42)
    assert arms[(1, "quorum_robust", "flip20")]["same_as"] == "robust_all" and arms[(2, "quorum_robust", "flip20")]["same_as"] == "fo_robust"
    assert arms[(0, "fo_robust", "flip20")]["influence_s1"] == pytest.approx(3.0)       # segment 1 only
    rows = {row["aggregation"]: row for row in dose.table_rows(arms, "flip20", range(4))}
    assert rows["fo_robust"]["cells"][0]["damage"] == pytest.approx(0.45) and rows["fo_robust"]["cells"][3]["damage"] == pytest.approx(0.01)
    assert rows["quorum_robust"]["cells"][1]["loss"] == 0.52 and rows["quorum_robust"]["cells"][2]["loss"] == 0.46
    assert rows["mean_all"]["cells"] == [None] * 4
    verdict = dose.decide_h2(arms, range(4))
    assert verdict["flip20"]["verdict"] == "SUPPORTED" and verdict["flip20"]["fo_robust_better_at"] == [2, 3]
    assert verdict["flip1"]["verdict"] == "CANNOT_JUDGE"

    fake_arm(tmp_path / "late", 0, "robust_all", "flip20", 0.52)
    fake_arm(tmp_path / "late", 0, "fo_robust", "flip20", 0.40)                    # better although there is no majority
    for m in (2, 3):
        fake_arm(tmp_path / "late", m, "robust_all", "flip20", 0.52)
        fake_arm(tmp_path / "late", m, "fo_robust", "flip20", 0.46, status="DIVERGED" if m == 3 else "COMPLETED")
    late = dose.collect(tmp_path / "late", "qwen", 42)
    assert late[(3, "fo_robust", "flip20")]["loss"] is None and late[(3, "fo_robust", "flip20")]["diverged_round"] == 7
    assert dose.decide_h2(late, range(4))["flip20"] == {"doses": [0, 2, 3], "fo_robust_better_at": [0, 2], "verdict": "NOT_SUPPORTED",
                                                        "reason": "first-order-only robust averaging is better at m = [0, 2]; the hypothesis expects exactly m >= 2"}
    assert dose.read_arm(tmp_path / "missing") is None
