"""The E0 pipeline (``experiments/sat/e0/e0.py``) without running anything: which stage may start, how results are
read from run directories, and the feedback report with its checklist, findings and answers. The stages themselves
are exercised by the ``--tiny`` dry run of the pipeline."""
import importlib.util
import json
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("sat_e0", REPO / "experiments" / "sat" / "e0" / "e0.py")
e0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e0)
digest = e0.digest


# --- a session as ``collect`` would return it -------------------------------------------------------------------------------

def check(item, passed, worst=0.0, bitwise=True):
    return {"status": "PASSED" if passed else "FAILED", "model": "vit", "device": "cuda", "gpu": "G4", "mu": 0.01, "prompt_tokens": None,
            "rows": [{"item": item, "check": "a check", "passed": passed, "detail": "detail", "worst": worst, "bitwise": bitwise}],
            "estimator": [{"cut": cut, "mu": mu, "configured": mu == 0.01, "directions": 64, "dimension": 1000, "finite": True,
                           "correlation": {0.001: 0.55, 0.01: 0.95, 0.1: 0.4}[mu], "slope": 1.0, "sign_agreement": 0.9}
                          for cut in (3, 12) for mu in (0.001, 0.01, 0.1)]}


def run(name, task, method, accuracies, group="short", round_s=1.5, **changes):
    hours = float(len(accuracies) - 1)
    rounds = int(6 * hours)
    record = {"name": name, "group": group, "task": task, "method": method, "hours": hours, "status": "COMPLETED", "note": None, "rounds": rounds,
              "depth": {"t1": 28, "t2": 12}[task], "diverged_round": None, "wall_s": 20 + rounds * round_s + len(accuracies) * 4.0,
              "verdict": {"status": "COMPLETED", "loss_first_third": 2.3, "loss_last_third": 1.9, "loss_decreased": True, "passed": True, "completed": True},
              "curve": [{"round": 6 * hour, "sim_hours": float(hour), "accuracy": value, "macro_f1": value, "loss": 1.0} for hour, value in enumerate(accuracies)],
              "participants": {"B": 6.0, "H": 6.0, "Z": 4.0}, "invited": {"B": 6.0, "H": 6.0, "Z": 6.0},
              "skipped": {"memory": 0, "eclipse": 24, "energy": 0, "unreachable": 0}, "late": 0, "moved_up": 0,
              "zo": {"sent": 120, "used": 120, "dropped": 0}, "replica_groups_max": 1, "replica_difference_max": 0.0, "battery_min": {"H": 0.85, "Z": 0.87},
              "seconds": {"round": round_s, "evaluation": 4.0, "startup": 20.0, "parts": {"train": 0.8, "deliver": 0.02, "aggregate": 0.15, "log": 0.03}},
              "participation_seconds": [{"kind": "B", "cut": 0, "zo": False, "count": 72, "mean_s": 0.05},
                                        {"kind": "H", "cut": 8 if task == "t1" else 3, "zo": True, "count": 72, "mean_s": 0.08},
                                        {"kind": "Z", "cut": 28 if task == "t1" else 12, "zo": True, "count": 48, "mean_s": 0.10}],
              "participations": {"B": 72, "H": 72, "Z": 48}, "bytes_per_day": {"rf": 2e9, "laser": 1e9, "ground": 0},
              "energy_wh_per_satellite_day": {"compute": {"B": 12.0, "H": 0.6, "Z": 0.4}, "transmit": {"B": 0.02, "H": 0.01, "Z": 0.01}}}
    record.update(changes)
    return record


def day(start, end):
    """25 hourly accuracies rising from ``start`` to ``end``."""
    return [start + (end - start) * hour / 24 for hour in range(25)]


def session(**changes):
    stages = {name: {"status": "ok", "wall_s": 60.0, "calls": 1, "blocked_by": None} for name in e0.STAGES}
    data = {"generated_at": "2026-11-25T10:00:00+00:00", "tiny": False, "stages": stages,
            "env": {"gpu_name": "G4", "torch": "2.7.1+cu128", "git": {"head": "abcdef123456", "dirty": False}, "land_mask_installed": True},
            "session": {"started_at": 0.0, "elapsed_hours": 1.3, "budget_hours": 2.5,
                        "steps": [{"name": "environment", "minutes": 2.0, "status": "ok", "gpu": False}, {"name": "dry_run", "minutes": 4.0, "status": "ok", "gpu": False}]},
            "baseline": {"status": "PASSED", "returncode": 0, "golden_mode": "bitwise", "replay": {"verdict": "PASS", "max_abs_error": 3.6e-12, "comparisons": 672},
                         "configs": {name: {"verdict": "PASS_BITWISE", "max_abs_state_diff": 0.0, "first_difference": None} for name in ("hofl", "fedavg", "decomfl")}},
            "tests": {"returncode": 0, "wall_s": 200.0, "tests": 400, "failures": 0, "errors": 0, "skipped": 0, "failed": []},
            "trace": {"returncode": 0, "land_source": "global_land_mask", "trace_sha256": "8a0e" * 16, "path": "p", "matches_report": True},
            "partitions": [{"name": f"{task}-{variant}-s{seed}", "task": task, "variant": variant, "seed": seed, "returncode": 0, "examples_per_satellite": 135,
                            "unused": 0, "partition_sha256": "47" * 32, "data_sha256": "3f", "matches_report": True}
                           for task in ("t1", "t2") for variant in ("base", "zonly") for seed in (42, 43, 44)],
            "selfcheck": {"t2": {**check(2, True), "rows": check(2, True)["rows"] + check(4, True)["rows"] + check(5, True)["rows"]},
                          "t1": check(4, True), "upstream": {**check(3, True), "estimator": []}},
            "memory": {task: {"model": model, "batch_shape": [16, 192], "device": "G4", "memory_states_gb": {"backbone": 9.0, "hybrid_idle": 7.0, "hybrid_imaging": 5.5, "forward_only": 4.0},
                              "rows": [{"cut": 0, "mode": "inference", "peak_allocated_gb": 3.5, "first_order_bytes": 0}]
                              + [{"cut": cut, "mode": "hybrid", "peak_allocated_gb": peak, "first_order_bytes": 1_000_000} for cut, peak in ((0, 9.0), (8, 7.0), (16, 5.5), (28, 4.0))]
                              + [{"cut": 8, "mode": "frozen", "peak_allocated_gb": 6.5, "first_order_bytes": 1_000_000}]}
                       for task, model in (("t1", "qwen"), ("t2", "vit"))},
            "runs": [run("t2-m7", "t2", "m7", [0.10, 0.50, 0.70]), run("t2-m0", "t2", "m0", [0.10, 0.60, 0.80]), run("t2-m2", "t2", "m2", [0.10, 0.50, 0.75]),
                     run("t2-m4", "t2", "m4", [0.10, 0.12, 0.15]), run("t1-m7", "t1", "m7", [0.30, 0.40, 0.50], round_s=6.0),
                     run("t2-m7-day", "t2", "m7", day(0.10, 0.93), group="day"), run("t2-m0-day", "t2", "m0", day(0.10, 0.95), group="day")],
            "repeats": [{"task": "t2", "method": method, "rounds_compared": 12, "evaluations_compared": 3, "identical": True,
                         "max_loss_difference": 0.0, "max_accuracy_difference": 0.0} for method in ("m7", "m0")],
            "t0": [{"name": f"qwen-zo{mode}-s{seed}", "seed": seed, "mode": mode, "status": "COMPLETED", "note": None, "accuracy": accuracy, "task_loss": 0.4,
                    "wall_s": 600.0, "timing": None} for seed, mode, accuracy in ((43, "on", 0.8600), (43, "off", 0.8580), (44, "on", 0.8610), (44, "off", 0.8601))],
            "t0_reference": {"on": 85.99, "off": 85.95}, "failures": [], "skipped_for_budget": []}
    data.update(changes)
    return data


def levels(data):
    return {(finding["level"], finding["topic"]) for finding in digest.findings(data)}


# --- checklist and gate -----------------------------------------------------------------------------------------------------

def test_a_clean_session_passes_the_gate():
    data = session()
    rows = digest.checklist(data)
    assert [row[0] for row in rows] == ["1", "1", "2", "3", "4", "5", "6", "7"] and {row[2] for row in rows} == {"PASS"}
    assert digest.gate(rows) == ("PASS", [])
    assert not {level for level, _ in levels(data)} & {"block", "fix", "decide"}
    text = digest.render(data)
    assert "**关卡 G0：通过。**" in text and "必须先修的 0 项，建议调整的 0 项，需要你决定的 0 项" in text
    assert "代码 abcdef123456 · GPU G4 · torch 2.7.1+cu128 · 会话已用 1.30 h（预算 2.5 h）" in text
    assert "关卡 G0 的七项检查都通过了" in text and "名称对照" in text and "## 5 失败与异常（日志末尾）\n\n无。" in text
    # The seven questions are answered in order, each from the numbers of the session.
    answers = [answer for _, answer in digest.answers(data, rows)]
    assert answers[0].startswith("没有。") and "CPU 测试 400 项，0 项没有通过" in answers[0]
    assert answers[1].startswith("对。") and "5 / 5 项检查通过" in answers[1] and "12 / 12 份划分" in answers[1]
    assert "强星 9.0 GB，GPU 卫星空闲 7.0 GB、成像 5.5 GB，推理芯片卫星 4.0 GB" in answers[2]
    assert "EuroSAT：每轮 1.5 s，一次 24 小时的运行实测 5.6 分钟（估计 3–6）" in answers[3] and "HumAID：每轮 6.0 s，一次 24 小时的运行折算约 16.4 分钟" in answers[3]
    assert "EuroSAT 上完整方法 2 个仿真小时内验证准确率 10.0% → 70.0%，训练损失下降" in answers[4] and "EuroSAT 0.95–0.95" in answers[4] and "逐位相同" in answers[4]
    assert "理想中心化第 24 小时的验证准确率是 95.0%，完整方法 93.0%" in answers[5] and "没有到 98%" in answers[5]
    assert "3 个种子上，保留零阶比关掉高 +0.11 个点" in answers[6]


def test_checklist_reports_pass_fail_and_not_run():
    def verdicts(data):
        return [(item, verdict) for item, _, verdict, _ in digest.checklist(data)]

    failed = session(selfcheck={"t2": check(2, True), "t1": check(2, False), "upstream": None}, tests=None)
    assert verdicts(failed) == [("1", "NOT RUN"), ("1", "PASS"), ("2", "FAIL"), ("3", "NOT RUN"), ("4", "NOT RUN"), ("5", "NOT RUN"), ("6", "PASS"), ("7", "PASS")]
    assert "1 / 2 项检查通过（t1、t2）" in digest.checklist(failed)[2][3] and digest.gate(digest.checklist(failed)) == ("FAIL", ["2"])
    nothing = session(tests=None, trace=None, baseline=None, selfcheck={"t2": None, "t1": None, "upstream": None}, runs=[])
    assert {verdict for _, verdict in verdicts(nothing)} == {"NOT RUN"} and digest.gate(digest.checklist(nothing))[0] == "OPEN"
    # A run that crashed fails item 7; a run that was never started leaves it open.
    crashed = session(runs=[run("t2-m7", "t2", "m7", [0.1], status="FAILED", rounds=0)])
    assert verdicts(crashed)[-1] == ("7", "FAIL") and "没有跑完（FAILED）" in digest.checklist(crashed)[-1][3]
    assert verdicts(session(runs=[run("t2-m7", "t2", "m7", [0.1], status="NOT RUN", rounds=0)]))[-1] == ("7", "NOT RUN")
    rising = session(runs=[run("t2-m7", "t2", "m7", [0.1, 0.1, 0.1], verdict={"status": "COMPLETED", "loss_first_third": 2.0, "loss_last_third": 2.1,
                                                                             "loss_decreased": False, "passed": False, "completed": True})])
    assert verdicts(rising)[-1] == ("7", "FAIL") and ("block", "运行") in levels(rising)
    # The regression of the baseline is the second half of item 1.
    broken = session(baseline={**session()["baseline"], "status": "FAILED",
                               "configs": {"hofl": {"verdict": "FAIL", "max_abs_state_diff": 0.01, "first_difference": "round 1: task_loss"}}})
    assert verdicts(broken)[1] == ("1", "FAIL") and ("block", "基线") in levels(broken)
    assert "`ho_fl/algorithms.py` 的 `hybrid_step`" in digest.render(broken) and "【必须先修】" in digest.render(broken)
    assert verdicts(session(baseline={"status": "INCOMPLETE", "returncode": 1}))[1] == ("1", "NOT RUN")


# --- findings ---------------------------------------------------------------------------------------------------------------

def test_findings_name_what_to_change():
    tests = {"returncode": 1, "wall_s": 9.0, "tests": 400, "failures": 1, "errors": 0, "failed": [{"id": "tests.sat.test_x::test_y", "message": "AssertionError: boom"}]}
    assert ("block", "测试") in levels(session(tests=tests)) and "`tests.sat.test_x::test_y`（AssertionError: boom）" in digest.render(session(tests=tests))
    # Hashes that differ from the ones recorded on the development machine are a fix, not a failure.
    partitions = session()["partitions"]
    partitions[0] = {**partitions[0], "matches_report": False}
    shifted = session(partitions=partitions, trace={**session()["trace"], "matches_report": False})
    assert {("fix", "数据"), ("fix", "轨迹")} <= levels(shifted) and digest.gate(digest.checklist(shifted)) == ("PASS", [])
    text = digest.render(shifted)
    assert "| t1-base-s42 | 135 | 0 | `" in text and "11 份划分的哈希与本机记录相同" in text and "expected.json" in text
    # A failed check points at the module to look at.
    wrong = session(selfcheck={"t2": check(4, False, worst=0.5, bitwise=False), "t1": None, "upstream": None})
    assert ("block", "正确性") in levels(wrong) and "`ho_fl/sat/aggregate.py`" in digest.render(wrong)
    loose = session(selfcheck={"t2": check(4, True, worst=3e-9, bitwise=False), "t1": check(2, True), "upstream": check(3, True)})
    assert ("note", "正确性") in levels(loose) and "最大差 3e-09" in digest.render(loose)
    # The estimator at the configured radius: good above 0.9, the research lead's call below 0.6.
    poor = check(2, True)
    poor["estimator"] = [{**row, "correlation": {0.001: 0.2, 0.01: 0.3, 0.1: 0.8}[row["mu"]]} for row in poor["estimator"]]
    found = digest.findings(session(selfcheck={"t2": poor, "t1": None, "upstream": None}))
    decision = next(finding for finding in found if finding["topic"] == "零阶估计")
    assert decision["level"] == "decide" and "相关系数在各分界上为 0.30" in decision["text"] and "扰动步长取 0.1 时最低相关系数是 0.80" in decision["text"]
    assert ("good", "零阶估计") in levels(session())
    # Unexpected events of a run without faults.
    odd = session(runs=[run("t2-m7", "t2", "m7", [0.1, 0.5, 0.7], late=3, replica_groups_max=2, replica_difference_max=0.1)])
    assert ("fix", "运行") in levels(odd) and "有 3 次一阶更新迟到，副本分成了 2 组" in digest.render(odd)
    assert ("note", "运行") in levels(session(runs=[run("t2-m7", "t2", "m7", [0.1, 0.5, 0.7], battery_min={"H": 0.7, "Z": 0.9})]))
    # Memory that does not separate the two states of a hybrid satellite.
    flat = session()["memory"]
    flat["t1"]["memory_states_gb"] = {"backbone": 9.0, "hybrid_idle": 7.0, "hybrid_imaging": 6.9, "forward_only": 4.0}
    assert ("decide", "显存") in levels(session(memory=flat))


def test_cost_saturation_reproducibility_and_the_seeds():
    # Qwen: the brief assumes 0.33 / 0.47 / 0.70 s per participation; the round is compared for the same participations.
    slow = run("t1-m7", "t1", "m7", [0.3, 0.4, 0.5], round_s=30.0)
    slow["seconds"]["parts"] = {"train": 0.5, "deliver": 0.02, "aggregate": 0.45, "log": 0.03}
    data = session(runs=[slow])
    assumed = (72 * 0.33 + 72 * 0.47 + 48 * 0.70) / 12
    assert digest.cost_ratio(slow)["round"] == pytest.approx(30.0 / assumed) and digest.cost_facts(data)["t1"]["over"]
    found = [finding for finding in digest.findings(data) if finding["topic"] == "耗时"]
    assert [finding["level"] for finding in found] == ["fix", "fix", "note"]
    assert "重放与聚合占每轮的 45%" in found[1]["text"] and "不保留中间量的重放" in found[1]["action"]
    assert f"大约需要 {13.5 * 30.0 / assumed + 12.0 + 0.7:.0f} GPU 小时（EuroSAT 没有实测，按原估计计入）" in found[2]["text"]
    assert digest.day_minutes(slow) == (pytest.approx((20 + 144 * 30.0 + 25 * 4.0) / 60), False)
    # EuroSAT: a centralised reference at 98% is the situation of gate G3, to be decided now.
    saturated = session(runs=[run("t2-m7-day", "t2", "m7", day(0.5, 0.985), group="day"), run("t2-m0-day", "t2", "m0", day(0.6, 0.99), group="day")])
    facts = digest.day_facts(saturated)
    assert facts["saturated_at"] == 24 and facts["ideal"] == pytest.approx(0.99) and facts["last_gap"] == pytest.approx(0.005)
    decision = next(finding for finding in digest.findings(saturated) if finding["topic"] == "数据集")
    assert decision["level"] == "decide" and "RESISC45" in decision["action"]
    assert ("note", "数据集") in levels(session()) and digest.day_facts(session(runs=[])) is None
    assert "| 理想中心化 | 10.0 | 13.5 |" in digest.render(session())
    # A repeated run that differs adds noise to every paired difference.
    noisy = session(repeats=[{"task": "t2", "method": "m7", "rounds_compared": 12, "evaluations_compared": 3, "identical": False,
                             "max_loss_difference": 1e-4, "max_accuracy_difference": 0.002}])
    assert ("decide", "可复现") in levels(noisy) and "评估准确率最大相差 0.2 个点" in digest.render(noisy)
    # HO-FL with and without its zeroth-order part: paired differences and the criterion of section 10.3.
    facts = digest.hofl_facts(session())
    # 0.04, 0.20 and 0.09 points: all positive and their mean of 0.11 above two standard errors (0.09).
    assert facts["gaps"] == pytest.approx([0.04, 0.20, 0.09]) and facts["holds"] is True
    text = digest.render(session())
    assert "| 43 | 本次会话 | 86.00 | 85.80 | +0.20 |" in text and "| 42 | 11-21 的会话 | 85.99 | 85.95 | +0.04 |" in text
    assert "零阶的作用成立，幅度远小于作者报告的值" in text
    # The HO-FL runs are no longer part of the session: without them the report has six answers and no table for them.
    assert digest.hofl_facts(session(t0=[])) is None and "HO-FL 的零阶开与关" not in digest.render(session(t0=[]))
    assert len(digest.answers(session(t0=[]), digest.checklist(session(t0=[])))) == 6 and len(digest.answers(session(), digest.checklist(session()))) == 7
    mixed = session(t0=[{**entry, "accuracy": 0.8590 if entry["name"] == "qwen-zoon-s44" else entry["accuracy"]} for entry in session()["t0"]])
    assert digest.hofl_facts(mixed)["gaps"][-1] == pytest.approx(-0.11) and digest.hofl_facts(mixed)["holds"] is False
    assert "不成立：差距在种子间的波动之内" in digest.render(mixed)
    one_seed = session(t0=session()["t0"][:2])
    assert digest.hofl_facts(one_seed)["holds"] is None and len(digest.hofl_facts(one_seed)["gaps"]) == 2


def test_what_did_not_run_and_what_comes_next():
    stages = session()["stages"]
    stages["selfcheck"] = {"status": "attention", "wall_s": 30.0, "calls": 1, "blocked_by": None}
    stages["smoke"] = stages["pilot"] = {"status": "skipped", "wall_s": 0.0, "calls": 1, "blocked_by": "selfcheck"}
    del stages["t0"]
    blocked = session(stages=stages, selfcheck={"t2": check(4, False), "t1": None, "upstream": None}, t0=[],
                      runs=[run("t2-m7", "t2", "m7", [0.1], status="NOT RUN", rounds=0)],
                      failures=[{"stage": "selfcheck", "name": "t1", "log": "selfcheck/t1.log", "tail": "RuntimeError: CUDA out of memory"}])
    text = digest.render(blocked)
    assert "**关卡 G0：未通过。** 清单第 4 项没有通过" in text and "- 短程运行：跳过（正确性检查没有通过）" in text
    assert "HO-FL 零阶开关" not in text and "| 短程运行 | 跳过 | 0.0 | 正确性检查没有通过 |" in text
    assert "**正确性检查 t1**（`selfcheck/t1.log`）" in text and "RuntimeError: CUDA out of memory" in text
    assert "重跑时可以关掉已经完成、与要改的代码无关的步骤：`RUN_MEMORY`（显存标定）" in text
    # Runs that did not fit into the budget are listed with what they needed.
    late = session(runs=session()["runs"][:5] + [run("t2-m7-day", "t2", "m7", [0.1], group="day", status="SKIPPED", rounds=0, note={"needed_min": 12.0, "left_min": 5.0})],
                   skipped_for_budget=["t2-m7-day"])
    text = digest.render(late)
    assert "EuroSAT · 完整方法 · 0 个仿真小时：预算不够：需要约 12 分钟，只剩 5 分钟" in text and ("note", "用时") in levels(late)
    # An interim report (the HO-FL runs still to come) says so instead of listing them as missing.
    del stages["smoke"], stages["pilot"]
    interim = digest.render(session(stages=stages, t0=[], pending=["t0"]))
    assert "**这是中期报告：HO-FL 零阶开关还没有开始" in interim and "- HO-FL 零阶开关：尚未开始（中期报告）" in interim
    assert "- 短程运行：没有打开，或会话在它之前结束" in interim
    # The dry run says that its numbers mean nothing and judges nothing about the environment.
    tiny = digest.render(session(tiny=True))
    assert "下面的数字没有意义" in tiny and "PyTorch 是" not in tiny
    assert ("note", "环境") in levels(session()) and "不用 GPU 的准备工作占了 9 分钟" in digest.render(session())


# --- reading results --------------------------------------------------------------------------------------------------------

def write_run(path, losses, accuracies, wall=1.0):
    path.mkdir(parents=True)
    rows = [{"round": r, "task_loss": loss, "participant_count": 3, "participants": {"B": 1, "H": 1, "Z": 1}, "invited": {"B": 1, "H": 1, "Z": 2},
             "skipped": {"memory": 0, "eclipse": 1, "energy": 0, "unreachable": 0}, "first_order_late": 0, "moved_up": 0,
             "zo_reports": {"sent": 2, "used": 2, "dropped": 0}, "replica_groups": 1, "battery_min": {"B": None, "H": 0.9, "Z": 0.95 - 0.01 * r},
             "wall_s": wall, "wall_parts_s": {"train": 0.7 * wall, "deliver": 0.1 * wall, "aggregate": 0.2 * wall, "log": 0.0}} for r, loss in enumerate(losses)]
    (path / "rounds.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    ledger = [{"round": r, "kind": kind, "cut": cut, "zo": cut > 0, "wall_s": seconds} for r in range(len(losses)) for kind, cut, seconds in (("B", 0, 0.1), ("H", 3, 0.2), ("Z", 12, 0.3))]
    (path / "ledger.jsonl").write_text("".join(json.dumps(row) + "\n" for row in ledger))
    evals = [{"round": 6 * index, "sim_hours": float(index), "accuracy": value, "macro_f1": value, "loss": 1.0, "replica_difference": 0.0, "wall_s": 2.0}
             for index, value in enumerate(accuracies)]
    (path / "eval.jsonl").write_text("".join(json.dumps(row) + "\n" for row in evals))
    (path / "config.json").write_text(json.dumps({"depth": 12}))
    (path / "summary.json").write_text(json.dumps({"status": "COMPLETED", "wall_s": 99.0, "wall_s_per_round": wall, "participations": {"B": 12, "H": 12, "Z": 12},
                                                   "wall_breakdown_s": {"startup": 15.0}, "energy_wh_per_satellite_day": None, "bytes_per_day": None}))


def test_runs_are_read_from_their_directories(tmp_path):
    assert e0.smoke_verdict(tmp_path / "none") == {"status": "MISSING", "passed": False, "completed": False}
    losses = [2.0, 1.9, None, 1.5, 1.4, 1.3] * 2
    write_run(tmp_path / "short", losses, [0.1, 0.4, 0.6])
    verdict = e0.smoke_verdict(tmp_path / "short")
    assert verdict["passed"] and verdict["completed"] and verdict["loss_first_third"] == pytest.approx(1.8) and verdict["loss_last_third"] == pytest.approx(1.4)
    record = e0.read_run(tmp_path / "short", "t2-m7", "short", "t2", "m7", 2.0, started=True)
    assert (record["status"], record["rounds"], record["depth"], record["wall_s"]) == ("COMPLETED", 12, 12, 99.0)
    assert record["participants"] == {"B": 1, "H": 1, "Z": 1} and record["skipped"]["eclipse"] == 12 and record["zo"] == {"sent": 24, "used": 24, "dropped": 0}
    assert record["battery_min"] == {"H": 0.9, "Z": pytest.approx(0.84)} and record["replica_groups_max"] == 1
    assert record["seconds"]["round"] == 1.0 and record["seconds"]["evaluation"] == 2.0 and record["seconds"]["startup"] == 15.0
    assert record["seconds"]["parts"] == pytest.approx({"train": 0.7, "deliver": 0.1, "aggregate": 0.2, "log": 0.0})
    assert record["participation_seconds"] == [{"kind": "B", "cut": 0, "zo": False, "count": 12, "mean_s": pytest.approx(0.1)},
                                               {"kind": "H", "cut": 3, "zo": True, "count": 12, "mean_s": pytest.approx(0.2)},
                                               {"kind": "Z", "cut": 12, "zo": True, "count": 12, "mean_s": pytest.approx(0.3)}]
    assert [point["accuracy"] for point in record["curve"]] == [0.1, 0.4, 0.6] and digest.finished(record)
    # A stage that ran leaves MISSING for a run without a directory; a stage that never ran leaves NOT RUN; a budget note SKIPPED.
    assert e0.read_run(tmp_path / "none", "x", "short", "t2", "m7", 2.0, started=True)["status"] == "MISSING"
    assert e0.read_run(tmp_path / "none", "x", "short", "t2", "m7", 2.0, started=False)["status"] == "NOT RUN"
    assert e0.read_run(tmp_path / "none", "x", "day", "t2", "m7", 24.0, started=True, note={"needed_min": 9, "left_min": 1})["status"] == "SKIPPED"
    # The time of a day-long run follows from the short one: start-up, 144 rounds, 25 evaluations.
    assert e0.day_estimate_s(tmp_path / "short", 24.0) == pytest.approx(15.0 + 144 * 1.0 + 25 * 2.0) and e0.day_estimate_s(tmp_path / "none", 24.0) is None
    # The first rounds of a longer run with the same configuration: identical, or how far apart.
    write_run(tmp_path / "day", losses + [1.2] * 6, [0.1, 0.4, 0.6, 0.7])
    same = e0.compare_repeat(tmp_path / "short", tmp_path / "day")
    assert same == {"rounds_compared": 12, "evaluations_compared": 3, "identical": True, "max_loss_difference": 0.0, "max_accuracy_difference": 0.0}
    write_run(tmp_path / "other", [loss if loss is None else loss + 0.01 for loss in losses], [0.1, 0.4, 0.65])
    apart = e0.compare_repeat(tmp_path / "short", tmp_path / "other")
    assert not apart["identical"] and apart["max_loss_difference"] == pytest.approx(0.01) and apart["max_accuracy_difference"] == pytest.approx(0.05)
    assert e0.compare_repeat(tmp_path / "short", tmp_path / "none") is None


def test_the_regression_verdicts_become_the_baseline_record(tmp_path):
    assert e0.baseline_record(tmp_path, 1) == {"status": "INCOMPLETE", "returncode": 1}

    def verdicts(hofl, replay="PASS"):
        path = tmp_path / "regress" / "abc-gpu"
        path.mkdir(parents=True, exist_ok=True)
        (path / "verdicts.json").write_text(json.dumps({"head": "abc", "golden_mode": "bitwise", "i1": {
            "hofl": {"verdict": hofl, "max_abs_state_diff": 0.0, "differences": ["round 2: task_loss differs"] if hofl == "FAIL" else []},
            "fedavg": {"verdict": "PASS_BITWISE", "max_abs_state_diff": 0.0, "differences": []}},
            "i2": {"verdict": replay, "head": {"status": "PASSED", "max_abs_error": 1e-12, "comparisons": 672}}}))
        return e0.baseline_record(tmp_path, 0)

    record = verdicts("PASS_BITWISE")
    assert record["status"] == "PASSED" and record["replay"] == {"verdict": "PASS", "max_abs_error": 1e-12, "comparisons": 672}
    assert verdicts("PASS_TOLERANCE")["status"] == "PASSED" and verdicts("INCONCLUSIVE")["status"] == "INCONCLUSIVE"
    failed = verdicts("FAIL")
    assert failed["status"] == "FAILED" and failed["configs"]["hofl"]["first_difference"] == "round 2: task_loss differs"
    assert verdicts("PASS_BITWISE", replay="FAIL")["status"] == "FAILED"


JUNIT = """<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" errors="0" failures="{failures}" skipped="0" tests="3">
<testcase classname="tests.sat.test_a" name="test_fine" time="0.1"/>{cases}</testsuite></testsuites>"""
FAILED_CASE = '<testcase classname="tests.sat.test_a" name="test_timing[t2]" time="0.2"><failure message="assert 0.0 &gt; 0">trace\nback</failure></testcase>'


def test_a_test_that_passes_when_repeated_is_reported_but_does_not_stop_the_session(tmp_path, monkeypatch):
    assert e0.read_junit(tmp_path / "none.xml") == ({}, [])
    (tmp_path / "one.xml").write_text(JUNIT.format(failures=1, cases=FAILED_CASE))
    counts, failed = e0.read_junit(tmp_path / "one.xml")
    assert counts == {"tests": 3, "failures": 1, "errors": 0, "skipped": 0} and failed == [{"id": "tests.sat.test_a::test_timing[t2]", "message": "assert 0.0 > 0"}]

    def session_of(outcomes):
        """Run the stage with pytest replaced: each call writes the given JUnit report and returns its exit code."""
        calls, root = [], tmp_path / f"root{len(list(tmp_path.iterdir()))}"
        root.mkdir()

        def stream(ctx, command, log_path, label=None, show=None, timeout=None, tail=True):
            failures = outcomes[len(calls)]
            calls.append(command)
            target = next(part.split("=", 1)[1] for part in command if part.startswith("--junitxml="))
            Path(target).write_text(JUNIT.format(failures=failures, cases=FAILED_CASE * failures))
            return (1 if failures else 0), 5.0, False

        monkeypatch.setattr(e0, "stream", stream)
        ok = e0.stage_tests(types.SimpleNamespace(root=root, python="python"))
        return ok, json.loads((root / "tests.json").read_text()), calls

    ok, record, calls = session_of([0])
    assert ok and len(calls) == 1 and (record["returncode"], record["failed"], record["flaky"]) == (0, [], [])
    assert f"cache_dir={calls[0][calls[0].index('-o') + 1].split('=', 1)[1]}".endswith("pytest-cache") and "no:cacheprovider" not in calls[0]
    # Failed once, passed when repeated: the stage passes and the test is listed as unreliable.
    ok, record, calls = session_of([1, 0])
    assert ok and len(calls) == 2 and "--last-failed" in calls[1] and record["returncode"] == 0 and record["failures"] == 0 and record["wall_s"] == 10.0
    assert record["failed"] == [] and [entry["id"] for entry in record["flaky"]] == ["tests.sat.test_a::test_timing[t2]"]
    data = session(tests=record)
    assert digest.gate(digest.checklist(data)) == ("PASS", []) and "（另有 1 项第一次失败、重跑通过）" in digest.checklist(data)[0][3]
    finding = next(finding for finding in digest.findings(data) if finding["topic"] == "测试")
    assert finding["level"] == "fix" and "`tests.sat.test_a::test_timing[t2]`（assert 0.0 > 0）" in finding["text"] and "不稳定的测试" in finding["text"]
    # Failed twice: the stage fails.
    ok, record, calls = session_of([1, 1])
    assert not ok and record["returncode"] == 1 and record["flaky"] == [] and len(record["failed"]) == 1
    assert ("block", "测试") in levels(session(tests=record))


# --- which stage may start --------------------------------------------------------------------------------------------------

def test_a_stage_does_not_start_when_what_it_builds_on_did_not_pass(tmp_path):
    assert all(e0.blocked(tmp_path, stage) is None for stage in e0.STAGES)   # nothing ran: nothing is blocked
    for name in ("env", "baseline", "tests", "trace", "data"):
        e0.record_stage(tmp_path, name, "ok", 1.0)
    assert e0.blocked(tmp_path, "selfcheck") is None and e0.blocked(tmp_path, "t0") is None
    e0.record_stage(tmp_path, "selfcheck", "attention", 2.0)
    assert e0.blocked(tmp_path, "smoke") == e0.blocked(tmp_path, "pilot") == "selfcheck"
    assert e0.blocked(tmp_path, "memory") is None and e0.blocked(tmp_path, "t0") is None   # neither builds on the self-check
    e0.record_stage(tmp_path, "selfcheck", "ok", 3.0)
    assert e0.stage_records(tmp_path)["selfcheck"] == {"status": "ok", "wall_s": 5.0, "calls": 2, "blocked_by": None, "error": None,
                                                       "finished_at": e0.stage_records(tmp_path)["selfcheck"]["finished_at"]}
    assert e0.blocked(tmp_path, "smoke") is None
    # A stage skipped for this reason blocks what builds on it in turn; the baseline blocks everything on the GPU.
    e0.record_stage(tmp_path, "baseline", "attention", 1.0)
    assert {stage: e0.blocked(tmp_path, stage) for stage in ("selfcheck", "memory", "smoke", "pilot", "t0")} == dict.fromkeys(
        ("selfcheck", "memory", "smoke", "pilot", "t0"), "baseline")
    assert e0.blocked(tmp_path, "tests") is None and e0.blocked(tmp_path, "feedback") is None
    e0.record_stage(tmp_path, "baseline", "ok", 1.0)
    # The day-long runs wait for the two-hour runs of the same methods.
    passed = {"status": "COMPLETED", "passed": True, "completed": True}
    (tmp_path / "smoke.json").write_text(json.dumps({"verdicts": {"t2-m7": passed, "t2-m0": {**passed, "passed": False}}}))
    assert e0.blocked(tmp_path, "pilot") is None   # the centralised reference only has to complete
    (tmp_path / "smoke.json").write_text(json.dumps({"verdicts": {"t2-m7": {**passed, "passed": False}, "t2-m0": passed}}))
    assert e0.blocked(tmp_path, "pilot") == "smoke"


def test_a_blocked_stage_is_recorded_as_skipped(tmp_path, capsys):
    e0.record_stage(tmp_path, "selfcheck", "attention", 2.0)
    assert e0.main(["smoke", "--tiny", "--root", str(tmp_path)]) == 1
    printed = capsys.readouterr().out
    assert "skipped: the stage 'selfcheck' did not pass" in printed and "smoke=SKIPPED" in printed
    assert e0.stage_records(tmp_path)["smoke"]["status"] == "skipped" and e0.stage_records(tmp_path)["smoke"]["blocked_by"] == "selfcheck"
    assert not (tmp_path / "smoke").exists()


def test_a_stage_that_breaks_is_recorded_and_reported(tmp_path, monkeypatch, capsys):
    def broken(ctx):
        raise ValueError("no such thing")

    monkeypatch.setattr(e0, "stage_trace", broken)
    assert e0.main(["trace", "smoke", "--tiny", "--root", str(tmp_path)]) == 1
    records = e0.stage_records(tmp_path)
    assert records["trace"]["status"] == "attention" and "ValueError: no such thing" in records["trace"]["error"]
    assert records["smoke"]["status"] == "skipped" and records["smoke"]["blocked_by"] == "trace" and "trace=ATTENTION, smoke=SKIPPED" in capsys.readouterr().out
    # The report names the step, shows the traceback and counts it against the gate's prerequisites.
    data = session(stages={**session()["stages"], "trace": records["trace"], "smoke": records["smoke"]})
    assert ("block", "流程") in levels(data)
    text = digest.render(data)
    assert "“轨迹”这一步的程序自身出错" in text and "**轨迹：程序自身出错**" in text and "ValueError: no such thing" in text


def test_the_report_survives_its_own_errors(monkeypatch):
    def broken(data, add):
        raise KeyError("missing")

    monkeypatch.setattr(digest, "_memory", broken)
    monkeypatch.setattr(digest, "_section_memory", broken)
    found = digest.findings(session())
    assert [finding["level"] for finding in found if finding["topic"] == "报告"] == ["fix"] and ("good", "基线") in {(f["level"], f["topic"]) for f in found}
    text = digest.render(session())
    assert "这一小节生成时出错：KeyError('missing')" in text and "### 4.5 零阶估计的保真度" in text and "**关卡 G0：通过。**" in text
    monkeypatch.setattr(digest, "answers", lambda data, rows: 1 / 0)
    assert "各个问题的答案生成时出错" in digest.render(session())
    # The last resort when nothing can be rendered: the error and the plain state of every stage.
    plain = digest.fallback(session(), "Traceback ...\nZeroDivisionError: division by zero")
    assert "报告生成出错" in plain and "ZeroDivisionError" in plain and "- selfcheck: ok，1.0 分钟" in plain and "- run t2-m7: COMPLETED" in plain
    assert "- baseline: status=PASSED, returncode=0" in plain and digest.fallback({}, "x").startswith("# RHO-FFT")


def test_expected_hashes_and_run_lists():
    assert set(e0.EXPECTED["partition_sha256"]) == {f"{task}-{variant}-s{seed}" for task in ("t1", "t2") for variant in ("base", "zonly") for seed in (42, 43, 44)}
    assert set(e0.EXPECTED["trace_sha256"]) == {"global_land_mask", "duty_cycle"}
    assert [name for name, _, _ in e0.SMOKE][0] == "t2-m7" and e0.T0_SEEDS == (43, 44) and e0.RUN_SPLIT == "validation"
    # Every day-long run repeats a two-hour run, which is what the reproducibility check and its time estimate rest on.
    assert {(task, method) for _, task, method in e0.DAY} <= {(task, method) for _, task, method in e0.SMOKE}
    assert e0.T0_BASE == {"model": "qwen", "dataset": "ag_news", "partition": "dirichlet_05", "method": "hofl", "beta": 0.75}
    reference = json.loads((REPO / e0.T0_REFERENCE["file"]).read_text())["runs"]
    assert (reference[e0.T0_REFERENCE["on"]]["final_accuracy_pct"], reference[e0.T0_REFERENCE["off"]]["final_accuracy_pct"]) == (85.99, 85.95)
    assert set(e0.REQUIRES) <= set(e0.STAGES) and all(set(needs) <= set(e0.STAGES) for needs in e0.REQUIRES.values())
    # The HO-FL runs stay callable but are not part of a session any more, so no switch of the notebook belongs to them.
    assert set(digest.SWITCHES) | {"env", "feedback", "t0"} == set(e0.STAGES) == set(digest.STAGES)
    assert e0.SESSION == tuple(stage for stage in e0.STAGES if stage != "t0")


def test_the_baseline_regression_is_not_repeated_while_its_record_stands(tmp_path, monkeypatch, capsys):
    record = e0.VERIFIED["baseline"]
    assert set(record["configs"]) == {"hofl", "fedavg", "decomfl"} and all(entry["verdict"] == "PASS_BITWISE" for entry in record["configs"].values())
    calls, changed = [], [False]
    monkeypatch.setattr(e0.subprocess, "run", lambda command, **options: (calls.append(command), types.SimpleNamespace(returncode=int(changed[0])))[1])
    ctx = types.SimpleNamespace(root=tmp_path, tiny=False)
    assert e0.baseline_verified(ctx) is None and calls == []            # no record of the environment: run it
    (tmp_path / "env.json").write_text(json.dumps({"gpu_name": record["gpu_name"], "torch": record["torch"]}))
    assert e0.baseline_verified(ctx) == record
    assert calls[0][3:6] == ["diff", "--quiet", record["commit"]] and list(calls[0][-4:]) == list(e0.BASELINE_PATHS)   # against the working tree
    assert e0.stage_baseline(ctx) is True and "is not repeated" in capsys.readouterr().out
    written = json.loads((tmp_path / "baseline.json").read_text())
    assert written["status"] == "PASSED" and written["reused"]["commit"] == record["commit"] and written["replay"]["verdict"] == "PASS"
    data = session(baseline=written)
    assert digest.gate(digest.checklist(data)) == ("PASS", []) and "此后 HO-FL 的代码没有改动，本次不重跑" in digest.checklist(data)[1][3]
    assert "所以没有重跑" in digest.render(data)
    changed[0] = True                                                   # the code of upstream HO-FL changed: run it
    assert e0.baseline_verified(ctx) is None
    changed[0] = False
    (tmp_path / "env.json").write_text(json.dumps({"gpu_name": "another GPU", "torch": record["torch"]}))
    assert e0.baseline_verified(ctx) is None                            # another GPU: run it
    assert e0.baseline_verified(types.SimpleNamespace(root=tmp_path, tiny=True)) == record   # a dry run only looks at the code


def test_what_the_page_shows_of_a_stage():
    lines = ["EVAL   0.0 h (round 0/144) | x", "EVAL   2.0 h (round 12/144) | x", "EVAL   3.0 h (round 18/144) | x", "EVAL   8.0 h (round 48/144) | x",
             "round 18/144 (3.0 h) | x", "COMPLETED: t2 m7 seed 42", "DIVERGED: t2 m7", "something else"]
    assert [e0.progress(line) for line in lines] == [True, True, False, True, False, True, True, False]
    assert e0.verdicts("Self-check of task t2 (vit)") and e0.verdicts("15 of 15 checks passed") and e0.verdicts("| 4 | a check | FAIL | detail |")
    assert not e0.verdicts("| 4 | a check | pass | detail |") and not e0.verdicts("| 3 | 0.01 (configured) | 64 | 0.99 | 1.00 | 0.95 |")
