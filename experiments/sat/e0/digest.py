"""The feedback report of the E0 session: what the session showed and what should change because of it.

``e0.py`` collects the results of every stage into one dictionary; this module turns it into the text the research
lead copies back from Colab. It only reads that dictionary, so it can be tested without running anything.

The report is written in Chinese, like the milestone reports (brief v2, section 12): it is read by the research
lead on the Colab page. Its parts:

    1  the state of gate G0 and short answers to the six questions the session was run for
    2  the checklist of gate G0 (brief v2, section 10.1, E0-a)
    3  findings: what has to be fixed, what the research lead has to decide, what is worth knowing
    4  the numbers behind them
    5  the end of the log of everything that failed
    6  what did not run, and what the next session should do

The Colab session runs measurements and training runs only (the checks of gate G0 passed there and were taken out
on 2026-11-25; they stay callable). A report of such a session has no checklist and judges no gate: part 2 says
what the session covered, and the questions and tables about the checks are left out. ``checked`` tells the two apart.

A finding is ``{"level", "topic", "text", "action"}``. Levels: ``block`` (a gate item failed), ``fix`` (code or
pipeline should change), ``decide`` (a choice that affects the experiments: the research lead's), ``note`` (worth
knowing) and ``good`` (a confirmation). Every threshold a finding depends on is a constant below.
"""
import statistics

# --- thresholds and reference values (brief v2) ------------------------------------------------------------------------

SATURATION = 0.98            # gate G3: every method close to 98% on EuroSAT makes accuracy uninformative
TARGET_SHARE = 0.95          # section 9.2: the target accuracy is 95% of the final accuracy of the centralised method
COST_FACTOR = 1.3            # section 10.4: report when the measured cost is 30% above the estimate
ASSUMED_PARTICIPATION_S = {"first_order": 0.33, "hybrid": 0.47, "forward_only": 0.70}   # section 10.4, Qwen2.5-1.5B
ASSUMED_DAY_RUN_MIN = {"t1": (7, 30), "t2": (3, 6)}
PLAN_HOURS = {"t1": 13.5, "t2": 12.0, "t0": 0.7}   # the table of section 10.4: 26 h in total
REPLAY_SHARE = 0.3           # share of a round spent on replay and aggregation above which a leaner replay pays off
ESTIMATOR_GOOD, ESTIMATOR_POOR = 0.9, 0.6          # correlation of v / mu with the exact directional derivative
AUTHORS_ZO_GAP = 3.32        # points HO-FL loses without its zeroth-order part, as reported by its authors
ROUNDS_PER_DAY, EVALUATIONS_PER_DAY = 144, 25

METHODS = {"m0": "理想中心化", "m2": "只让强星训练", "m4": "纯零阶泛洪", "m7": "完整方法"}
TASKS = {"t1": "HumAID", "t2": "EuroSAT"}
MODELS = {"t1": "Qwen2.5-1.5B", "t2": "ViT-B/16"}
KINDS = {"B": "强星", "H": "GPU 卫星", "Z": "推理芯片卫星"}
VERDICTS = {"PASS": "通过", "FAIL": "未通过", "NOT RUN": "未运行"}
STAGES = {"env": "环境记录", "baseline": "基线回归", "tests": "CPU 测试", "trace": "轨迹", "data": "数据与划分",
          "selfcheck": "正确性检查", "memory": "显存标定", "smoke": "短程运行", "pilot": "一整天的运行",
          "t0": "HO-FL 零阶开关", "feedback": "反馈报告"}
STEPS = {"code": "取代码", "environment": "建环境", "dry_run": "预演（CPU）"}   # steps of the notebook itself
SWITCHES = {"memory": "RUN_MEMORY", "smoke": "RUN_SHORT_RUNS", "pilot": "RUN_DAY_RUNS"}   # the steps of the notebook and their switches
CHECKS = ("baseline", "tests", "trace", "data", "selfcheck")   # the check stages: callable, not part of the Colab session
REASONS = {"memory": "显存", "eclipse": "地影", "energy": "电量", "unreachable": "不可达"}
WHERE = {2: "`ho_fl/algorithms.py` 的 `hybrid_step`（显式给出分界的路径）",
         3: "`ho_fl/sat/client.py` 的 `train_participation`",
         4: "`ho_fl/sat/aggregate.py`（`round_terms`、`sum_terms` 和零阶重放）",
         5: "`ho_fl/sat/aggregate.py` 的 `replay_zeroth_order`（重放在这块 GPU 上不确定）"}


# --- formatting ---------------------------------------------------------------------------------------------------------

def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(" --- " for _ in headers) + "|"]
    lines += ["| " + " | ".join("" if cell is None else str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def pct(value, digits=2):
    return "无" if value is None else f"{100 * value:.{digits}f}"


def num(value, digits=2):
    return "无" if value is None else f"{value:.{digits}f}"


def small(value):
    return "无" if value is None else ("0" if value == 0 else f"{value:.3g}")


def why(entry):
    """Why a stage was skipped: the stage it builds on that did not pass."""
    name = (entry or {}).get("blocked_by")
    return f"{STAGES.get(name, name)}没有通过" if name else ""


def budget(note):
    """Why a single run was not started: it did not fit into what was left of the session's time budget."""
    if not note:
        return "没有开始"
    return f"预算不够：需要约 {note['needed_min']:.0f} 分钟，只剩 {note['left_min']:.0f} 分钟"


def describe(run):
    """A run in words: task, method, simulated duration."""
    return f"{TASKS.get(run['task'], run['task'])} · {METHODS.get(run['method'], run['method'])} · {run['hours']:g} 个仿真小时"


def checked(data):
    """Whether a check stage ran in this session, or left a result in its directory. Without one the report is
    that of a session of measurements and runs: no checklist, no verdict on the gate."""
    stages = data.get("stages") or {}
    return bool(any(name in stages for name in CHECKS) or data.get("baseline") or data.get("tests") or data.get("trace") or data.get("partitions")
                or any((data.get("selfcheck") or {}).values()))


def find_run(data, name):
    return next((run for run in data.get("runs") or [] if run["name"] == name), None)


def finished(run):
    return bool(run is not None and run.get("status") in ("COMPLETED", "DIVERGED") and run.get("rounds"))


def accuracy_at(run, hours):
    return next((point["accuracy"] for point in run["curve"] if abs(point["sim_hours"] - hours) < 1e-6), None)


# --- numbers several parts of the report share ---------------------------------------------------------------------------

def day_minutes(run):
    """Minutes of a 24-hour run: measured if this run is one, otherwise projected from its time per round and per
    evaluation. The second value says which of the two it is."""
    if run["rounds"] >= ROUNDS_PER_DAY and run.get("wall_s"):
        return run["wall_s"] / 60, True
    seconds = run["seconds"]
    if seconds["evaluation"] is None:
        return None, False
    return (seconds["startup"] + ROUNDS_PER_DAY * seconds["round"] + EVALUATIONS_PER_DAY * seconds["evaluation"]) / 60, False


def cost_ratio(run):
    """Measured seconds of a round over what the brief assumed for the same participations (section 10.4): for the
    whole round and for the local training alone. ``None`` when a participation has no assumed time."""
    assumed = measured = 0.0
    for row in run.get("participation_seconds") or []:
        if row["cut"] == 0:
            kind = "first_order"
        elif run.get("depth") and row["cut"] >= run["depth"]:
            kind = "forward_only"
        elif row["zo"]:
            kind = "hybrid"
        else:
            return None
        assumed += row["count"] * ASSUMED_PARTICIPATION_S[kind]
        measured += row["count"] * row["mean_s"]
    if not assumed:
        return None
    return {"round": run["seconds"]["round"] / (assumed / run["rounds"]), "training": measured / assumed, "assumed_round_s": assumed / run["rounds"]}


def cost_facts(data):
    """Per task, from the full method (its day-long run where there is one): time per round and where it goes,
    minutes of a 24-hour run, and the factor by which the measurement differs from the estimate of the brief."""
    facts = {}
    for task, name in (("t2", "t2-m7"), ("t1", "t1-m7")):
        day = find_run(data, f"{name}-day")
        run = day if finished(day) else find_run(data, name)
        if not finished(run):
            continue
        minutes, measured = day_minutes(run)
        low, high = ASSUMED_DAY_RUN_MIN[task]
        ratio = cost_ratio(run) if task == "t1" else None
        scale = ratio["round"] if ratio else (minutes / ((low + high) / 2) if task == "t2" and minutes is not None else None)
        facts[task] = {"run": run, "minutes": minutes, "measured": measured, "ratio": ratio, "scale": scale,
                       "over": bool((ratio and ratio["round"] > COST_FACTOR) or (minutes is not None and minutes > high * COST_FACTOR)),
                       "train": run["seconds"]["parts"].get("train", 0.0), "replay": run["seconds"]["parts"].get("aggregate", 0.0)}
    return facts


def plan_hours(facts):
    """The 26 GPU hours of the brief scaled by the measured factors; a task without a measurement keeps its estimate."""
    if not any(entry["scale"] for entry in facts.values()):
        return None
    return sum(PLAN_HOURS[task] * ((facts.get(task) or {}).get("scale") or 1.0) for task in ("t1", "t2")) + PLAN_HOURS["t0"]


def estimator_facts(data):
    """Per task: the range over the cuts of the correlation between the zeroth-order scalars and the exact
    directional derivatives at the configured radius, and the radius with the best worst case."""
    facts = {}
    for name in ("t2", "t1"):
        rows = ((data.get("selfcheck") or {}).get(name) or {}).get("estimator") or []
        by_mu = {}
        for row in rows:
            if row["correlation"] is not None:
                by_mu.setdefault(row["mu"], []).append(row["correlation"])
        configured = next((row["mu"] for row in rows if row["configured"]), None)
        if configured not in by_mu:
            continue
        best = max(by_mu, key=lambda mu: min(by_mu[mu]))
        facts[name] = {"mu": configured, "low": min(by_mu[configured]), "high": max(by_mu[configured]), "best_mu": best, "best_low": min(by_mu[best]),
                       "nonfinite": [row["mu"] for row in rows if not row["finite"]]}
    return facts


def day_facts(data):
    """The two day-long runs on the vision task: final accuracies, when the centralised reference reaches the
    saturation level, when each reaches the target of section 9.2, and how far apart they are."""
    ideal, full = find_run(data, "t2-m0-day"), find_run(data, "t2-m7-day")
    if not finished(ideal) or not ideal["curve"]:
        return None
    final = ideal["curve"][-1]["accuracy"]
    facts = {"hours": ideal["hours"], "ideal": final, "full": None, "target": TARGET_SHARE * final,
             "saturated_at": next((point["sim_hours"] for point in ideal["curve"] if point["accuracy"] >= SATURATION), None)}
    reach = lambda run: next((point["sim_hours"] for point in run["curve"] if point["accuracy"] >= facts["target"]), None)  # noqa: E731
    facts["ideal_reaches"] = reach(ideal)
    if finished(full) and full["curve"]:
        gaps = [(point["sim_hours"], other["accuracy"] - point["accuracy"]) for point, other in zip(full["curve"], ideal["curve"])]
        facts.update(full=full["curve"][-1]["accuracy"], full_reaches=reach(full), widest=max(gaps, key=lambda pair: pair[1]), last_gap=gaps[-1][1])
    return facts


def hofl_by_seed(data):
    reference = data.get("t0_reference") or {}
    by_seed = {42: {"on": reference.get("on"), "off": reference.get("off"), "source": "11-21 的会话"}}
    for run in data.get("t0") or []:
        entry = by_seed.setdefault(run["seed"], {"source": "本次会话"})
        entry[run["mode"]] = None if run.get("accuracy") is None else 100 * run["accuracy"]
    return by_seed


def hofl_facts(data):
    """The paired differences (zeroth-order part on minus off) over the seeds and the criterion of section 10.3:
    every difference positive and the mean above two standard errors."""
    gaps = [entry["on"] - entry["off"] for _, entry in sorted(hofl_by_seed(data).items()) if entry.get("on") is not None and entry.get("off") is not None]
    if len(gaps) < 2 or not any(run.get("accuracy") is not None for run in data.get("t0") or []):
        return None
    mean, deviation = statistics.mean(gaps), statistics.stdev(gaps)
    return {"gaps": gaps, "mean": mean, "sd": deviation,
            "holds": None if len(gaps) < 3 else all(gap > 0 for gap in gaps) and mean > 2 * deviation / len(gaps) ** 0.5}


# --- the checklist of gate G0 -------------------------------------------------------------------------------------------

def checklist(data):
    """E0-a as rows ``(item, what, verdict, evidence)``; verdicts are PASS, FAIL or NOT RUN."""
    def verdict(flag):
        return "NOT RUN" if flag is None else ("PASS" if flag else "FAIL")

    tests, trace, baseline = data.get("tests"), data.get("trace"), data.get("baseline")
    rows = [("1", "原有测试与新增测试全部通过", verdict(None if tests is None else tests["returncode"] == 0),
             "未运行" if tests is None else f"{tests.get('tests', '?')} 项，失败 {tests.get('failures', '?')}，出错 {tests.get('errors', '?')}"
             + (f"（另有 {len(tests['flaky'])} 项第一次失败、重跑通过）" if tests.get("flaky") else ""))]
    if baseline is None or baseline["status"] == "INCOMPLETE":
        rows.append(("1", "基线回归：默认设置下与上游 HO-FL 结果一致，重放通过", "NOT RUN", "未运行" if baseline is None else "没有跑完，见第 5 节"))
    else:
        verdicts = "，".join(f"{name} {entry['verdict']}" for name, entry in sorted(baseline["configs"].items()))
        reused = baseline.get("reused")
        rows.append(("1", "基线回归：默认设置下与上游 HO-FL 结果一致，重放通过", verdict(baseline["status"] == "PASSED"),
                     f"{verdicts}；重放 {baseline['replay']['verdict']}" + ("；需要研究负责人判定" if baseline["status"] == "INCONCLUSIVE" else "")
                     + (f"（{reused['date']} 在提交 {reused['commit'][:7]} 上通过；此后 HO-FL 的代码没有改动，本次不重跑）" if reused else "")))
    items = {"2": "显式给出分界的本地训练一步：与原推导逐位相同；分界 0 等于全反向传播；分界在最高层时各层只靠零阶",
             "3": "卫星侧一次参与与上游 `train_client` 逐位相同",
             "4": "理想网络下，卫星聚合等于中心化的“求和后除以人数”（一阶逐位相同，零阶相差不超过 1e-7）",
             "5": "零阶重放在每个副本上逐位相同"}
    for item, what in items.items():
        found = [(name, row) for name, check in (data.get("selfcheck") or {}).items() if check for row in check["rows"] if str(row["item"]) == item]
        sources = sorted({name for name, _ in found})
        rows.append((item, what, verdict(all(row["passed"] for _, row in found) if found else None),
                     "未运行" if not found else f"{sum(row['passed'] for _, row in found)} / {len(found)} 项检查通过（{'、'.join(sources)}）"))
    rows.append(("6", "轨迹统计与实现说明附录 A 一致", verdict(None if trace is None else trace["returncode"] == 0),
                 "未运行" if trace is None else f"陆地信息来自 {trace['land_source']}，轨迹哈希 {(trace['trace_sha256'] or '无')[:16]}"))
    first = find_run(data, "t2-m7")
    started = first is not None and first["status"] not in ("SKIPPED", "NOT RUN")
    if not started:
        evidence = "未运行"
    elif finished(first):
        evidence = f"{first['status']}，训练损失 {num(first['verdict'].get('loss_first_third'), 4)} → {num(first['verdict'].get('loss_last_third'), 4)}"
    else:
        evidence = f"没有跑完（{first['status']}）"
    rows.append(("7", "EuroSAT 上完整方法跑 2 个仿真小时：损失下降，数值有限",
                 verdict(finished(first) and first["verdict"]["passed"] if started else None), evidence))
    return rows


def gate(rows):
    """``(state, items)``: FAIL with the failed items, OPEN with the items that have no result, or PASS."""
    failed = sorted({row[0] for row in rows if row[2] == "FAIL"})
    missing = sorted({row[0] for row in rows if row[2] == "NOT RUN"})
    if failed:
        return "FAIL", failed
    return ("OPEN", missing) if missing else ("PASS", [])


# --- findings -----------------------------------------------------------------------------------------------------------

def findings(data):
    found = []

    def add(level, topic, text, action=None):
        found.append({"level": level, "topic": topic, "text": text, "action": action})

    for rule in (_environment, _stage_errors, _baseline, _tests, _inputs, _selfcheck, _estimator, _memory, _runs, _run_inputs, _repeats, _cost, _day_runs, _hofl, _session):
        try:
            rule(data, add)
        except Exception as error:   # one rule that cannot read the data must not take the other findings with it
            add("fix", "报告", f"生成“{rule.__name__.strip('_')}”这部分发现时出错：{error!r}。其余发现不受影响。", "按这个错误修改 `experiments/sat/e0/digest.py`。")
    return found


def _stage_errors(data, add):
    for name, entry in (data.get("stages") or {}).items():
        if entry.get("error"):
            add("fix" if name == "feedback" else "block", "流程", f"“{STAGES.get(name, name)}”这一步的程序自身出错，没有跑完（回溯见第 5 节）。依赖它的步骤已被跳过。",
                "这是流水线的错误，不是实验结果：按回溯修改 `experiments/sat/e0/e0.py` 或它调用的模块，然后重跑。")


def _environment(data, add):
    env = data.get("env") or {}
    if not env or data.get("tiny"):
        return
    if env.get("torch") and not env["torch"].startswith("2.6.0"):
        add("note", "环境", f"PyTorch 是 {env['torch']}，不是固定的 2.6.0（固定版本用不了这块 GPU 时，notebook 会自动换用）。"
                            + ("基线回归在同一环境里做，所以“没有改坏 HO-FL”的结论不受影响；" if data.get("baseline") else "") + "数值只对这个环境成立。")
    if env.get("land_mask_installed") is False:
        add("fix", "环境", "陆地掩膜包没有装上，成像判断用的是占空比替代，成像比例和轨迹哈希都与正式设置不同。",
            "检查 notebook 的环境步骤是否安装了可选依赖组 `sat`（`global-land-mask`）。")
    if (env.get("git") or {}).get("dirty"):
        add("note", "环境", "运行时代码目录里有未提交的改动，本次结果不能对应到某一个提交。")


def _baseline(data, add):
    baseline = data.get("baseline")
    if baseline is None:
        return
    if baseline["status"] == "INCOMPLETE":
        add("block", "基线", "基线回归没有跑完（日志末尾见第 5 节），“没有改坏 HO-FL”这一条没有结果，后面依赖它的步骤都被跳过。",
            "多半是参照提交的运行或环境本身出了问题；按日志修复后重跑。")
        return
    if baseline["status"] == "PASSED":
        mode = "逐位相同" if all(entry["verdict"] == "PASS_BITWISE" for entry in baseline["configs"].values()) else "在容差内相同"
        reused = baseline.get("reused")
        add("good", "基线", f"HO-FL、FedAvg、DeComFL 三个冒烟配置在当前代码与上游提交上{mode}，重放通过"
                            f"（{baseline['replay'].get('comparisons', '?')} 项比较，最大误差 {small(baseline['replay'].get('max_abs_error'))}）。"
                            + (f"这是 {reused['date']} 在同一块 GPU、同一版 PyTorch 上的结果；HO-FL 的代码此后没有改动，所以没有重跑。" if reused else ""))
        return
    for name, entry in sorted(baseline["configs"].items()):
        if entry["verdict"] in ("FAIL", "RUN_FAILED"):
            add("block", "基线", f"基线回归未通过：`{name}` 的结果与上游不同（{entry.get('first_difference') or '日志没有差别，最终参数不同'}）。",
                "卫星分支对原有文件只有三处改动：`ho_fl/algorithms.py` 的 `hybrid_step`（参数 `cut`）、`ho_fl/train.py` 的 `local_step`、"
                "`ho_fl/model.py` 的 `block_index`。先查这三处；第一处出现差别的轮次说明是哪一步。")
        elif entry["verdict"] == "INCONCLUSIVE":
            add("decide", "基线", f"`{name}` 的基线回归无法判定：这块 GPU 上参照提交自己跑两次的差就超过了容差"
                                  f"（当前代码与参照的差 {small(entry.get('max_abs_state_diff'))}）。", "请决定在这块 GPU 上怎样判定基线等价。")
    if baseline["replay"]["verdict"] != "PASS":
        add("block", "基线", "`ho-fl-replay` 在当前代码上没有通过。", "`Backend.local_step` 或 `hybrid_step` 的行为与 `replay.py` 的预期不一致，先查显式分界的改动。")


def _tests(data, add):
    tests = data.get("tests")
    if tests is not None and tests.get("flaky"):
        add("fix", "测试", f"有 {len(tests['flaky'])} 项测试第一次失败、单独重跑后通过：" + "；".join(f"`{entry['id']}`（{entry['message']}）" for entry in tests["flaky"][:6])
            + "。这是不稳定的测试，不是被测代码的问题，本次按通过处理。", "把这些测试改成不依赖时间和机器负载。")
    if tests is None or tests["returncode"] == 0:
        return
    names = "；".join(f"`{entry['id']}`（{entry['message']}）" for entry in (tests.get("failed") or [])[:6])
    missing = any("ModuleNotFoundError" in entry["message"] for entry in tests.get("failed") or [])
    add("block", "测试", f"CPU 测试有 {tests.get('failures', 0) + tests.get('errors', 0)} 项没有通过：{names or '见第 5 节的日志'}。",
        ("测试用到了 Colab 的固定环境里没有装的包（本机恰好有）：让测试不依赖它，或把它加进 `pyproject.toml` 的依赖。" if missing else
         "这些测试在本机（版本与 Colab 的固定版本不同）是通过的，多半是 torch / transformers / peft / numpy 版本之间的接口差别；按失败信息修改后重跑。"))


def _inputs(data, add):
    trace = data.get("trace")
    if trace is not None:
        if trace["returncode"] != 0:
            add("block", "轨迹", "轨迹统计与附录 A 不一致（见第 5 节的日志）。", "查 `ho_fl/sat/scenario.py`；这些数值在本机是一致的，差别来自数值库版本或平台。")
        elif trace.get("matches_report") is False:
            add("fix", "轨迹", f"轨迹统计与附录 A 一致，但轨迹哈希与本机记录的不同（Colab：{(trace['trace_sha256'] or '')[:16]}…）。"
                               "多半是不同平台的数学库在末位上的差别（轨迹按 float32 存储）。本次会话内所有运行用的是同一份轨迹，配对不受影响。",
                "把 Colab 的哈希记为参照值（`experiments/sat/e0/expected.json`），以后比对 Colab 对 Colab。")
    rows = data.get("partitions") or []
    compared = [row for row in rows if row.get("matches_report") is not None]
    different = [row for row in compared if not row["matches_report"]]
    failed = [row["name"] for row in rows if row.get("returncode")]
    if failed:
        add("block", "数据", f"{len(failed)} 份划分没有生成：{'、'.join(failed)}。", "见第 5 节的日志；查 `ho_fl/sat/datasets.py` 和数据集的下载。")
    if different:
        add("fix", "数据", f"{len(compared)} 份划分里有 {len(different)} 份的哈希与本机记录的不同：{'、'.join(row['name'] for row in different)}。"
                           "划分用到的 Dirichlet 抽样含浮点运算，平台之间可能在末位上不同。本次会话内各方法用的是同一份划分，配对不受影响。",
            "把 Colab 的哈希记为参照值（`experiments/sat/e0/expected.json`），并在里程碑报告里改用 Colab 的划分统计。")
    elif compared and not failed:
        add("good", "数据", f"{len(compared)} 份划分（3 个种子 × 2 个任务 × 2 种数据版本）的哈希与本机记录的全部相同，轨迹哈希"
                            + {True: "也相同", False: "不同", None: "没有可比的记录"}[(trace or {}).get("matches_report")] + "。")


def _selfcheck(data, add):
    checks = data.get("selfcheck") or {}
    for name, check in checks.items():
        if not check:
            continue
        for row in check["rows"]:
            if not row["passed"]:
                add("block", "正确性", f"正确性检查未通过（{name}，第 {row['item']} 项）：{row['check']}；{row['detail']}。",
                    f"先查 {WHERE.get(int(row['item']), '对应的模块')}。")
    loose = [(name, row) for name, check in checks.items() if check for row in check["rows"]
             if row["passed"] and int(row["item"]) in (4, 5) and row.get("bitwise") is False]
    if loose:
        worst = max((row.get("worst") or 0.0) for _, row in loose)
        replicas = [row for check in checks.values() if check for row in check["rows"] if int(row["item"]) == 5]
        add("note", "正确性", f"有 {len(loose)} 项检查在容差内通过但不是逐位相同（最大差 {small(worst)}，容差 1e-7）。这是参与者本地算出的零阶更新"
                              "与副本重放结果之间的差。" + ("各副本用的都是重放的结果，第 5 项确认它们彼此逐位一致。" if replicas and all(row["passed"] for row in replicas) else ""))
    for name in ("t1", "t2"):
        meta = (checks.get(name) or {}).get("prompt_tokens")
        if meta and meta.get("truncated"):
            add("note", "数据", f"{TASKS[name]} 的提示词里有 {meta['truncated']} 条达到了长度上限 {meta['max_length']}（最长 {meta['longest']}），"
                                "截断去掉的是推文开头，类别选项保留。")


def _estimator(data, add):
    for name, fact in estimator_facts(data).items():
        if fact["nonfinite"]:
            radii = "、".join(format(mu, "g") for mu in fact["nonfinite"])
            add("fix", "零阶估计", f"{TASKS[name]}：零阶标量出现了非有限值（扰动步长 {radii}）。", "查 `hybrid_step` 的扰动前向在 BF16 下是否溢出。")
        span = f"{fact['low']:.2f}" if fact["high"] - fact["low"] < 0.005 else f"{fact['low']:.2f}–{fact['high']:.2f}"
        other = ("" if fact["best_mu"] == fact["mu"] or fact["best_low"] - fact["low"] < 0.1
                 else f"扰动步长取 {fact['best_mu']:g} 时最低相关系数是 {fact['best_low']:.2f}，比现在的设置高。")
        text = f"{TASKS[name]}（{MODELS[name]}）：扰动步长 {fact['mu']:g} 下，训练用的零阶标量与精确方向导数的相关系数在各分界上为 {span}。"
        if fact["low"] >= ESTIMATOR_GOOD:
            add("good", "零阶估计", text + "零阶标量忠实地反映了损失沿随机方向的变化，没有被 BF16 舍入淹没。" + other)
        elif fact["low"] >= ESTIMATOR_POOR:
            add("note", "零阶估计", text + "有噪声，但方向仍然可用（AG News 上 11-22 的测量是 0.78–0.90）。" + other)
        else:
            add("decide", "零阶估计", text + "低于 0.6：在这个设置下零阶标量大半是舍入或非线性造成的噪声，推理芯片卫星传回的信息很少。" + other,
                "扰动步长按实现说明是固定的（不变量 I5）。是否为这个任务另取步长，或把这一点作为结果如实报告，请你决定。")


def _memory(data, add):
    for name, memory in (data.get("memory") or {}).items():
        if not memory or not memory.get("rows"):
            continue
        hybrid = sorted((row for row in memory["rows"] if row["mode"] == "hybrid" and row["peak_allocated_gb"] is not None), key=lambda row: row["cut"])
        if len(hybrid) < 2:
            continue
        peaks = [row["peak_allocated_gb"] for row in hybrid]
        if any(later > earlier * 1.02 for earlier, later in zip(peaks, peaks[1:])):
            add("note", "显存", f"{TASKS[name]}：峰值显存没有随分界上移而单调下降（" + " / ".join(f"分界 {row['cut']} {row['peak_allocated_gb']:.2f} GB" for row in hybrid)
                + "），与“分界越高越省显存”的设想不符。", "看 `ho_fl/sat/calibrate.py` 的测量方式，或零阶部分的扰动前向是否占了更多显存。")
        states = memory["memory_states_gb"]
        idle, imaging = states.get("hybrid_idle"), states.get("hybrid_imaging")
        if idle and imaging and (idle - imaging) / idle < 0.05:
            inference = next((row["peak_allocated_gb"] for row in memory["rows"] if row["mode"] == "inference"), None)
            add("decide", "显存", f"{TASKS[name]}：GPU 卫星空闲与成像两种状态的峰值显存只差 {idle - imaging:.2f} GB（{idle:.2f} 对 {imaging:.2f}"
                                  + (f"；只做前向推理就要 {inference:.2f}" if inference else "") + "）。"
                                  "批大小 16 时显存大部分是骨干本身，“成像占用显存所以分界上移”这一动机在数字上很弱。",
                "可以如实报告，或用更大的批次、更长的输入来标定；这会影响系统模型的说法，请你决定。")


def _runs(data, add):
    for run in data.get("runs") or []:
        name = describe(run)
        if run["status"] in ("SKIPPED", "NOT RUN"):
            continue
        if run["status"] in ("FAILED", "MISSING", "TIMEOUT"):
            add("block" if run["name"] == "t2-m7" else "fix", "运行", f"{name}：没有跑完（{run['status']}），日志末尾见第 5 节。", "按日志里的报错修改后重跑这一步。")
            continue
        if run["status"] == "DIVERGED":
            add("block" if run["name"] == "t2-m7" else "note", "运行", f"{name}：第 {run.get('diverged_round')} 轮出现非有限值，运行按“发散”结束。"
                + ("服务器步长取 1、没有故障时不应发散。" if run["method"] != "m4" else ""), "查发散那一轮各参与者的损失（`ledger.jsonl`）。")
            continue
        if not run["rounds"]:
            continue
        if run["verdict"].get("loss_decreased") is False:
            add("block" if run["name"] == "t2-m7" else "note", "运行",
                f"{name}：训练损失没有下降（{num(run['verdict'].get('loss_first_third'), 4)} → {num(run['verdict'].get('loss_last_third'), 4)}）。",
                "服务器步长没有调过；" + ("纯零阶方法在 2 个仿真小时内不下降不算意外。" if run["method"] == "m4" else "先看验证准确率是否在升，再决定是否要查学习率。"))
        if run["method"] == "m0":
            continue
        odd = []
        if run["late"]:
            odd.append(f"有 {run['late']} 次一阶更新迟到")
        if run["moved_up"]:
            odd.append(f"有 {run['moved_up']} 次参与为赶截止时间上移了分界")
        if run["zo"]["dropped"]:
            odd.append(f"有 {run['zo']['dropped']} 份零阶报告没有被采用")
        if run["replica_groups_max"] > 1 or (run["replica_difference_max"] or 0) > 0:
            odd.append(f"副本分成了 {run['replica_groups_max']} 组（副本差别最大 {small(run['replica_difference_max'])}）")
        if odd:
            add("fix", "运行", f"{name}：" + "，".join(odd) + "。默认链路、没有故障时这些都应当是 0。",
                "查 `ho_fl/sat/policy.py`（到达时间与分界上移）、`ho_fl/sat/network.py`（投递）和 `ho_fl/sat/aggregate.py`（副本组）。")
        low = {kind: value for kind, value in (run.get("battery_min") or {}).items() if value is not None and value < 0.8}
        if low:
            add("note", "运行", f"{name}：电量最低到了 " + "、".join(f"{KINDS[kind]} {100 * value:.0f}%" for kind, value in low.items())
                + "，低于附录 A 的约 85% / 87%。", "对照 `ho_fl/sat/energy.py` 的电池记账；下限是 30%，目前不影响参与。")


def _run_inputs(data, add):
    """The trace and the partition a run built for itself, against the pinned hashes: what the trace and data
    stages used to compare, read from what every run records."""
    other = {"trace": [], "partition": []}
    for run in data.get("runs") or []:
        for key, same in (run.get("inputs_as_recorded") or {}).items():
            if same is False:
                other[key].append(describe(run))
    if other["trace"]:
        add("fix", "轨迹", f"{len(other['trace'])} 次运行用的轨迹哈希与记录的参照值不同（{'、'.join(other['trace'][:3])}{' 等' if len(other['trace']) > 3 else ''}）。"
                           "本次会话内各次运行用的是同一份轨迹，配对不受影响；与以前会话的结果不能直接比较。",
            "先确认陆地掩膜包装上了（见“环境”）；若装上了仍不同，查 `ho_fl/sat/scenario.py` 或数值库版本，并决定是否更新 `experiments/sat/e0/expected.json`。")
    if other["partition"]:
        add("fix", "数据", f"{len(other['partition'])} 次运行用的划分哈希与记录的参照值不同（{'、'.join(other['partition'][:3])}{' 等' if len(other['partition']) > 3 else ''}）。"
                           "本次会话内各方法用的是同一份划分，配对不受影响；与以前会话的结果不能直接比较。",
            "查数据集的修订版本和 `ho_fl/sat/datasets.py`，并决定是否更新 `experiments/sat/e0/expected.json`。")


def _repeats(data, add):
    for repeat in data.get("repeats") or []:
        pair = f"{TASKS.get(repeat['task'], repeat['task'])} · {METHODS.get(repeat['method'], repeat['method'])}"
        if repeat["identical"]:
            add("good", "可复现", f"{pair}：同一配置跑了两次（2 小时的运行和一整天运行的前 {repeat['rounds_compared']} 轮），每轮损失和各次评估完全相同。"
                                  "按这次的结果，卫星引擎在这块 GPU 上可以逐位复现，方法之间的配对差值不含运行间的噪声。")
        else:
            add("decide", "可复现", f"{pair}：同一配置跑两次并不完全相同：前 {repeat['rounds_compared']} 轮里训练损失最大相差 "
                                    f"{small(repeat['max_loss_difference'])}，评估准确率最大相差 {small(100 * repeat['max_accuracy_difference'])} 个点。"
                                    "方法之间的配对差值里会带有这个量级的噪声。",
                "可以接受这个噪声并在统计判据里写明，也可以打开 PyTorch 的确定性算法（会变慢）。差值若只有零点几个点，这个选择会影响结论，请你决定。")


def _cost(data, add):
    facts = cost_facts(data)
    for task, fact in facts.items():
        run, (low, high) = fact["run"], ASSUMED_DAY_RUN_MIN[task]
        text = (f"{TASKS[task]} · 完整方法：每轮 {run['seconds']['round']:.2f} s（本地训练占 {100 * fact['train']:.0f}%，重放与聚合占 {100 * fact['replay']:.0f}%），"
                f"一次评估 {num(run['seconds']['evaluation'], 1)} s，启动 {run['seconds']['startup']:.0f} s；一次 24 小时的运行"
                + (f"实测 {fact['minutes']:.1f} 分钟" if fact["measured"] else f"按此折算约 {num(fact['minutes'], 1)} 分钟") + f"（实现说明估计 {low}–{high} 分钟）。")
        if fact["ratio"]:
            text += (f"按同样的参与构成，实现说明假设每轮 {fact['ratio']['assumed_round_s']:.1f} s，实测是它的 {fact['ratio']['round']:.2f} 倍"
                     f"（只算本地训练是 {fact['ratio']['training']:.2f} 倍）。")
        add("fix" if fact["over"] else "note", "耗时", text + ("比估计高出 30% 以上。" if fact["over"] else ""),
            "按实现说明 §10.4 的削减方案安排后面的实验；能省的地方见下一条。" if fact["over"] else None)
        if fact["replay"] > REPLAY_SHARE:
            add("fix", "耗时", f"{TASKS[task]}：重放与聚合占每轮的 {100 * fact['replay']:.0f}%。现在的重放直接调用 `replay_client`，它会把每一步的中间量复制到 CPU 用于比对。",
                "在 `ho_fl/sat/aggregate.py` 里写一个不保留中间量的重放，并用测试保证与 `replay_client` 逐位相同；长时间的运行开始前做。")
    hours = plan_hours(facts)
    if hours is not None:
        missing = [TASKS[task] for task in ("t1", "t2") if not (facts.get(task) or {}).get("scale")]
        add("note", "耗时", f"按实测比例粗略折算，实现说明里约 {sum(PLAN_HOURS.values()):.0f} 小时的全部实验大约需要 {hours:.0f} GPU 小时"
            + (f"（{'、'.join(missing)} 没有实测，按原估计计入）" if missing else "") + "。这是折算，不是实测。")


def _day_runs(data, add):
    fact = day_facts(data)
    if fact is None:
        return
    end = f"第 {fact['hours']:g} 个仿真小时"
    if fact["saturated_at"] is not None:
        add("decide", "数据集", f"EuroSAT 上理想中心化在服务器步长没有调过的情况下，验证准确率在第 {fact['saturated_at']:g} 个仿真小时就达到 {100 * SATURATION:.0f}%，"
            f"{end}时为 {pct(fact['ideal'])}%。" + (f"完整方法{end}时为 {pct(fact['full'])}%。" if fact["full"] is not None else "")
            + "这正是实现说明关卡 G3 担心的情况：各方法都接近上限时，准确率区分不了方法。",
            "建议现在就定，而不是等 EuroSAT 上的实验全部跑完：换成更难的 RESISC45（`timm/resisc45`），或者 EuroSAT 改以达标时间为主要指标。")
    else:
        add("note", "数据集", f"EuroSAT 上理想中心化{end}时的验证准确率是 {pct(fact['ideal'])}%（服务器步长没有调过，调过之后可能更高），"
                              f"还没有到 {100 * SATURATION:.0f}%。数据集是否太简单，等选完步长再看。")
    if fact["full"] is None:
        return
    reach = lambda value: "到运行结束都没有达到" if value is None else f"在第 {value:g} 小时达到"  # noqa: E731
    add("note", "学习曲线", f"EuroSAT 一整天（单种子、步长未调、验证集）：以理想中心化最终准确率的 95%（{pct(fact['target'])}%）为目标，理想中心化{reach(fact['ideal_reaches'])}，"
                            f"完整方法{reach(fact['full_reaches'])}；理想中心化领先完整方法最多 {100 * fact['widest'][1]:+.1f} 个点（第 {fact['widest'][0]:g} 小时），"
                            f"运行结束时领先 {100 * fact['last_gap']:+.1f} 个点。方法之间的差别主要出现在哪一段时间，决定了达标时间这个指标有没有区分度。")


def _hofl(data, add):
    fact = hofl_facts(data)
    if fact is None or data.get("tiny"):
        return
    text = (f"HO-FL（Qwen2.5-1.5B、AG News）{len(fact['gaps'])} 个种子上，保留零阶比关掉零阶高 {fact['mean']:+.2f} 个点（标准差 {fact['sd']:.2f}，"
            f"各种子：{'、'.join(format(gap, '+.2f') for gap in fact['gaps'])}）；HO-FL 的作者报告的差距是 {AUTHORS_ZO_GAP} 个点。")
    if fact["holds"] is not None:
        size = "幅度远小于作者报告的值" if abs(fact["mean"]) < AUTHORS_ZO_GAP / 2 else "幅度与作者报告的值相近"
        text += "按实现说明 §10.3 的判据（每个种子的差都为正，且均值大于 2 倍标准误），零阶的作用" + (f"成立，{size}。" if fact["holds"] else "不成立：差距在种子间的波动之内。")
    add("note", "HO-FL 对齐", text)


def _session(data, add):
    stages, session = data.get("stages") or {}, data.get("session") or {}
    # Steps of the notebook itself (getting the code, building the environment, the dry run) carry ``gpu: false``.
    idle = {STEPS.get(step["name"], step["name"]): step["minutes"] for step in session.get("steps") or [] if step.get("gpu") is False and step.get("minutes")}
    idle.update({STAGES[name]: stages[name]["wall_s"] / 60 for name in ("tests", "trace", "data") if (stages.get(name) or {}).get("wall_s")})
    total = sum(idle.values())
    if total >= 5 and not data.get("tiny"):
        add("note", "用时", f"不用 GPU 的准备工作占了 {total:.0f} 分钟（" + "、".join(f"{name} {minutes:.1f}" for name, minutes in idle.items())
            + "）。这段时间 GPU 是空着的。")
    late = data.get("skipped_for_budget") or []
    if late:
        add("note", "用时", f"有 {len(late)} 个运行因为预算不够没有开始：{'、'.join(late)}。", "下次只开这些运行对应的开关，或把 `BUDGET_HOURS` 调大。")


# --- the seven questions ------------------------------------------------------------------------------------------------

def answers(data, rows):
    """``(question, answer)`` for the questions of the session, each in a sentence or two; the findings of
    section 3 carry the details. "未运行" where nothing was measured."""
    verdicts = {}
    for row in rows:
        verdicts.setdefault(row[0], []).append(row[2])

    def state(items):
        seen = [verdict for item in items for verdict in verdicts.get(item, ["NOT RUN"])]
        if "FAIL" in seen:
            return "FAIL"
        return "PASS" if set(seen) == {"PASS"} else ("NOT RUN" if set(seen) == {"NOT RUN"} else "PARTLY")

    lead = {"PASS": "", "FAIL": "有问题，见第 3.1 节。", "NOT RUN": "", "PARTLY": "还不完整。"}
    tests, baseline = data.get("tests"), data.get("baseline")
    first = {"PASS": "没有。", "FAIL": lead["FAIL"], "NOT RUN": "", "PARTLY": lead["PARTLY"]}[state(["1"])]
    first += {"PASSED": "三个冒烟配置与上游提交结果一致，重放通过；", "FAILED": "基线回归未通过；", "INCONCLUSIVE": "基线回归无法判定；",
              "INCOMPLETE": "基线回归没有跑完；", None: "基线回归未运行；"}[(baseline or {}).get("status")]
    first += "CPU 测试未运行。" if tests is None else f"CPU 测试 {tests.get('tests', '?')} 项，{tests.get('failures', 0) + tests.get('errors', 0)} 项没有通过。"
    checks = [row for check in (data.get("selfcheck") or {}).values() if check for row in check["rows"]]
    second = {"PASS": "对。", "FAIL": lead["FAIL"], "NOT RUN": "未运行。", "PARTLY": lead["PARTLY"]}[state(["2", "3", "4", "5", "6"])]
    if checks:
        second += f"真实模型上 {sum(row['passed'] for row in checks)} / {len(checks)} 项检查通过；"
    if data.get("trace") is not None:
        second += "轨迹统计与附录 A " + ("一致；" if data["trace"]["returncode"] == 0 else "不一致；")
    compared = [row for row in data.get("partitions") or [] if row.get("matches_report") is not None]
    if compared:
        second += f"{sum(row['matches_report'] for row in compared)} / {len(compared)} 份划分的哈希与本机记录相同。"
    memory = []
    for name, entry in (data.get("memory") or {}).items():
        states = (entry or {}).get("memory_states_gb") or {}
        if states.get("backbone") is not None:
            memory.append(f"{TASKS[name]}（{MODELS[name]}）：强星 {states['backbone']:.1f} GB，GPU 卫星空闲 {states['hybrid_idle']:.1f} GB、成像 "
                          f"{states['hybrid_imaging']:.1f} GB，推理芯片卫星 {states['forward_only']:.1f} GB。")
    facts = cost_facts(data)
    cost = [f"{TASKS[task]}：每轮 {fact['run']['seconds']['round']:.1f} s，一次 24 小时的运行" + ("实测" if fact["measured"] else "折算约") + f" {num(fact['minutes'], 1)} 分钟"
            f"（估计 {ASSUMED_DAY_RUN_MIN[task][0]}–{ASSUMED_DAY_RUN_MIN[task][1]}）" + ("，超出估计 30% 以上" if fact["over"] else "") + "。" for task, fact in facts.items()]
    hours = plan_hours(facts)
    if hours is not None:
        cost.append(f"全部实验折算约 {hours:.0f} GPU 小时（原估计 {sum(PLAN_HOURS.values()):.0f}）。")
    learning = []
    for name in ("t2-m7", "t1-m7"):
        run = find_run(data, name)
        if finished(run) and run["curve"]:
            learning.append(f"{TASKS[run['task']]} 上完整方法 2 个仿真小时内验证准确率 {pct(run['curve'][0]['accuracy'], 1)}% → {pct(run['curve'][-1]['accuracy'], 1)}%，"
                            f"训练损失{'下降' if run['verdict'].get('loss_decreased') else '没有下降'}。")
    estimator = estimator_facts(data)
    if estimator:
        learning.append("零阶标量与精确方向导数的相关系数（1 为完全一致）：" + "，".join(f"{TASKS[name]} {fact['low']:.2f}–{fact['high']:.2f}" for name, fact in estimator.items()) + "。")
    repeats = data.get("repeats") or []
    if repeats:
        learning.append("同一配置重复运行" + ("逐位相同。" if all(repeat["identical"] for repeat in repeats) else "不完全相同，见第 3.2 节。"))
    day = day_facts(data)
    if day is None:
        sixth = "未运行。"
    else:
        sixth = f"理想中心化第 {day['hours']:g} 小时的验证准确率是 {pct(day['ideal'], 1)}%" + (f"，完整方法 {pct(day['full'], 1)}%" if day["full"] is not None else "") + "。"
        sixth += (f"第 {day['saturated_at']:g} 小时就到了 {100 * SATURATION:.0f}%：偏简单，见第 3.2 节。" if day["saturated_at"] is not None
                  else f"步长未调时没有到 {100 * SATURATION:.0f}%，是否太简单等选完步长再看。")
    # The two questions about the checks belong to a session that ran checks.
    result = [("原有的 HO-FL 有没有被改坏？", first), ("卫星部分的代码在真实模型上对不对？", second)] if checked(data) else []
    result += [("每个分界要多少显存？", " ".join(memory) or "未运行。"), ("一次运行要多久，后面的实验要多少 GPU 小时？", " ".join(cost) or "未运行。"),
               ("模型在学吗，零阶估计可信吗？" if checked(data) else "模型在学吗，同一配置重复运行是否相同？", " ".join(learning) or "未运行。"),
               ("EuroSAT 会不会太简单，24 小时的时长合适吗？", sixth)]
    hofl = hofl_facts(data)
    if hofl is not None:   # only when the HO-FL runs of E0-b were part of this session
        result.append(("HO-FL 里零阶到底有多大作用？", f"{len(hofl['gaps'])} 个种子上，保留零阶比关掉高 {hofl['mean']:+.2f} 个点（标准差 {hofl['sd']:.2f}）；"
                                                       f"作者报告 {AUTHORS_ZO_GAP} 个点。"))
    return result


# --- the report ---------------------------------------------------------------------------------------------------------

def next_steps(data, rows, found):
    state, items = gate(rows)
    stages = data.get("stages") or {}
    lines = []
    decisions = sum(finding["level"] == "decide" for finding in found)
    if not checked(data):
        open_steps = [STAGES[name] for name in SWITCHES if (stages.get(name) or {}).get("status") != "ok"]
        if any(finding["level"] == "block" for finding in found):
            lines.append("有必须先修的问题（第 3.1 节）：修改之后只重跑受影响的步骤。")
        elif open_steps:
            lines.append(f"{'、'.join(open_steps)}没有完成（原因见上）。补跑时只打开对应的开关，已完成的步骤不用重跑。")
        else:
            lines.append("测量和训练运行都完成了。这些数字填进里程碑报告 `docs/reports/P-S3.md`，用来判定关卡 G0，并按实测耗时安排后面的方法对比实验。")
        if decisions:
            lines.append(f"请先定第 3.2 节的 {decisions} 件事，它们会改变后面实验的设计。")
        lines.append("把这份报告整段复制回来即可；需要查细节时再给 `session_bundle.zip`（每次运行的日志和 JSON 都在里面）。")
        return lines
    if state == "PASS":
        lines.append("关卡 G0 的七项检查都通过了。接下来不需要 GPU：写另外四种方法（经地面服务器、邻居平均、按节点混合、固定分界的静态版本）、"
                     "保护措施与故障注入、指标汇总。下一次会话在 EuroSAT 上对全部方法做冒烟并选服务器步长。")
        if decisions:
            lines.append(f"开始前请先定第 3.2 节的 {decisions} 件事，它们会改变后面实验的设计。")
    elif state == "FAIL":
        lines.append(f"清单第 {'、'.join(items)} 项未通过：先按第 3.1 节修改，再开一次会话重跑。在此之前不要开始后面的实验。")
    else:
        reasons = [f"{STAGES.get(name, name)}（{why(entry)}）" for name, entry in stages.items() if entry.get("status") == "skipped"]
        lines.append(f"清单第 {'、'.join(items)} 项还没有结果" + (f"：{'；'.join(reasons)}" if reasons else "（对应的步骤没有打开或没有跑完）") + "。补跑之后才能判定关卡。")
    if state != "PASS":
        done = [name for name in ("memory", "pilot") if (stages.get(name) or {}).get("status") == "ok"]
        if done:
            lines.append("重跑时可以关掉已经完成、与要改的代码无关的步骤：" + "、".join(f"`{SWITCHES[name]}`（{STAGES[name]}）" for name in done)
                         + "。")
    lines.append("把这份报告整段复制回来即可；需要查细节时再给 `session_bundle.zip`（每次运行的日志和 JSON 都在里面）。")
    return lines


def render(data):
    lines = []
    add = lines.append
    env, session = data.get("env") or {}, data.get("session") or {}
    rows, found = checklist(data), findings(data)
    state, items = gate(rows)
    with_checks = checked(data)
    add("# RHO-FFT 实验反馈：卫星场景的第一个实验（关卡 G0）\n" if with_checks else "# RHO-FFT 实验反馈：卫星场景的测量与训练运行\n")
    head = [f"生成于 {data['generated_at']}（UTC）"]
    if not data.get("tiny"):
        head += [f"代码 {((env.get('git') or {}).get('head') or '?')[:12]}", f"GPU {env.get('gpu_name') or '?'}", f"torch {env.get('torch') or '?'}"]
    if session.get("elapsed_hours") is not None:
        head.append(f"会话已用 {session['elapsed_hours']:.2f} h（预算 {session.get('budget_hours', '?')} h）")
    add(" · ".join(head) + "。")
    if data.get("tiny"):
        add("\n**这是 CPU 上用随机小模型做的预演：只说明流程能跑通，下面的数字没有意义。**")
    pending = [name for name in data.get("pending") or [] if name in STAGES]
    if pending:
        add(f"\n**这是中期报告：{'、'.join(STAGES[name] for name in pending)}还没有开始，会话最后的完整报告会包含它。**")
    add("\n## 1 结论\n")
    if with_checks:
        add({"PASS": "**关卡 G0：通过。** 七项检查都通过。",
             "FAIL": f"**关卡 G0：未通过。** 清单第 {'、'.join(items)} 项没有通过，见第 3.1 节。",
             "OPEN": f"**关卡 G0：还不能判定。** 清单第 {'、'.join(items)} 项没有结果。"}[state])
    else:
        stages = data.get("stages") or {}
        states = {"ok": "完成", "attention": "有问题", "skipped": "跳过", None: "没有运行"}
        add("**本次会话只做测量和训练运行：**" + "，".join(f"{STAGES[name]}{states.get((stages.get(name) or {}).get('status'), '有问题')}" for name in SWITCHES) + "。")
    counts = {level: sum(finding["level"] == level for finding in found) for level in ("block", "fix", "decide")}
    add(f"\n必须先修的 {counts['block']} 项，建议调整的 {counts['fix']} 项，需要你决定的 {counts['decide']} 项（都在第 3 节）。本次会话要回答的问题：\n")
    try:
        replies = answers(data, rows)
    except Exception as error:
        replies = [("各个问题的答案生成时出错", f"{error!r}；请看第 2、3 节。")]
    for index, (question, answer) in enumerate(replies, 1):
        add(f"{index}. **{question}** {answer}")
    if with_checks:
        add("\n## 2 关卡清单（实现说明 §10.1 的 E0-a）\n")
        add(table(["项", "检查", "结论", "依据"], [[item, what, VERDICTS[verdict], evidence] for item, what, verdict, evidence in rows]))
    else:
        add("\n## 2 本次会话的范围\n")
        add("- 显存标定：Qwen2.5-1.5B 和 ViT-B/16 在每个分界上一次参与的峰值显存。")
        for group, title in (("short", "短程运行"), ("day", "一整天的运行")):
            runs = [run for run in data.get("runs") or [] if run["group"] == group]
            if runs:
                add(f"- {title}（{len(runs)} 次，各 {runs[0]['hours']:g} 个仿真小时）：" + "；".join(f"{TASKS.get(run['task'], run['task'])} · {METHODS.get(run['method'], run['method'])}" for run in runs) + "。")
        add("- 不在会话里的：测试、基线回归、轨迹与数据的核对、真实模型上的正确性检查。它们只确认代码正确，不产生实验数据；"
            "关卡 G0 的清单以里程碑报告 `docs/reports/P-S3.md` 为准。每次运行自己生成轨迹和划分，哈希是否与记录一致见第 4 节。")
    add("\n## 3 发现与建议的调整\n")
    groups = (("3.1 需要改代码或流程的", ("block", "fix")), ("3.2 需要你决定的（会影响实验设计或结论）", ("decide",)), ("3.3 值得知道的观察", ("note",)),
              ("3.4 得到确认的", ("good",)))
    for title, levels in groups:
        add(f"### {title}\n")
        chosen = sorted((finding for finding in found if finding["level"] in levels), key=lambda finding: levels.index(finding["level"]))
        if not chosen:
            add("无。\n")
        for finding in chosen:
            mark = "【必须先修】" if finding["level"] == "block" else ""
            add(f"- {mark}**{finding['topic']}**：{finding['text']}" + (f"\n  - 建议：{finding['action']}" if finding["action"] else ""))
        if chosen:
            add("")
    add("## 4 数据\n")
    number = 0
    for section in (_section_time, _section_checks, _section_inputs, _section_memory, _section_estimator, _section_runs, _section_day, _section_hofl):
        part = []
        try:
            section(data, part.append)
        except Exception as error:   # a table that cannot be drawn is named; the rest of the report stands
            part = [f"### 4.0 （这一小节生成时出错：{error!r}；数据在结果包的 `feedback/feedback_E0.json` 里。）\n"]
        if part:   # a section that has nothing to say about this session leaves no heading; the others are numbered in order
            number += 1
            lines.extend([part[0].replace("### 4.0 ", f"### 4.{number} ", 1)] + part[1:])
    add("## 5 失败与异常（日志末尾）\n")
    failures = data.get("failures") or []
    broken = [(name, entry["error"]) for name, entry in (data.get("stages") or {}).items() if entry.get("error")]
    for name, error in broken:
        add(f"**{STAGES.get(name, name)}：程序自身出错**\n\n```\n{error}\n```\n")
    if not failures and not broken:
        add("无。\n")
    for failure in failures[:4]:
        add(f"**{STAGES.get(failure['stage'], failure['stage'])} {failure['name']}**（`{failure['log']}`）：\n\n```\n{failure['tail']}\n```\n")
    if len(failures) > 4:
        add(f"另有 {len(failures) - 4} 处失败，日志在结果包里：" + "、".join(f"`{failure['log']}`" for failure in failures[4:]) + "。\n")
    add("## 6 没有运行的部分与下一步\n")
    stages = data.get("stages") or {}
    absent = [f"{STAGES[name]}：" + (("尚未开始（中期报告）" if name in pending else "没有打开，或会话在它之前结束") if name not in stages else f"跳过（{why(stages[name])}）")
              for name in STAGES if name not in ("env", "feedback") and (name in SWITCHES or name in pending or name in stages)
              and (name not in stages or stages[name].get("status") == "skipped")]
    absent += [f"{describe(run)}：{budget(run.get('note'))}" for run in data.get("runs") or [] if run["status"] == "SKIPPED"]
    absent += [f"HO-FL {run['name']}：{budget(run.get('note'))}" for run in data.get("t0") or [] if run["status"] == "SKIPPED"]
    add("没有运行的：" + ("无。" if not absent else "\n\n" + "\n".join(f"- {entry}" for entry in absent)))
    add("\n下一步：\n")
    for line in next_steps(data, rows, found):
        add(f"- {line}")
    add("\n## 名称对照\n")
    add("- 方法：" + "；".join(f"`{key}` {value}" for key, value in METHODS.items()) + "。")
    add("- 卫星类别：强星每个轨道面 1 颗，全部层用反向传播；GPU 卫星每面 4 颗，分界以上用反向传播、以下只做前向（零阶）；推理芯片卫星每面 5 颗，全部层只做前向。")
    add("- 分界：从这一层起往上用反向传播（一阶），以下用零阶。Qwen 有 28 层，ViT 有 12 层。")
    add("- 仿真小时：星座里过去的时间，一小时 6 轮；与 GPU 上花的时间无关。")
    return "\n".join(lines) + "\n"


def fallback(data, error):
    """What is written when the report itself cannot be built: the error and the plain state of every stage."""
    lines = ["# RHO-FFT 实验反馈：报告生成出错\n", "报告没有生成出来，下面是出错的回溯和各步骤的原始状态；完整数据在结果包的 `feedback/feedback_E0.json` 里。"
             "请把这一段和上面的整页输出一起带回来。\n", "```", str(error).strip(), "```\n"]
    for name, entry in (data.get("stages") or {}).items():
        lines.append(f"- {name}: {entry.get('status')}，{(entry.get('wall_s') or 0) / 60:.1f} 分钟" + (f"，被 {entry['blocked_by']} 挡住" if entry.get("blocked_by") else ""))
    for key in ("baseline", "tests", "trace"):
        value = data.get(key) or {}
        lines.append(f"- {key}: " + ", ".join(f"{field}={value.get(field)}" for field in ("status", "returncode", "tests", "failures", "errors") if field in value))
    for name, check in (data.get("selfcheck") or {}).items():
        lines.append(f"- selfcheck {name}: {(check or {}).get('status')}")
    for run in (data.get("runs") or []) + (data.get("t0") or []):
        lines.append(f"- run {run.get('name')}: {run.get('status')}")
    return "\n".join(lines) + "\n"


def _section_time(data, add):
    stages, session = data.get("stages") or {}, data.get("session") or {}
    add("### 4.0 用时\n")
    states = {"ok": "完成", "attention": "有问题", "skipped": "跳过"}
    rows = [[STEPS.get(step["name"], step["name"]), states.get(step.get("status"), step.get("status", "")), num(step.get("minutes"), 1), ""]
            for step in session.get("steps") or []]
    rows += [[STAGES[name], states.get(stages[name].get("status"), stages[name].get("status")), num(stages[name].get("wall_s", 0) / 60, 1), why(stages[name])]
             for name in STAGES if name in stages and name != "feedback"]
    add(table(["步骤", "状态", "分钟", "说明"], rows) if rows else "没有记录。")
    add("")


def _section_checks(data, add):
    if not checked(data):
        return
    add("### 4.0 正确性检查\n")
    baseline = data.get("baseline")
    if baseline and baseline.get("configs"):
        add("基线回归（与上游提交比较）：" + "；".join(f"`{name}` {entry['verdict']}，最终参数的最大差 {small(entry.get('max_abs_state_diff'))}"
                                                 for name, entry in sorted(baseline["configs"].items())) + f"；重放 {baseline['replay']['verdict']}。\n")
    tests = data.get("tests")
    if tests and tests.get("failed"):
        add(table(["没有通过的测试", "信息"], [[f"`{entry['id']}`", entry["message"]] for entry in tests["failed"][:12]]))
        add("")
    rows = []
    for name, check in (data.get("selfcheck") or {}).items():
        if not check:
            continue
        for item in sorted({row["item"] for row in check["rows"]}):
            group = [row for row in check["rows"] if row["item"] == item]
            worst = [row["worst"] for row in group if row.get("worst") is not None]
            exact = [row["bitwise"] for row in group if row.get("bitwise") is not None]
            rows.append([name, check.get("model"), check.get("gpu") or check.get("device"), item, f"{sum(row['passed'] for row in group)} / {len(group)}",
                         small(max(worst)) if worst else "", ("全部" if all(exact) else f"{sum(exact)} / {len(exact)}") if exact else ""])
    if rows:
        add(table(["检查对象", "模型", "设备", "清单项", "通过", "最大差", "逐位相同"], rows))
        add("")
    else:
        add("正确性检查没有运行。\n")
    failed = [[name, row["item"], row["check"], row["detail"]] for name, check in (data.get("selfcheck") or {}).items() if check
              for row in check["rows"] if not row["passed"]]
    if failed:
        add(table(["没有通过的检查", "清单项", "内容", "细节"], failed))
        add("")


def _section_inputs(data, add):
    add("### 4.0 轨迹与划分\n")
    if not checked(data):   # no trace or data stage: what every run recorded of the inputs it built for itself
        compared = [(run, run["inputs_as_recorded"]) for run in data.get("runs") or [] if any(value is not None for value in (run.get("inputs_as_recorded") or {}).values())]
        if not compared:
            add("没有可比的运行（没有运行跑到读入轨迹和划分这一步，或这是预演）。\n")
            return
        same = [run for run, flags in compared if all(value is not False for value in flags.values())]
        add(f"{len(compared)} 次运行各自记录了所用轨迹和划分的哈希，其中 {len(same)} 次与记录的参照值（`experiments/sat/e0/expected.json`）相同。")
        other = [[describe(run), f"`{(run['inputs'].get('trace_sha256') or '')[:16]}`", {True: "是", False: "**否**", None: "无记录"}[flags.get("trace")],
                  f"`{(run['inputs'].get('partition_sha256') or '')[:16]}`", {True: "是", False: "**否**", None: "无记录"}[flags.get("partition")]]
                 for run, flags in compared if run not in same]
        if other:
            add("")
            add(table(["运行", "轨迹哈希", "与记录相同", "划分哈希", "与记录相同"], other))
        add("")
        return
    trace = data.get("trace")
    if trace is None:
        add("轨迹：没有生成。")
    else:
        add(f"轨迹哈希 `{trace['trace_sha256']}`，陆地信息来自 {trace['land_source']}；与本机记录的："
            + {True: "相同", False: "**不同**", None: "没有可比的记录"}[trace.get("matches_report")] + "。")
    rows = data.get("partitions") or []
    same = [row for row in rows if row.get("matches_report") is True]
    if same:
        sizes = sorted({(row["task"], row["variant"], row["examples_per_satellite"]) for row in same})
        add(f"\n{len(same)} 份划分的哈希与本机记录相同；每星样本数：" + "，".join(f"{TASKS.get(task, task)} {'常规' if variant == 'base' else 'Z 独有'} {count}" for task, variant, count in sizes) + "。")
    other = [row for row in rows if row.get("matches_report") is not True]
    if other:
        add("")
        add(table(["划分", "每星样本数", "未用样本", "划分哈希", "与本机记录相同"],
                  [[row["name"], row["examples_per_satellite"], row["unused"], f"`{row['partition_sha256']}`", {False: "**否**", None: "无记录"}[row.get("matches_report")]]
                   for row in other]))
    add("")


def _section_memory(data, add):
    add("### 4.0 一次参与的峰值显存（GB）\n")
    shown = False
    for name, memory in (data.get("memory") or {}).items():
        if not memory:
            continue
        shown = True
        by_mode = {mode: {row["cut"]: row for row in memory["rows"] if row["mode"] == mode} for mode in ("inference", "hybrid", "frozen")}
        cuts = sorted(by_mode["hybrid"])
        peak = lambda mode, cut: num(by_mode[mode][cut]["peak_allocated_gb"]) if cut in by_mode[mode] else ""  # noqa: E731
        add(f"{TASKS[name]}（{memory['model']}，批次 {'×'.join(map(str, memory['batch_shape']))}，{memory['device']}）；只做一次前向推理：{peak('inference', 0)}。\n")
        add(table(["分界"] + [str(cut) for cut in cuts],
                  [["分界以下只做前向（零阶）"] + [peak("hybrid", cut) for cut in cuts],
                   ["分界以下冻结（不做零阶）"] + [peak("frozen", cut) for cut in cuts],
                   ["一阶更新的大小（MB）"] + [num(by_mode["hybrid"][cut]["first_order_bytes"] / 1e6, 3) for cut in cuts]]))
        add("")
    if not shown:
        add("没有测量。\n")


def _section_estimator(data, add):
    if not checked(data):
        return
    add("### 4.0 零阶估计的保真度（诊断，不计入关卡）\n")
    shown = False
    for name in ("t2", "t1"):
        rows = ((data.get("selfcheck") or {}).get(name) or {}).get("estimator")
        if not rows:
            continue
        if not shown:
            add("训练用的零阶标量除以扰动步长，对比反向传播算出的精确方向导数。每格是“相关系数 / 斜率”，都是 1 表示完全一致；步长太小会被 BF16 舍入淹没，太大则离开线性区。\n")
        shown = True
        radii = sorted({row["mu"] for row in rows})
        configured = next((row["mu"] for row in rows if row["configured"]), None)
        cell = {(row["cut"], row["mu"]): f"{num(row['correlation'])} / {num(row['slope'])}" for row in rows}
        add(table([f"{TASKS[name]}：分界"] + [f"步长 {mu:g}" + ("（运行采用）" if mu == configured else "") for mu in radii],
                  [[cut] + [cell.get((cut, mu), "") for mu in radii] for cut in sorted({row["cut"] for row in rows})]))
        add("")
    if not shown:
        add("没有测量。\n")


def _section_runs(data, add):
    add("### 4.0 训练运行（种子 42，服务器步长 1，在验证集上评估）\n")
    runs = [run for run in data.get("runs") or [] if run.get("rounds")]
    if not runs:
        add("没有完成的运行。\n")
        return
    rows = []
    for run in runs:
        people = run["participants"]
        skipped = "、".join(f"{REASONS[reason]} {count}" for reason, count in run["skipped"].items() if count) or "无"
        rows.append([describe(run), run["status"], " / ".join(f"{people[kind]:.1f}" for kind in "BHZ"), skipped,
                     f"{num(run['verdict'].get('loss_first_third'), 4)} → {num(run['verdict'].get('loss_last_third'), 4)}",
                     f"{pct(run['curve'][0]['accuracy'], 1)} → {pct(run['curve'][-1]['accuracy'], 1)}" if run["curve"] else "无",
                     num(run["seconds"]["round"]), num(run["seconds"]["evaluation"], 1), num((run.get("wall_s") or 0) / 60, 1)])
    add(table(["运行", "状态", "每轮参与（强星 / GPU 卫星 / 推理芯片卫星）", "受邀但没参加（累计）", "训练损失（前三分之一 → 后三分之一）",
               "验证准确率 %（开始 → 结束）", "每轮秒数", "每次评估秒数", "总分钟"], rows))
    add("")
    seconds = {}
    for run in runs:
        for entry in run.get("participation_seconds") or []:
            total = seconds.setdefault((run["task"], entry["cut"], entry["zo"]), [0, 0.0])
            total[0] += entry["count"]
            total[1] += entry["count"] * entry["mean_s"]
    if seconds:
        add("一次参与（5 步本地训练）在 GPU 上的实际耗时，按分界汇总（与仿真里的名义计算时间无关）：\n")
        add(table(["任务", "分界", "做零阶", "次数", "平均秒数"],
                  [[TASKS.get(task, task), cut, "是" if zo else "否", count, num(total / count, 3)] for (task, cut, zo), (count, total) in sorted(seconds.items())]))
        add("")
    extra = [run for run in runs if run["method"] != "m0"]
    lowest = {kind: min((run["battery_min"][kind] for run in extra if (run.get("battery_min") or {}).get(kind) is not None), default=None) for kind in "HZ"}
    if extra and all(run["late"] == 0 and run["moved_up"] == 0 and run["zo"]["sent"] == run["zo"]["used"] and run["replica_groups_max"] == 1 for run in extra):
        add(f"在轨的各次运行里：没有迟到的一阶更新，没有为赶截止时间上移分界，零阶报告全部被采用，副本始终只有 1 组；电量最低时 GPU 卫星 {pct(lowest['H'], 1)}%、"
            f"推理芯片卫星 {pct(lowest['Z'], 1)}%。\n")
    elif extra:
        add(table(["运行", "迟到的一阶更新", "上移分界的次数", "零阶报告（发出 / 采用）", "副本组数（最多）", "电量最低 %（GPU 卫星 / 推理芯片卫星）"],
                  [[describe(run), run["late"], run["moved_up"], f"{run['zo']['sent']} / {run['zo']['used']}", run["replica_groups_max"],
                    " / ".join(pct((run.get("battery_min") or {}).get(kind), 1) for kind in "HZ")] for run in extra]))
        add("")


def _section_day(data, add):
    add("### 4.0 一整天的运行：验证准确率（%）随仿真时间的变化\n")
    runs = [run for run in data.get("runs") or [] if run["group"] == "day" and run.get("rounds") and run["curve"]]
    if not runs:
        add("没有运行。\n")
        return
    hours = sorted({point["sim_hours"] for run in runs for point in run["curve"]})
    marks = [hour for hour in hours if hour in (0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24) or hour == hours[-1]]
    add(table(["仿真小时"] + [f"{hour:g}" for hour in marks],
              [[METHODS.get(run["method"], run["method"])] + [pct(accuracy_at(run, hour), 1) for hour in marks] for run in runs]))
    add("")
    for run in runs:
        energy, traffic = run.get("energy_wh_per_satellite_day"), run.get("bytes_per_day")
        if run["method"] == "m0" or not energy:
            continue
        add(f"{METHODS[run['method']]}：参与次数 " + "、".join(f"{KINDS[kind]} {count}" for kind, count in (run.get("participations") or {}).items())
            + "；每颗卫星每天的能耗（Wh，计算 / 发送）：" + "、".join(f"{KINDS[kind]} {energy['compute'][kind]:.2f} / {energy['transmit'][kind]:.3f}" for kind in "BHZ")
            + (f"；每天的传输量：射频 {traffic['rf'] / 1e9:.2f} GB，激光 {traffic['laser'] / 1e9:.2f} GB。" if traffic else "。"))
        add("")


def _section_hofl(data, add):
    if not any(run.get("accuracy") is not None or run["status"] == "SKIPPED" for run in data.get("t0") or []):
        return   # the HO-FL runs of E0-b are not part of the session any more; the table appears only when they were run
    add("### 4.0 HO-FL 的零阶开与关（Qwen2.5-1.5B、AG News、Dirichlet 0.5、β = 0.75、160 轮）\n")
    by_seed = hofl_by_seed(data)
    rows = []
    for seed in sorted(by_seed):
        entry = by_seed[seed]
        on, off = entry.get("on"), entry.get("off")
        rows.append([seed, entry["source"], num(on), num(off), "无" if on is None or off is None else f"{on - off:+.2f}"])
    add(table(["种子", "来源", "保留零阶的准确率 %", "关掉零阶", "差（点）"], rows))
    timed = [run for run in data.get("t0") or [] if run.get("wall_s")]
    if timed:
        add("\n各次运行的分钟数：" + "，".join(f"{run['name']} {run['wall_s'] / 60:.1f}（{run['status']}）" for run in timed) + "。")
    add("")
