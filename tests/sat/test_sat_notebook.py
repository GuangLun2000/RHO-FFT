"""The Colab notebook: the file on disk is the one ``notebooks/build_notebook.py`` generates, and its code cells
run in the intended order when every external effect is stubbed (no command is executed, nothing is downloaded,
no Colab). What a step does is tested with the pipeline; this is about the session around it: the numbered steps,
which stages are called, that nothing is limited in time, that a failure does not stop the session, and that the
summary, the report and the bundle are shown and the GPU released in every case."""
import importlib.util
import json
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("build_notebook", REPO / "notebooks" / "build_notebook.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

# The record of the environment, the three steps of the experiment (a measurement and two groups of training runs),
# then the report twice: once for the summary and the figures of Step 4, once more for Step 5, so that the bundle
# holds the figures. The checks of gate G0 are not part of a Colab session: no dry run, tests, regression, self-check.
ALL_STAGES = [["env"], ["study"], ["feedback"], ["feedback"]]


class FakeProcess:
    pid = 0

    def __init__(self, code, lines=()):
        self.returncode, self.stdout = code, iter(lines)

    def wait(self, timeout=None):
        return self.returncode


class Colab:
    """Runs the code cells of the notebook in one namespace, as "Run all" does, and records what they would do.
    ``data`` is what the feedback stage would collect (``feedback_E0.json``), from which Step 4 draws."""

    def __init__(self, tmp_path, monkeypatch, codes=None, data=None):
        self.root, self.codes, self.data = tmp_path / "content", codes or {}, data if data is not None else {"runs": [], "memory": {}}
        self.commands, self.events, self.handlers = [], [], []
        monkeypatch.setattr(subprocess, "Popen", self.popen)
        monkeypatch.setattr(subprocess, "run", lambda command, **options: types.SimpleNamespace(returncode=self.command(command), stdout="pipeline output\n", stderr=""))
        monkeypatch.setattr(time, "sleep", lambda seconds: None)
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
        colab = types.ModuleType("google.colab")
        colab.runtime = types.SimpleNamespace(unassign=lambda: self.events.append("release"))
        monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
        monkeypatch.setitem(sys.modules, "google.colab", colab)
        # What a cell shows in the page. IPython is part of Colab's own Python, not of the environment these tests run
        # in, so it is stood in for. The bundle is a link clicked by the page itself (b"zip" is "emlw" in base64).
        display = types.ModuleType("IPython.display")
        display.HTML = lambda data: ("bundle", "auto" if "a.click();" in data else "link", 'download="session_bundle.zip"' in data and "emlw" in data)
        display.Markdown = lambda text: ("markdown", text)
        display.Image = lambda filename, width=None: ("image", Path(filename).name, Path(filename).exists(), width)
        display.display = self.events.append
        ipython = types.ModuleType("IPython")
        ipython.version_info = (9, 0, 0)   # matplotlib looks at the version of an IPython it finds loaded
        monkeypatch.setitem(sys.modules, "IPython", ipython)
        monkeypatch.setitem(sys.modules, "IPython.display", display)

    def command(self, command):
        shown = command if isinstance(command, str) else " ".join(str(part) for part in command)
        self.commands.append(shown)
        if "e0.py feedback" in shown:   # what the feedback stage of the pipeline leaves behind
            out = self.root / "work" / "sat" / "feedback"
            out.mkdir(parents=True, exist_ok=True)
            (out / "feedback_E0.md").write_text("# 报告正文\n", encoding="utf-8")
            (out / "summary_card.md").write_text("| 项 | 值 |\n", encoding="utf-8")
            (out / "feedback_E0.json").write_text(json.dumps(self.data), encoding="utf-8")
            (out / "feedback_E0_bundle.zip").write_bytes(b"zip")
        return next((code for key, code in self.codes.items() if key in shown), 0)

    def popen(self, command, **options):
        assert "start_new_session" not in options   # a stopped cell stops what it started
        return FakeProcess(self.command(command))

    def run_all(self):
        shell = types.SimpleNamespace(set_custom_exc=lambda kinds, handler: self.handlers.append(handler))
        self.namespace = {"get_ipython": lambda: shell, "__name__": "__main__"}
        for kind, source in builder.CELLS:
            if kind != "code":
                continue
            title = source.splitlines()[0]
            exec(compile(source.replace('"/content/', f'"{self.root}/'), title, "exec"), self.namespace)
            if title.startswith("# Session"):
                self.namespace["REPO_DIR"] = str(REPO)   # stands in for the clone; no git command is executed
        return self.namespace

    def stages(self):
        """The stage lists of the pipeline calls, in order."""
        return [[part for part in shown.split("e0.py ")[1].split(" --root")[0].split() if not part.startswith("--")]
                for shown in self.commands if "e0.py" in shown]

    def shown(self, kind):
        return [event for event in self.events if isinstance(event, tuple) and event[0] == kind]


def builder_stages():
    """The stages a Colab session calls, as the pipeline names them (``NOTEBOOK`` in e0.py)."""
    source = (REPO / "experiments" / "sat" / "e0" / "e0.py").read_text(encoding="utf-8")
    return eval(next(line for line in source.splitlines() if line.startswith("NOTEBOOK = ")).split("=", 1)[1].split("#")[0])


def test_the_notebook_on_disk_is_the_generated_one():
    assert builder.NOTEBOOK.read_text(encoding="utf-8") == builder.text(), "run python notebooks/build_notebook.py"
    notebook = json.loads(builder.NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4 and notebook["metadata"]["accelerator"] == "GPU" and builder.main(["--check"]) == 0
    kinds = [cell["cell_type"] for cell in notebook["cells"]]
    assert kinds[0] == "markdown" and kinds.count("code") == 9
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell {index}", "exec")
            assert cell["outputs"] == [] and cell["execution_count"] is None
    # Numbered steps, each a Markdown heading followed by its code; every code cell starts with a comment saying what it does.
    markdown = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"]
    headings = [text.splitlines()[0] for text in markdown]
    assert headings[0] == "# RHO-FFT — Google Colab"
    assert [heading.split(":")[0] for heading in headings[1:]] == [f"## Step {step}" for step in ("0", "1", "2", "3", "4", "5", "6", "7")]
    firsts = ["".join(cell["source"]).splitlines()[0] for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(first.startswith("# ") for first in firsts) and firsts[-1] == "# Step 7: release the GPU runtime"
    # No forms, no switches and no time budget: what runs is decided by the code the notebook clones.
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert not any(word in code for word in ("#@title", "#@param", "BUDGET", "DEADLINE", "--deadline", "--budget", "RUN_", "Timer", "killpg", "HARD_STOP"))
    # One stage of the pipeline per step of the experiment, none of the checks, and no Colab download call.
    assert builder_stages() == ("study",) and [code.count(f', "{stage}")') for stage in builder_stages()] == [1] and "google.colab import files" not in code
    assert not any(f'"{name}"' in code for name in ("baseline", "tests", "trace", "data", "selfcheck", "memory", "smoke", "pilot", "t0", "all", "--tiny", "--force"))
    assert "不设时间上限" in markdown[0] and "Run All" in markdown[0]
    # What the page says a step runs is what the pipeline runs.
    e0_spec = importlib.util.spec_from_file_location("sat_e0_for_notebook", REPO / "experiments" / "sat" / "e0" / "e0.py")
    e0 = importlib.util.module_from_spec(e0_spec)
    e0_spec.loader.exec_module(e0)
    assert f"## Step 3: 方法的对照运行（EuroSAT，{len(e0.STUDY)} 次，各 {e0.DAY_HOURS:g} 个仿真小时）" in headings and {task for _, task, _, _ in e0.STUDY} == {"t2"}
    # The table of the step has a line for every question the runs answer; the page names no run by a code.
    step = next(text for text in markdown if text.startswith("## Step 3:"))
    assert step.count("\n| ") == 2 + 8 and not any(code_name in step for code_name in ("m7", "m2", "m4", "fo_mean", "zo_zeroed", "zonly"))


def test_a_session_runs_the_steps_in_order_and_ends_with_report_bundle_and_release(tmp_path, monkeypatch, capsys):
    colab = Colab(tmp_path, monkeypatch)
    namespace = colab.run_all()
    assert colab.stages() == ALL_STAGES
    # No command carries a limit of time, nothing is forced and nothing runs on tiny models.
    pipeline = [shown for shown in colab.commands if "e0.py" in shown]
    assert all("--cache-dir" in shown and "--device cuda" in shown for shown in pipeline)
    assert not any(word in shown for shown in colab.commands for word in ("--deadline", "--budget-hours", "--force", "--tiny"))
    assert not any("git" in shown and "cat-file" in shown for shown in colab.commands)   # no reference commits are needed without the regression
    # The page: the summary card, then (nothing to draw here) the report between its marker lines, the bundle, the release.
    assert colab.events == [("markdown", "## 实验摘要\n\n| 项 | 值 |\n"), ("bundle", "auto", True), "release"]
    printed = capsys.readouterr().out
    assert printed.count("反馈报告：从下一行开始复制") == 1 and printed.count("# 报告正文") == 1 and printed.count("反馈报告：复制到上一行为止") == 1
    # Step 4 needs matplotlib, which the notebook's own Python has and the pinned environment of the pipeline need not have.
    drawing = "no finished measurement or run to draw yet" if importlib.util.find_spec("matplotlib") else "the figures could not be drawn"
    assert drawing in printed and "pipeline output" not in printed
    assert "Releasing the GPU runtime in 20 seconds" in printed and "core files present" in printed and "\nTrue\n" not in printed
    assert printed.index("# Step 3 · Method study") < printed.index("# Feedback of this session") < printed.index("All steps done")
    assert list(namespace["STATUS"]) == ["Step 3 · Method study: day-long runs on EuroSAT"] and set(namespace["STATUS"].values()) == {"ok"}
    session = json.loads((colab.root / "work" / "sat" / "session.json").read_text())
    assert [step["name"] for step in session["steps"]] == ["code", "environment"] and all(step["gpu"] is False for step in session["steps"])
    assert "budget_hours" not in session and session["repo_ref"] == "main" and len(colab.handlers) == 1
    # The one-line programs handed to the training interpreter are valid Python.
    for name in ("GPU_PROBE", "GPU_FACTS"):
        compile(namespace[name], name, "exec")


def test_a_failing_stage_does_not_stop_the_session(tmp_path, monkeypatch, capsys):
    colab = Colab(tmp_path, monkeypatch, codes={"e0.py study": 1})
    namespace = colab.run_all()
    # The notebook goes on: the pipeline records which runs failed, and the report says so.
    assert colab.stages() == ALL_STAGES and colab.events[-1] == "release"
    assert all(state.startswith("ATTENTION") for state in namespace["STATUS"].values()) and len(namespace["STATUS"]) == 1
    printed = capsys.readouterr().out
    assert "Step 3 · Method study: day-long runs on EuroSAT: ATTENTION" in printed and "- Step 3 · Method study: day-long runs on EuroSAT: ATTENTION" in printed


def test_an_error_prints_the_report_and_releases_the_gpu(tmp_path, monkeypatch, capsys):
    colab = Colab(tmp_path, monkeypatch, codes={"e0.py env": 1})   # the training interpreter finds no GPU
    with pytest.raises(RuntimeError, match="does not see a CUDA device"):
        colab.run_all()
    assert colab.stages() == [["env"]] and colab.events == []
    # IPython hands the error to the handler registered in the first cell.
    shown = []
    colab.handlers[0](types.SimpleNamespace(showtraceback=lambda info, tb_offset=None: shown.append(info[0])), RuntimeError, RuntimeError("x"), None)
    assert shown == [RuntimeError] and colab.stages()[-1] == ["feedback"]
    assert colab.events == [("bundle", "auto", True), "release"]
    printed = capsys.readouterr().out
    assert "# Feedback of what ran before the error" in printed and "# 报告正文" in printed
    # Before the environment exists there is no pipeline to ask for a report; the GPU is released all the same.
    early = Colab(tmp_path / "early", monkeypatch, codes={"pip install -q uv": 1})
    with pytest.raises(RuntimeError, match="Command failed"):
        early.run_all()
    early.handlers[0](types.SimpleNamespace(showtraceback=lambda info, tb_offset=None: None), RuntimeError, RuntimeError("x"), None)
    assert early.stages() == [] and early.events == ["release"] and "反馈报告没有生成" in capsys.readouterr().out


def test_the_summary_and_the_figures_are_shown_in_the_page(tmp_path, monkeypatch, capsys):
    pytest.importorskip("matplotlib")   # the notebook's own Python has it; the pinned environment of the pipeline does not need it
    memory = {"model": "vit", "batch_shape": [16, 3, 224, 224], "rows": [{"cut": 0, "mode": "inference", "peak_allocated_gb": 0.33}]
              + [{"cut": cut, "mode": "hybrid", "peak_allocated_gb": peak} for cut, peak in ((0, 1.57), (3, 1.25), (6, 0.93), (9, 0.61), (12, 0.38))]}
    colab = Colab(tmp_path, monkeypatch, data={"runs": [], "memory": {"t2": memory}})
    colab.run_all()
    # The card first, then a Chinese heading and the picture for every figure; the picture is a file of the session.
    assert colab.events[0] == ("markdown", "## 实验摘要\n\n| 项 | 值 |\n") and colab.shown("markdown")[1] == ("markdown", "### 一次参与的峰值显存（按分界）")
    assert colab.shown("image") == [("image", "memory.png", True, 760)]
    assert (colab.root / "work" / "sat" / "feedback" / "figures" / "memory.png").stat().st_size > 10_000
    assert colab.events[-2:] == [("bundle", "auto", True), "release"] and "could not be drawn" not in capsys.readouterr().out
    # Figures that cannot be drawn cost nothing else: the report, the bundle and the release follow as always.
    broken = Colab(tmp_path / "broken", monkeypatch, data={"runs": "not a list", "memory": {"t2": {"rows": "neither"}}})
    broken.run_all()
    assert broken.events[-2:] == [("bundle", "auto", True), "release"] and broken.shown("image") == []
    printed = capsys.readouterr().out
    assert "not drawn:" in printed and printed.count("# 报告正文") == 1
