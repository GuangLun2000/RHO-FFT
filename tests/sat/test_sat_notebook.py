"""The Colab notebook: the file on disk is the one ``notebooks/build_notebook.py`` generates, and its code cells
run in the intended order when every external effect is stubbed (no command is executed, nothing is downloaded,
no Colab). What a step does is tested with the pipeline; this is about the session around it: which stages are
called, that a failure does not stop the session, and that the feedback is printed and the GPU released in every
case."""
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

ALL_STAGES = [["env"], ["baseline", "tests", "trace", "data"], ["selfcheck"], ["memory"], ["smoke"], ["pilot"], ["feedback"]]


class FakeProcess:
    pid = 0
    stdout = iter(())

    def __init__(self, code):
        self.returncode = code

    def wait(self, timeout=None):
        return self.returncode


class Colab:
    """Runs the code cells of the notebook in one namespace, as "Run all" does, and records what they would do."""

    def __init__(self, tmp_path, monkeypatch, codes=None, **flags):
        self.root, self.codes, self.flags = tmp_path / "content", codes or {}, flags
        self.commands, self.events, self.handlers = [], [], []
        monkeypatch.setattr(subprocess, "Popen", self.popen)
        monkeypatch.setattr(subprocess, "run", lambda command, **options: types.SimpleNamespace(returncode=self.command(command), stdout="dry run output\n", stderr=""))
        monkeypatch.setattr(time, "sleep", lambda seconds: None)
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
        colab = types.ModuleType("google.colab")
        colab.runtime = types.SimpleNamespace(unassign=lambda: self.events.append("release"))
        monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
        monkeypatch.setitem(sys.modules, "google.colab", colab)
        # The bundle goes to the browser inside the cell's output: a link, clicked by the page itself when asked to.
        # IPython is part of Colab's own Python, not of the environment these tests run in, so it is stood in for too.
        display = types.ModuleType("IPython.display")
        display.HTML = lambda data: types.SimpleNamespace(data=data)
        display.display = lambda html: self.events.append(
            ("bundle", "auto" if "a.click();" in html.data else "link", 'download="session_bundle.zip"' in html.data and "emlw" in html.data))
        monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
        monkeypatch.setitem(sys.modules, "IPython.display", display)

    def command(self, command):
        shown = command if isinstance(command, str) else " ".join(str(part) for part in command)
        self.commands.append(shown)
        if "e0.py feedback" in shown:   # what the feedback stage of the pipeline leaves behind
            out = self.root / "work" / "sat" / "feedback"
            out.mkdir(parents=True, exist_ok=True)
            (out / "feedback_E0.md").write_text("# 报告正文\n", encoding="utf-8")
            (out / "feedback_E0_bundle.zip").write_bytes(b"zip")
        return next((code for key, code in self.codes.items() if key in shown), 0)

    def popen(self, command, **options):
        return FakeProcess(self.command(command))

    def run_all(self):
        shell = types.SimpleNamespace(set_custom_exc=lambda kinds, handler: self.handlers.append(handler))
        self.namespace = {"get_ipython": lambda: shell, "__name__": "__main__"}
        for kind, source in builder.CELLS:
            if kind != "code":
                continue
            title = source.splitlines()[0]
            exec(compile(source.replace('"/content/', f'"{self.root}/'), title, "exec"), self.namespace)
            if title.startswith("#@title Parameters"):
                self.namespace.update(self.flags)
            if title.startswith("#@title A1"):
                self.namespace["REPO_DIR"] = str(REPO)   # stands in for the clone; no git command is executed
        return self.namespace

    def stages(self):
        """The stage lists of the pipeline calls on the real root, in order."""
        calls = [shown.split("e0.py ")[1].split(" --root")[0].split() for shown in self.commands if "e0.py" in shown and "--tiny" not in shown]
        return calls


def test_the_notebook_on_disk_is_the_generated_one():
    assert builder.NOTEBOOK.read_text(encoding="utf-8") == builder.text(), "run python notebooks/build_notebook.py"
    notebook = json.loads(builder.NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4 and notebook["metadata"]["accelerator"] == "GPU" and builder.main(["--check"]) == 0
    kinds = [cell["cell_type"] for cell in notebook["cells"]]
    assert kinds[0] == "markdown" and kinds.count("code") == 11
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell {index}", "exec")
            assert cell["outputs"] == [] and cell["execution_count"] is None
    titles = ["".join(cell["source"]).splitlines()[0] for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(title.startswith("#@title ") for title in titles) and titles[-1].startswith("#@title Feedback")
    # Only the satellite session: five numbered steps, nothing of the HO-FL replication, no Colab download call.
    assert [title.split()[1] for title in titles if title.split()[1].isdigit()] == list("12345")
    text = builder.text()
    assert "e0.py\", \"t0" not in text and '"t0"' not in text and "google.colab import files" not in text
    markdown = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown")
    assert "## 这次要回答的问题" in markdown and "## 反馈：这次实验说明了什么，代码该怎么改" in markdown


def test_a_session_runs_the_stages_in_order_and_ends_with_feedback_and_release(tmp_path, monkeypatch, capsys):
    colab = Colab(tmp_path, monkeypatch)
    namespace = colab.run_all()
    assert colab.stages() == ALL_STAGES
    # The record of the environment and the dry run come before the steps; every step carries the deadline of the session.
    order = [index for index, shown in enumerate(colab.commands) if "e0.py" in shown]
    assert "e0.py env" in colab.commands[order[0]] and "all --tiny --quiet" in colab.commands[order[1]] and "e0.py baseline tests trace data" in colab.commands[order[2]]
    steps = [shown for shown in colab.commands if "e0.py" in shown and "--tiny" not in shown and "e0.py env" not in shown and "e0.py feedback" not in shown]
    assert len(steps) == 5 and all(f"--deadline {namespace['DEADLINE'] - 300}" in shown and "--force" not in shown for shown in steps)
    assert all("--cache-dir" in shown and "--device cuda" in shown for shown in colab.commands if "e0.py" in shown and "--tiny" not in shown)
    # One report between the marker lines, the bundle inside the output (the bytes b"zip" are "emlw" in base64), then the GPU goes back.
    assert colab.events == [("bundle", "auto", True), "release"]
    printed = capsys.readouterr().out
    assert printed.count("反馈报告：从下一行开始复制") == 1 and printed.count("# 报告正文") == 1 and printed.count("反馈报告：复制到上一行为止") == 1
    assert "Releasing the GPU runtime in 15 seconds" in printed and "\nTrue\n" not in printed
    titles = [title for title in namespace["STATUS"] if title[0].isdigit()]
    assert [title[0] for title in titles] == list("12345") and all(namespace["STATUS"][title] == "ok" for title in titles)
    session = json.loads((colab.root / "work" / "sat" / "session.json").read_text())
    assert [step["name"] for step in session["steps"]] == ["code", "environment", "dry_run"] and all(step["gpu"] is False for step in session["steps"])
    assert session["budget_hours"] == 1.5 and session["repo_ref"] == "main" and len(colab.handlers) == 1


def test_a_failing_stage_does_not_stop_the_session(tmp_path, monkeypatch, capsys):
    colab = Colab(tmp_path, monkeypatch, codes={"e0.py selfcheck": 1})
    namespace = colab.run_all()
    # The notebook goes on: the pipeline itself skips what builds on the failed stage, and says so in the report.
    assert colab.stages() == ALL_STAGES and colab.events[-1] == "release"
    status = {title[0]: state for title, state in namespace["STATUS"].items() if title[0].isdigit()}
    assert status["2"].startswith("ATTENTION") and status["3"] == "ok"
    assert "2 · Correctness checks on the real models: ATTENTION" in capsys.readouterr().out


def test_switches_and_the_budget_decide_what_is_called(tmp_path, monkeypatch, capsys):
    colab = Colab(tmp_path, monkeypatch, RUN_MEMORY=False, RUN_DRY_RUN=False, FORCE=True, DOWNLOAD_BUNDLE=False, RELEASE_GPU=False)
    namespace = colab.run_all()
    assert colab.stages() == [["env"], ["baseline", "tests", "trace", "data"], ["selfcheck"], ["smoke"], ["pilot"], ["feedback"]]
    assert not any("--tiny" in shown for shown in colab.commands) and colab.events == []
    assert all("--force" in shown for shown in colab.commands if "e0.py smoke" in shown)
    printed = capsys.readouterr().out
    assert printed.count("反馈报告：从下一行开始复制") == 1 and "RELEASE_GPU is off" in printed and "result bundle: not available" in printed
    assert namespace["STATUS"]["3 · Peak GPU memory of a participation per cut"] == "switched off"
    # A budget that is used up: nothing new starts, the feedback is still written and the GPU released.
    spent = Colab(tmp_path / "spent", monkeypatch, BUDGET_HOURS=0.0, RUN_DRY_RUN=False)
    namespace = spent.run_all()
    assert spent.stages() == [["env"], ["feedback"]] and spent.events[-1] == "release"
    assert {state for title, state in namespace["STATUS"].items() if title[0].isdigit()} == {"skipped (no budget left)"}


def test_an_error_prints_the_feedback_and_releases_the_gpu(tmp_path, monkeypatch, capsys):
    colab = Colab(tmp_path, monkeypatch, codes={"all --tiny": 1})
    with pytest.raises(RuntimeError, match="The dry run failed"):
        colab.run_all()
    assert colab.stages() == [["env"]] and colab.events == [] and "dry run output" in capsys.readouterr().out
    # IPython hands the error to the handler registered in the first cell.
    shown = []
    colab.handlers[0](types.SimpleNamespace(showtraceback=lambda info, tb_offset=None: shown.append(info[0])), RuntimeError, RuntimeError("x"), None)
    assert shown == [RuntimeError] and colab.stages()[-1] == ["feedback"]
    assert colab.events == [("bundle", "auto", True), "release"]
    printed = capsys.readouterr().out
    assert "# Feedback of what ran before the error" in printed and "# 报告正文" in printed
