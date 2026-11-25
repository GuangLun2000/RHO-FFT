#!/usr/bin/env python
"""Source of ``notebooks/RHO-FFT_colab.ipynb``.

The notebook is generated: edit the cells below and run

    python notebooks/build_notebook.py            # rewrites the notebook
    python notebooks/build_notebook.py --check    # exit code 1 if the notebook on disk is not what this file gives

``tests/sat/test_sat_notebook.py`` checks that the notebook on disk is the generated one and executes its code cells
with every external effect stubbed (no GPU, no network, no Colab).

What belongs where: the notebook holds what must exist before the repository does (parameters, getting the code,
building the environment) and the order of the session; everything a step does lives in the pipeline it calls
(``experiments/sat/e0/e0.py``), which also decides whether a step may start and writes the feedback report. Text
for the research lead (Markdown cells, the feedback report) is Chinese; what code prints is English.
"""
import json
import sys
from pathlib import Path

NOTEBOOK = Path(__file__).with_name("RHO-FFT_colab.ipynb")
CELLS = []


def md(text):
    CELLS.append(("markdown", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


# --- overview ------------------------------------------------------------------------------------------------------------

md(r'''
# RHO-FFT · Colab：卫星场景的实验

**用法**：运行时类型选 GPU（默认 G4）→“全部运行”→ 等最后一节“反馈”打印出报告 → 把两条分隔线之间的内容复制回来。跑完或中途出错后，GPU 会自动释放。

估计约 30 分钟（其中取代码、建环境约 5 分钟），预算默认 1.5 小时。代码只从 GitHub 读取（`REPO_REF` 指定分支），不使用 Google Drive。

## 这次跑什么

只有测量和训练运行，每一步都产生实验数据。用时是估计，实测值写在反馈报告里。

| 步骤 | 得到什么 | 结果用来决定什么 | 估计 |
|---|---|---|---|
| 1 显存 | Qwen2.5-1.5B 和 ViT-B/16 在每个分界上一次参与的峰值显存 | 三类卫星的分界表是否成立 | 3 分钟 |
| 2 短程运行 | 已实现的四种方法各跑 2 个仿真小时：能否跑通、损失是否下降、每轮和每次评估的实际耗时 | 后面全部实验的 GPU 预算 | 6 分钟 |
| 3 一整天的运行 | EuroSAT 上完整方法与理想中心化 24 个仿真小时的学习曲线 | EuroSAT 是否太简单，24 小时的时长是否合适 | 12 分钟 |

页面上实时打印进度：每一步开始时说明要做什么，训练运行每轮一行（带已用时间和预计剩余时间），任何命令连续 30 秒没有输出时会补一行“仍在运行”。

检查步骤（预演、测试、基线回归、轨迹与数据的核对、真实模型上的正确性检查）不在这个 notebook 里：它们只确认代码正确，不产生实验数据。流水线保留着这些阶段；代码有大的改动、需要在 GPU 上复核时，在第 1 步之前加一格 `sat("tests", "selfcheck")` 即可。
''')

code(r'''
#@title Parameters { display-mode: "form" }
# The code is read from GitHub only (a public repository: no token, no Colab secrets).
REPO_URL = "https://github.com/GuangLun2000/RHO-FFT.git"  #@param {type:"string"}
REPO_REF = "main"  #@param {type:"string"}

# Wall-clock budget of this notebook, in hours: no new run starts after it, and a command that is still running
# half an hour later is stopped so that the feedback can be written and the GPU released.
BUDGET_HOURS = 1.5  #@param {type:"number"}

# "venv": Python 3.11 with the pinned dependencies of pyproject.toml (recommended).
# "system": Colab's own Python and torch; only a fallback, versions then differ from the pins.
ENV_MODE = "venv"  #@param ["venv", "system"]
# PyTorch installed only if the pinned 2.6.0 (CUDA 12.4) cannot use the GPU. The G4 runtime is an RTX PRO 6000 of
# the Blackwell generation, which needs a CUDA 12.8 build (PyTorch 2.7 or later).
TORCH_FALLBACK = "2.7.1"  #@param {type:"string"}

# Steps, in the order they run. Switch off a step whose results are already in hand.
RUN_MEMORY = True  #@param {type:"boolean"}
RUN_SHORT_RUNS = True  #@param {type:"boolean"}
RUN_DAY_RUNS = True  #@param {type:"boolean"}
# Put the result bundle into the page (a link, and an automatic download at the end) before the runtime is released.
DOWNLOAD_BUNDLE = True  #@param {type:"boolean"}
# Release the GPU runtime after the last cell, and after an error in any cell.
RELEASE_GPU = True  #@param {type:"boolean"}
''')

# --- A: preparation ------------------------------------------------------------------------------------------------------

md(r'''
## A. 准备：会话、代码、环境

这三步不用 GPU。A1 定义后面用到的函数，并登记“出错即释放”：之后任何一格报错，都会先打印已有结果的反馈报告、下载结果包，再释放 GPU。手动中断（停止按钮）不会触发释放。
''')

code(r'''
#@title A1 · Session, helpers and GPU
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

SESSION_START = time.time()
DEADLINE = SESSION_START + 3600 * BUDGET_HOURS   # no new run starts after this
HARD_STOP = DEADLINE + 1800                      # a command still running then is taken to hang and is stopped
REPO_DIR = "/content/RHO-FFT"
WORK = "/content/work/sat"           # results of this session (local disk, fast)
BUNDLE = "/content/work/session_bundle.zip"
CACHE = "/content/cache"             # models, datasets, token caches, traces and partitions
VENV = "/content/venv"
STATUS = {}                          # one line per step, printed in the last cell
# What the notebook itself did before the pipeline starts; the feedback report lists it with the stages.
SESSION = {"started_at": SESSION_START, "budget_hours": BUDGET_HOURS, "repo_ref": REPO_REF, "steps": []}


def sh(command, cwd=None, env=None, check=True):
    # Run a command and stream its output into the cell. A command still running at the hard stop is ended together
    # with everything it started, so that the notebook can go on to the feedback and release the GPU.
    shown = command if isinstance(command, str) else " ".join(str(part) for part in command)
    print(f"$ {shown}", flush=True)
    process = subprocess.Popen(command, shell=isinstance(command, str), cwd=cwd, env={**(env or os.environ), "PYTHONUTF8": "1"},
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                               bufsize=1, start_new_session=True)

    def end():
        print(f"\nSTOPPED, still running half an hour after the budget of {BUDGET_HOURS} h: {shown}", flush=True)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass

    timer = threading.Timer(max(180.0, HARD_STOP - time.time()), end)
    timer.daemon = True
    timer.start()
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
        code = process.wait()
    finally:
        timer.cancel()
    if check and code != 0:
        raise RuntimeError(f"Command failed with exit code {code}: {shown}")
    return code


def hours_left():
    return (DEADLINE - time.time()) / 3600


def stage(title, text=""):
    # A Markdown heading in the output: a copy of the page into a .md file gets one section per step.
    print(f"\n# {title}\n" + (f"\n{text}\n" if text else ""), flush=True)


def note_step(name, started, ok):
    # A step of the notebook itself (code, environment): no GPU work, but the GPU is held meanwhile.
    SESSION["steps"].append({"name": name, "minutes": (time.time() - started) / 60, "status": "ok" if ok else "attention", "gpu": False})
    Path(WORK).mkdir(parents=True, exist_ok=True)
    (Path(WORK) / "session.json").write_text(json.dumps(SESSION, indent=1))


def release_gpu(delay=10):
    # Hand the GPU runtime back. Everything on the VM's local disk is lost, so this runs only after the results
    # have been shown and downloaded. The page keeps its output.
    if not RELEASE_GPU:
        print("RELEASE_GPU is off; the runtime stays connected.")
        return
    print(f"Releasing the GPU runtime in {delay} seconds ...", flush=True)
    time.sleep(delay)
    from google.colab import runtime
    runtime.unassign()


def stop(message):
    # A failed precondition stops "Run all"; the error handler below then releases the GPU.
    raise RuntimeError(message)


def download_results(auto=True):
    # Hand the result bundle (logs and JSON files of every run, the feedback report) to the browser. Returns whether
    # it was handed over.
    #
    # google.colab.files.download cannot do this here. It only shows a script; the browser then asks the kernel for
    # the file over a comm, and the kernel answers such a request only once it is idle. A cell that goes on to
    # sleep and release the runtime never is, however long it sleeps, so that download never starts. Instead the
    # bytes travel inside this cell's output: the browser builds the file from what it already has, without asking
    # the runtime for anything, and the link keeps working after the runtime is gone. ``auto`` also starts the download.
    source = Path(WORK) / "feedback" / "feedback_E0_bundle.zip"
    if not (DOWNLOAD_BUNDLE and source.exists()):
        return False
    megabytes = source.stat().st_size / 1e6
    if megabytes > 20:
        print(f"The result bundle has {megabytes:.0f} MB, too much to pass through the page; the report above is the record.")
        return False
    try:
        import base64
        import uuid
        from IPython.display import HTML, display
        name, link = Path(BUNDLE).name, "bundle-" + uuid.uuid4().hex
        display(HTML(
            f'<a id="{link}" download="{name}" style="font-size:15px">结果包 {name}（{megabytes:.1f} MB）：点这里下载</a>'
            f'<script>(function() {{ const bytes = Uint8Array.from(atob("{base64.b64encode(source.read_bytes()).decode("ascii")}"), c => c.charCodeAt(0));'
            f' const a = document.getElementById("{link}"); a.href = URL.createObjectURL(new Blob([bytes], {{type: "application/zip"}}));'
            f' {"a.click();" if auto else ""} }})();</script>'))
        return True
    except Exception as error:
        print("the result bundle could not be handed to the browser:", repr(error))
        return False


def print_feedback(title):
    # Rebuild the feedback report from whatever exists and print it between two marker lines: that is what goes back.
    stage(title)
    if "sat" in globals():
        sat("feedback", extra=["--quiet"])
    path = Path(WORK) / "feedback" / "feedback_E0.md"
    print("\n" + "=" * 24 + " 反馈报告：从下一行开始复制 " + "=" * 24)
    print(path.read_text(encoding="utf-8") if path.exists() else "反馈报告没有生成；请把上面的整页输出带回来。")
    print("=" * 24 + " 反馈报告：复制到上一行为止 " + "=" * 24, flush=True)


def _release_on_error(shell, etype, value, traceback, tb_offset=None):
    # Any error in any later cell: show it, print the feedback of what exists, save it, then release the GPU
    # instead of leaving it idle.
    shell.showtraceback((etype, value, traceback), tb_offset=tb_offset)
    try:
        print_feedback("Feedback of what ran before the error")
    except Exception as error:
        print("the feedback report could not be built:", repr(error))
    download_results()
    release_gpu(15)


# Manual interruption (KeyboardInterrupt) is not an Exception and does not release the runtime.
get_ipython().set_custom_exc((Exception,), _release_on_error)

sh("nvidia-smi", check=False)
''')

code(r'''
#@title A2 · Get the code
# The repository is public: an anonymous clone, no credentials and no access to Colab secrets.
started = time.time()
git_env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
if not os.path.isdir(f"{REPO_DIR}/.git"):
    sh(["git", "clone", "--quiet", REPO_URL, REPO_DIR], env=git_env)
else:
    sh(["git", "-C", REPO_DIR, "fetch", "--quiet", "origin"], env=git_env)
remote_ref = f"origin/{REPO_REF}"
has_remote_ref = subprocess.run(["git", "-C", REPO_DIR, "rev-parse", "--verify", "--quiet", remote_ref],
                                capture_output=True).returncode == 0
sh(["git", "-C", REPO_DIR, "checkout", "--quiet", "--detach", remote_ref if has_remote_ref else REPO_REF])
sh(["git", "-C", REPO_DIR, "log", "--oneline", "-5"])
note_step("code", started, True)
''')

md(r'''
### A3. 环境

建一个独立的 Python 3.11 虚拟环境，按 `pyproject.toml` 的固定版本安装依赖（PyTorch 2.6.0 / CUDA 12.4），外加卫星部分的可选依赖（陆地掩膜、Pillow）。所有运行都是这个环境里的子进程；仓库本身不安装，运行时通过 `PYTHONPATH` 指向源码树。

G4 运行时是 Blackwell 架构的 RTX PRO 6000，固定版本的 PyTorch 2.6.0 不支持它。环境建好后会先探测 GPU：固定版本用不了时，自动换成参数 `TORCH_FALLBACK` 指定的版本（CUDA 12.8 构建），实际使用的版本会记在每个结果里。
''')

code(r'''
#@title A3 · Environment
started = time.time()


def clean_environment():
    # Colab exports constraint files for its own package set (PIP_CONSTRAINT, UV_CONSTRAINT, ...). They must not
    # apply to the separate environment built here, which pins different versions.
    return {key: value for key, value in os.environ.items() if not key.startswith(("UV_", "PIP_"))}


if ENV_MODE == "venv":
    uv = f"{sys.executable} -m uv"
    sh(f"{sys.executable} -m pip install -q uv")
    if not os.path.exists(f"{VENV}/bin/python"):
        if sh(f"{uv} venv --python 3.11 {VENV}", env=clean_environment(), check=False) != 0:
            print("Python 3.11 could not be provisioned; building the environment on the runtime's own Python instead")
            shutil.rmtree(VENV, ignore_errors=True)
            sh(f"{uv} venv --python {sys.executable} {VENV}", env=clean_environment())
    # The optional group "sat" adds the land mask (where a satellite is imaging) and Pillow (EuroSAT is stored as JPEG).
    install = f"{uv} pip install --python {VENV}/bin/python -r pyproject.toml --extra dev --extra sat"
    if sh(install + " --torch-backend cu124", cwd=REPO_DIR, env=clean_environment(), check=False) != 0:
        print("retrying without --torch-backend (the default Linux wheel of torch 2.6.0 is the CUDA 12.4 build)")
        sh(install, cwd=REPO_DIR, env=clean_environment())
    PY = f"{VENV}/bin/python"
else:
    PY = sys.executable
    sh(f"{PY} -m pip install -q transformers==4.56.2 peft==0.17.1 datasets==3.6.0 accelerate==1.10.1 "
       "safetensors==0.6.2 sentencepiece==0.2.1 global-land-mask==1.0.0 pillow pytest", env=clean_environment())

sh([PY, "-c", "import torch, transformers, peft, datasets; "
              "print('torch', torch.__version__, '| cuda runtime', torch.version.cuda, '| cuda available', torch.cuda.is_available()); "
              "print('transformers', transformers.__version__, '| peft', peft.__version__, '| datasets', datasets.__version__)"])

# Probe the GPU with the interpreter that will train. The pinned PyTorch has no kernels for Blackwell GPUs (the G4
# runtime); if it cannot use this GPU, the fallback build for CUDA 12.8 is installed and probed again. The versions
# actually used are recorded with every result.
GPU_PROBE = ("import torch; x = torch.randn(256, 256, device='cuda', dtype=torch.bfloat16); "
             "assert torch.isfinite((x @ x).float()).all(); "
             "print('GPU usable:', torch.cuda.get_device_name(0), '| torch', torch.__version__)")


def gpu_usable():
    return sh([PY, "-c", GPU_PROBE], check=False) == 0


if shutil.which("nvidia-smi") is None:
    stop("No GPU in this runtime. Choose a GPU runtime type and run again.")
if not gpu_usable():
    if ENV_MODE != "venv":
        stop("The runtime's own PyTorch cannot use this GPU. Send the output above back.")
    print(f"\nThe pinned PyTorch cannot use this GPU; installing torch {TORCH_FALLBACK} built for CUDA 12.8 instead.")
    fallback = f"{uv} pip install --python {VENV}/bin/python torch=={TORCH_FALLBACK}"
    if sh(fallback + " --torch-backend cu128", env=clean_environment(), check=False) != 0:
        sh(fallback + " --index-url https://download.pytorch.org/whl/cu128", env=clean_environment())
    if not gpu_usable():
        stop("PyTorch cannot use this GPU with either build. Send the output above back.")
    STATUS["environment note"] = f"torch {TORCH_FALLBACK} (CUDA 12.8) in place of the pinned 2.6.0, which does not support this GPU"


def sat(*stages, root=WORK, extra=()):
    # Stages of experiments/sat/e0/e0.py, the pipeline of this session; 0 = every stage ok.
    command = [PY, "experiments/sat/e0/e0.py", *stages, "--root", root, "--cache-dir", CACHE, "--device", "cuda", *extra]
    return sh(command, cwd=REPO_DIR, check=False)


def run_stage(title, stages, enabled):
    # One step of the session. The pipeline decides which runs still fit into the budget and skips a day-long run
    # whose two-hour run did not pass; nothing here raises, so the notebook always reaches the feedback and the
    # release of the GPU.
    if not enabled:
        print(f"{title}: switched off")
        STATUS[title] = "switched off"
        return
    if hours_left() <= 0:
        print(f"{title}: skipped, the budget of {BUDGET_HOURS} h is used up")
        STATUS[title] = "skipped (no budget left)"
        return
    stage(title)
    started = time.time()
    code = sat(*stages, extra=["--deadline", str(DEADLINE - 300)])
    STATUS[title] = "ok" if code == 0 else "ATTENTION: failed, skipped or incomplete (the feedback report says which and why)"
    print(f"\n{title}: {STATUS[title]} | {(time.time() - started) / 60:.1f} min | {hours_left():.2f} h of budget left\n", flush=True)


note_step("environment", started, True)
# The record of the environment (GPU, versions, commit) that every later result refers to.
if sat("env") != 0:
    stop("The training interpreter does not see a CUDA device. Send the output above back.")
''')

# --- the steps -----------------------------------------------------------------------------------------------------------

md(r'''
## 1. 显存：每个分界上一次参与的峰值

对每个分界做一次完整参与（5 步本地训练，批大小 16），记录峰值显存：Qwen 在分界 0、8、16、24、28，ViT 在分界 0、3、6、9、12。页面每测完一项打印一行。
''')

code(r'''
#@title 1 · Peak GPU memory per cut
run_stage("1 · Peak GPU memory of a participation per cut", ["memory"], RUN_MEMORY)
''')

md(r'''
## 2. 短程运行（5 次，各 2 个仿真小时）

一个仿真小时是 6 轮，每轮约 18 颗卫星受邀。完整方法在 EuroSAT、HumAID 上各一次；理想中心化、只让强星训练、纯零阶泛洪在 EuroSAT 上各一次。第一次运行会先生成轨迹和数据划分，之后的运行直接复用。页面每轮打印一行。

服务器步长都取 1，没有调过，所以这里的准确率不能用来比较方法；评估用验证集，不碰测试集。
''')

code(r'''
#@title 2 · Two simulated hours of every implemented method
run_stage("2 · Short runs: two simulated hours each", ["smoke"], RUN_SHORT_RUNS)
''')

md(r'''
## 3. 一整天的运行（EuroSAT，2 次）

“一整天”是仿真里的 24 小时（144 轮、25 次评估），不是 GPU 上的时间。完整方法和理想中心化各一次；开始前用第 2 步的实测耗时估算要多久，放不进剩余预算就跳过。页面每轮打印一行。

它不是正式结果（步长没调、单种子、验证集），只为提前知道三件事：EuroSAT 会不会太简单、一次完整运行要多久、同一配置跑两次是否逐位相同。
''')

code(r'''
#@title 3 · A whole simulated day on EuroSAT
run_stage("3 · Day-long runs on EuroSAT: the full method and the centralised reference", ["pilot"], RUN_DAY_RUNS)
''')

# --- feedback ------------------------------------------------------------------------------------------------------------

md(r'''
## 反馈：这次实验说明了什么，代码该怎么改

这一节把整次会话汇总成一份报告，打印在两条分隔线之间。**把这段复制回来即可。** 报告依次是：结论和各个问题的答案；本次会话的范围；发现与建议的调整（必须先修的、建议修改的、需要你决定的）；数据表；失败步骤的日志末尾；没有运行的部分与下一步。

之后浏览器下载结果包 `session_bundle.zip`（每次运行的日志和 JSON，备查），再释放 GPU。结果包的内容直接放在这一格的输出里：如果没有自动下载，点输出里的“点这里下载”即可，GPU 释放之后也能点。
''')

code(r'''
#@title Feedback · report, download, then release the GPU
print_feedback(f"Feedback of this session ({(time.time() - SESSION_START) / 3600:.2f} h of {BUDGET_HOURS} h used)")
if not download_results():
    print("\nresult bundle: not available; the report above is the record of this session")
release_gpu(15)
''')


# --- writing the notebook ------------------------------------------------------------------------------------------------

def lines(source):
    parts = source.split("\n")
    return [part + "\n" for part in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def build():
    cells = []
    for kind, source in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": lines(source)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {"cells": cells,
            "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "G4"},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def text():
    return json.dumps(build(), indent=1, ensure_ascii=False) + "\n"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    for index, (kind, source) in enumerate(CELLS):
        if kind == "code":
            compile(source, f"cell {index}", "exec")
    if "--check" in argv:
        same = NOTEBOOK.exists() and NOTEBOOK.read_text(encoding="utf-8") == text()
        print("the notebook is up to date" if same else f"{NOTEBOOK} differs from what {Path(__file__).name} gives; run it without --check")
        return 0 if same else 1
    NOTEBOOK.write_text(text(), encoding="utf-8")
    print(f"{NOTEBOOK}: {len(CELLS)} cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
