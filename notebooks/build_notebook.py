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
# RHO-FFT · 第 3 次会话（Colab）：卫星场景的第一个实验（关卡 G0）

**用法**：运行时类型选 GPU（默认 G4）→“全部运行”→ 等最后一节“反馈”打印出报告 → 把报告整段复制回来。跑完或中途出错后，GPU 会自动释放。

预计 1.4 小时左右，预算默认 2.5 小时。最后约 40 分钟（第 7 步）是与卫星代码无关的 HO-FL 对齐，想先快速走一遍时可以用开关 `RUN_HOFL_SEEDS` 关掉。代码只从 GitHub 读取（`REPO_REF` 默认是 `sat` 分支，**运行前请确认它已经推送**），不使用 Google Drive。

## 这次会话要回答的七个问题

每一步只为回答一个问题而存在。便宜、又可能推翻后面一切的检查排在前面；前面的检查没有通过时，依赖它的步骤自动跳过，不白占 GPU。下表的用时是估计，实测值会写在反馈报告里。

| # | 问题 | 靠哪一步回答 | 答案决定什么 | 预计 |
|---|---|---|---|---|
| 1 | 原有的 HO-FL 有没有被改坏？ | 第 1 步基线回归；第 2 步的测试 | 不通过：后面的 GPU 步骤全部跳过，先修 | 3 + 4 分钟 |
| 2 | 卫星部分的代码在真实模型上对不对？ | 第 2 步的轨迹与数据；第 3 步正确性检查 | 不通过：训练运行跳过，先修 | 2 + 6 分钟 |
| 3 | 每个分界要多少显存？ | 第 4 步显存标定 | 系统模型里 GPU 卫星“空闲 / 成像”两种状态的数值 | 3 分钟 |
| 4 | 一次运行要多久，后面的实验要多少 GPU 小时？ | 第 5、6 步的计时 | 后面实验的预算；要不要先优化重放 | 不另占时间 |
| 5 | 模型在学吗，零阶估计可信吗？ | 第 5 步短程运行；第 3 步里的零阶诊断 | 学习率、扰动步长有没有问题 | 6 分钟 |
| 6 | EuroSAT 会不会太简单，24 小时的时长合适吗？ | 第 6 步一整天的运行（2 次） | 要不要现在就换数据集或换主要指标，而不是等 EuroSAT 上的十几个 GPU 小时跑完 | 12 分钟 |
| 7 | HO-FL 里零阶到底有多大作用？ | 第 7 步零阶开 / 关，种子 43、44 | 论文里对零阶的说法；不影响卫星代码 | 40 分钟 |

## 反馈怎么循环

1. 最后一节“反馈”把整次会话汇总成一份报告：关卡是否通过、七个问题的答案、需要改的代码、需要你决定的事、数据表、失败的日志、下一步。
2. 把两条分隔线之间的报告复制回来；需要查细节时，再给浏览器下载的 `session3_bundle.zip`。
3. 我据此修改代码和这个 notebook，推送之后再跑下一次。
4. 第 7 步开始之前会先打印一份中期报告（卫星部分已经全部在里面），可以不等那 40 分钟，先复制回来。
''')

code(r'''
#@title Parameters { display-mode: "form" }
# The code is read from GitHub only (a public repository: no token, no Colab secrets).
REPO_URL = "https://github.com/GuangLun2000/RHO-FFT.git"  #@param {type:"string"}
# The satellite work lives on the branch "sat"; it must be on GitHub before this notebook can read it.
REPO_REF = "sat"  #@param {type:"string"}

# Wall-clock budget of this notebook, in hours: no new run starts after it, and a command that is still running
# half an hour later is stopped so that the feedback can be written and the GPU released.
BUDGET_HOURS = 2.5  #@param {type:"number"}

# "venv": Python 3.11 with the pinned dependencies of pyproject.toml (recommended).
# "system": Colab's own Python and torch; only a fallback, versions then differ from the pins.
ENV_MODE = "venv"  #@param ["venv", "system"]
# PyTorch installed only if the pinned 2.6.0 (CUDA 12.4) cannot use the GPU. The G4 runtime is an RTX PRO 6000 of
# the Blackwell generation, which needs a CUDA 12.8 build (PyTorch 2.7 or later).
TORCH_FALLBACK = "2.7.1"  #@param {type:"string"}

# Steps, in the order they run. A step whose prerequisites did not pass is skipped by the pipeline itself.
RUN_DRY_RUN = True  #@param {type:"boolean"}
RUN_BASELINE = True  #@param {type:"boolean"}
RUN_SETUP = True  #@param {type:"boolean"}
RUN_SELFCHECK = True  #@param {type:"boolean"}
RUN_MEMORY = True  #@param {type:"boolean"}
RUN_SHORT_RUNS = True  #@param {type:"boolean"}
RUN_DAY_RUNS = True  #@param {type:"boolean"}
RUN_HOFL_SEEDS = True  #@param {type:"boolean"}
# Run a step even when a step it builds on did not pass (not recommended: its results rest on unverified code).
FORCE = False  #@param {type:"boolean"}
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
TINY = "/content/work/tiny"          # results of the CPU dry run, kept apart from the real ones
BUNDLE = "/content/work/session3_bundle.zip"
CACHE = "/content/cache"             # models, datasets, token caches, traces and partitions
VENV = "/content/venv"
STATUS = {}                          # one line per step, printed in the last cell
# What the notebook itself did before the pipeline starts; the feedback report lists it with the stages.
SESSION = {"number": 3, "started_at": SESSION_START, "budget_hours": BUDGET_HOURS, "repo_ref": REPO_REF, "steps": []}


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
    # A step of the notebook itself (code, environment, dry run): no GPU work, but the GPU is held meanwhile.
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


def print_feedback(title, pending=""):
    # Rebuild the feedback report from whatever exists and print it between two marker lines: that is what goes back.
    # ``pending`` names the stages still to come, so that an interim report does not list them as missing.
    stage(title)
    if "sat" in globals():
        sat("feedback", extra=["--quiet", *(["--pending", pending] if pending else [])])
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
# The repository is public: an anonymous clone, no credentials and no access to Colab secrets. The baseline
# regression reads two reference commits with "git archive", so the clone has the full history.
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

revisions = json.loads(Path(f"{REPO_DIR}/experiments/p0/revisions.json").read_text())
for name, sha in revisions.items():
    if subprocess.run(["git", "-C", REPO_DIR, "cat-file", "-e", sha + "^{commit}"], capture_output=True).returncode != 0:
        raise RuntimeError(f"Reference revision {name} ({sha}) is missing from this clone; a full-history clone is required.")
print("reference revisions present:", revisions)
note_step("code", started, True)
''')

md(r'''
### A3. 环境

建一个独立的 Python 3.11 虚拟环境，按 `pyproject.toml` 的固定版本安装依赖（PyTorch 2.6.0 / CUDA 12.4），外加卫星部分的可选依赖（陆地掩膜、Pillow）。所有运行都是这个环境里的子进程；仓库本身不安装，运行时通过 `PYTHONPATH` 指向源码树。

G4 运行时是 Blackwell 架构的 RTX PRO 6000，固定版本的 PyTorch 2.6.0 不支持它。环境建好后会先探测 GPU：固定版本用不了时，自动换成参数 `TORCH_FALLBACK` 指定的版本（CUDA 12.8 构建），实际使用的版本会记在每个结果里。基线回归在同一环境里进行，所以“有没有改坏 HO-FL”的结论不受影响。
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
# actually used are recorded with every result, and the baseline regression runs in the same environment.
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


def run_stage(title, stages, enabled, text=""):
    # One step of the session. The pipeline decides whether its stages may start (a stage is skipped when one it
    # builds on did not pass) and which runs still fit into the budget; nothing here raises, so the notebook always
    # reaches the feedback and the release of the GPU.
    if not enabled:
        print(f"{title}: switched off")
        STATUS[title] = "switched off"
        return None
    if hours_left() <= 0:
        print(f"{title}: skipped, the budget of {BUDGET_HOURS} h is used up")
        STATUS[title] = "skipped (no budget left)"
        return None
    stage(title, text)
    started = time.time()
    code = sat(*stages, extra=["--deadline", str(DEADLINE - 300), *(["--force"] if FORCE else [])])
    STATUS[title] = "ok" if code == 0 else "ATTENTION: failed, skipped or incomplete (the feedback report says which and why)"
    print(f"\n{title}: {STATUS[title]} | {(time.time() - started) / 60:.1f} min | {hours_left():.2f} h of budget left\n", flush=True)
    return code == 0


note_step("environment", started, True)
# The record of the environment (GPU, versions, commit) that every later result refers to.
if sat("env") != 0:
    stop("The training interpreter does not see a CUDA device. Send the output above back.")
''')

# --- B: dry run ----------------------------------------------------------------------------------------------------------

md(r'''
## B. 预演（CPU，几分钟）

用随机初始化的小模型和合成数据，把本次会话的每一步（基线回归、轨迹、数据划分、正确性检查、显存标定、短程与一整天的运行、HO-FL 对齐、反馈报告）完整走一遍。它只检查流程在这个环境里能不能跑通，数字没有意义，所以只打印一行结果；失败时打印最后 60 行，notebook 停下并释放 GPU。流程稳定之后可以用 `RUN_DRY_RUN` 关掉。
''')

code(r'''
#@title B · Dry run on CPU
if RUN_DRY_RUN:
    stage("B · Dry run on CPU", "Tiny random models and synthetic data: checks that every step of this session runs here; its numbers mean nothing.")
    started = time.time()
    result = subprocess.run([PY, "experiments/sat/e0/e0.py", "all", "--tiny", "--quiet", "--root", TINY], cwd=REPO_DIR,
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=2400)
    ok = result.returncode == 0
    note_step("dry_run", started, ok)
    STATUS["dry run"] = "ok" if ok else "FAILED"
    print(f"- every step of this session on tiny models: {STATUS['dry run']} ({time.time() - started:.0f} s)", flush=True)
    if not ok:
        print("\n".join((result.stdout + result.stderr).splitlines()[-60:]))
        stop("The dry run failed. Nothing has been run on the GPU yet; send the output above back.")
    print(f"{hours_left():.2f} h of budget left")
else:
    print("B · Dry run: switched off")
''')

md(r'''
## 本次会话的计划

打印这次要跑的每一步：回答什么问题、答案决定什么、大约多久、这次是否打开。复制整页输出时，这张表就是目录。
''')

code(r'''
#@title Plan · What this session runs
PLAN = [
    ("1", RUN_BASELINE, "Baseline regression", "Is upstream HO-FL unchanged by the satellite branch on this GPU",
     "every later GPU step is skipped when it fails", "about 3 min"),
    ("2", RUN_SETUP, "Tests, trace, data", "Do all CPU tests pass in the pinned environment; does the trace agree with appendix A; are the partitions the recorded ones",
     "the GPU steps of the satellite code are skipped when a test fails", "about 6 min, no GPU"),
    ("3", RUN_SELFCHECK, "Correctness on the real models", "Explicit cuts, a participation against upstream, aggregation against the central rule, "
     "replay on two replicas; how faithful the zeroth-order scalars are", "the training runs are skipped when a check fails", "about 6 min"),
    ("4", RUN_MEMORY, "Memory per cut", "Peak GPU memory of a participation at every cut, Qwen and ViT", "the memory states of the system model", "about 3 min"),
    ("5", RUN_SHORT_RUNS, "Two simulated hours", "Does every implemented method run end to end and learn; seconds per participation, round and evaluation",
     "the GPU budget of the experiments; whether the replay needs optimising first", "about 6 min"),
    ("6", RUN_DAY_RUNS, "A whole simulated day on EuroSAT", "Does EuroSAT saturate (gate G3); what a full run costs; does a repeated run reproduce",
     "dataset or main metric of the vision task, before its experiments are run", "about 12 min"),
    ("7", RUN_HOFL_SEEDS, "HO-FL with and without its zeroth-order part", "Seeds 43 and 44 on AG News, to go with seed 42 of 2026-11-21",
     "how the paper describes the zeroth-order part; independent of the satellite code", "about 40 min"),
]
stage("Plan of this session", f"Budget {BUDGET_HOURS} h ({hours_left():.2f} h left). Times are estimates; the feedback report gives the measured ones.")
print("| step | question | what the answer decides | time | this session |\n| --- | --- | --- | --- | --- |")
for key, enabled, title, question, decides, duration in PLAN:
    print(f"| {key} {title} | {question} | {decides} | {duration} | {'yes' if enabled else 'no (switched off)'} |")
''')

# --- the steps -----------------------------------------------------------------------------------------------------------

md(r'''
## 1. 基线回归：原有的 HO-FL 有没有被改坏

三个冒烟配置（HO-FL、FedAvg、DeComFL，OPT-125M / SST-2，3 轮）在上游原样提交上各跑一次，在插桩提交上各跑两次，在当前代码上各跑一次，再加两次 `ho-fl-replay`。卫星分支改动了 `hybrid_step`（可以显式给出分界）、`local_step` 和层号的识别；这一步检查默认设置下结果仍与上游一致。

读法：表里的 `I1 verdict` 为 `PASS_BITWISE` 或 `PASS_TOLERANCE` 即通过。`FAIL` 说明当前代码改变了基线，后面所有 GPU 步骤都会跳过；`INCONCLUSIVE` 说明这块 GPU 上参照提交自己两次运行的差就超过了容差，需要你决定怎么判。
''')

code(r'''
#@title 1 · Baseline regression
run_stage("1 · Baseline regression: is upstream HO-FL unchanged", ["baseline"], RUN_BASELINE,
          "Three smoke configurations (HO-FL, FedAvg, DeComFL on OPT-125M / SST-2, 3 rounds) on the unchanged upstream commit and on this code, "
          "and ho-fl-replay. The single runs are in the log; the tables show the verdicts.")
''')

md(r'''
## 2. 测试、轨迹、数据（不用 GPU）

- **测试**：原有的 CPU 测试和 `tests/sat/` 下的新测试，在固定版本的环境里跑一遍（本机的版本与这里不同，所以这一遍不能省）。
- **轨迹**：生成 120 颗卫星 24 小时的轨迹，逐项对比附录 A（周期、星间距离、地影比例、成像比例、地面站可见比例、可达轨道面数），并给出轨迹哈希。
- **数据**：下载 HumAID 和 EuroSAT，生成 3 个种子、两种数据版本共 12 份划分，打印种子 42 的划分统计，并把 12 个哈希与本机记录的比对。哈希不同不算失败（平台之间浮点末位可能不同），但会写进反馈，之后以 Colab 的为准。
''')

code(r'''
#@title 2 · Tests, trace, data
run_stage("2 · Tests, trace and data", ["tests", "trace", "data"], RUN_SETUP,
          "The CPU test suite in the pinned environment; the 24-hour trace of 120 satellites against appendix A of the brief; "
          "HumAID and EuroSAT with the partitions of seeds 42, 43, 44 in both data variants.")
''')

md(r'''
## 3. 真实模型上的正确性检查，以及零阶估计的保真度

在 ViT-B/16（EuroSAT）和 Qwen2.5-1.5B（HumAID）上各跑一遍，另用 Qwen + AG News 与上游的训练函数比对。每行是一项检查，`pass` 为通过；任何一行是 `FAIL`，后面的训练运行都会跳过。

- 显式给出分界时，本地训练的一步与原来的推导方式逐位相同；分界为 0 时梯度与全反向传播一致；分界在最高层时，各层只靠零阶更新。
- 卫星侧的一次参与，与上游的 `train_client` 逐位相同。
- 理想网络下，卫星聚合与“求和后除以人数”的中心化聚合相比：一阶部分逐位相同，零阶部分相差不超过 1e-7。
- 同一批零阶报告在两个副本上重放，结果逐位相同。

每个任务的检查之后还有一张**诊断表**（不计入关卡）：在各个分界上取 64 个随机方向，把训练实际用的零阶标量除以扰动步长，与反向传播算出的精确方向导数比较，扰动步长取 0.001、0.01（运行采用）、0.1。相关系数和斜率接近 1，说明零阶标量没有被 BF16 舍入淹没；这是“扰动步长取 0.01”这一设定在新任务上的直接检验。
''')

code(r'''
#@title 3 · Correctness on the real models
run_stage("3 · Correctness checks on the real models", ["selfcheck"], RUN_SELFCHECK,
          "Explicit cuts against the derived ones, a participation against upstream train_client, the satellite aggregation against the central "
          "sum divided by the number of participants, the replay of zeroth-order reports on two replicas; then the zeroth-order scalars against "
          "exact directional derivatives (a diagnostic).")
''')

md(r'''
## 4. 显存标定

对每个分界做一次完整参与（5 步本地训练，批大小 16；Qwen 的输入长度取 192，ViT 的输入是 224×224），记录峰值显存。表里另有两组参照：只做一次前向推理，以及冻结分界以下的层（不做零阶）。系统模型里 GPU 卫星“空闲 / 成像”两种状态的显存，分别取分界 8 / 16（ViT 为 3 / 6）的峰值。
''')

code(r'''
#@title 4 · Peak GPU memory per cut
run_stage("4 · Peak GPU memory of a participation per cut", ["memory"], RUN_MEMORY,
          "Qwen2.5-1.5B at cuts 0, 8, 16, 24, 28 and ViT-B/16 at cuts 0, 3, 6, 9, 12; batch 16.")
''')

md(r'''
## 5. 短程运行（5 次，各 2 个仿真小时）

一个仿真小时是 6 轮，每轮约 18 颗卫星受邀。运行时每个仿真小时打印一行训练损失和一行验证准确率。

- 完整方法：EuroSAT、HumAID 各一次；
- 理想中心化、只让强星训练、纯零阶泛洪：EuroSAT 各一次。

这一步回答三件事：流程能否跑完、损失是否下降且没有非有限值、每次参与 / 每轮 / 每次评估各花多少秒（后面实验的 GPU 小时由此估算）。服务器步长都取 1，没有调过，所以这里的准确率不能用来比较方法；评估用验证集，不碰测试集。
''')

code(r'''
#@title 5 · Two simulated hours of every implemented method
run_stage("5 · Short runs: two simulated hours each", ["smoke"], RUN_SHORT_RUNS,
          "The full method on EuroSAT and on HumAID; the ideal centralised, backbone-only and zeroth-order methods on EuroSAT. "
          "Server step 1 everywhere (not tuned), evaluated on the validation split: these accuracies do not compare methods.")
''')

md(r'''
## 6. 一整天的运行（EuroSAT，2 次）

完整方法和理想中心化各跑 24 个仿真小时（144 轮，25 次评估）。开始前会用第 5 步的实测耗时估算这一步要多久，放不进剩余预算就跳过。它不是正式结果（步长没调、单种子、验证集），只为提前回答三个会改变实验设计的问题：

- **EuroSAT 会不会太简单**：理想中心化若很快就到 98%，各方法都会挤在上限附近，准确率区分不了方法。实现说明把这个判断放在 EuroSAT 的实验全部跑完之后（关卡 G3）；用十来分钟提前知道，可以避免十几个 GPU 小时白跑。
- **一次完整运行实际要多久**：后面一百多次 EuroSAT 运行的单位成本。
- **同一配置跑两次是否完全相同**：这两次运行的前 12 轮与第 5 步的对应运行配置相同，逐轮比较就知道 GPU 上是否逐位可复现。方法之间的差距可能只有一两个点，这决定了配对比较里有没有运行间的噪声。
''')

code(r'''
#@title 6 · A whole simulated day on EuroSAT
run_stage("6 · Day-long runs on EuroSAT: the full method and the centralised reference", ["pilot"], RUN_DAY_RUNS,
          "24 simulated hours each, server step 1, validation split. Not a result: it shows whether EuroSAT saturates, what a full run costs "
          "and whether a repeated run reproduces.")
''')

md(r'''
## 反馈（中期）：卫星部分

第 7 步与卫星代码无关，要跑约 40 分钟。卫星部分到这里已经全部跑完，所以先打印一份报告，并给出结果包的下载链接（不自动下载）：可以现在就把报告复制回来，不必等。最后一节会再打印一份完整的（多出第 7 步的结果）。第 7 步关掉时，这一格什么也不做。
''')

code(r'''
#@title Feedback so far · the satellite part
if RUN_HOFL_SEEDS:
    print_feedback("Feedback so far: the satellite part (the HO-FL runs follow)", pending="t0")
    print("\nresult bundle:", "link above (kept in the page; the final section downloads it)" if download_results(auto=False) else "not available")
else:
    print("The HO-FL runs are switched off; the feedback of the whole session follows in the last section.")
''')

md(r'''
## 7. HO-FL 的零阶开与关（4 次运行，约 40 分钟）

与 11-22 状态报告相同的设置：Qwen2.5-1.5B、AG News（Dirichlet 0.5）、HO-FL 默认采样（β=0.75）、160 轮。种子 43、44 各跑两次：保留零阶，以及关掉零阶（冻结分界以下的层）。种子 42 用 11-21 会话的结果（85.99 对 85.95）。反馈报告给出三个种子上“开 − 关”的差值、均值和标准差；作者报告的差值是 3.32 个点。

这一步用的是原有的 `ho-fl-train`，只依赖第 1 步的基线回归。放不进剩余预算的运行会被跳过，并写进反馈。
''')

code(r'''
#@title 7 · HO-FL with and without its zeroth-order part (seeds 43, 44)
run_stage("7 · HO-FL on AG News with the zeroth-order part on and off", ["t0"], RUN_HOFL_SEEDS,
          "Qwen2.5-1.5B, Dirichlet 0.5, beta 0.75, 160 rounds; seeds 43 and 44. Seed 42 is read from the session of 2026-11-21.")
''')

# --- feedback ------------------------------------------------------------------------------------------------------------

md(r'''
## 反馈：这次实验说明了什么，代码该怎么改

这一节把整次会话汇总成一份报告，打印在两条分隔线之间。**把这段复制回来即可。** 报告的内容：

1. **结论**：关卡 G0 是否通过，以及开头那七个问题各自的答案。
2. **关卡清单**：七项检查逐项的结论和依据。
3. **发现与建议的调整**：必须先修的、建议修改的（写明改哪个文件）、需要你决定的（会影响实验设计或结论）、值得知道的观察、得到确认的。
4. **数据**：每一步的用时、正确性检查、轨迹与划分的哈希、显存表、零阶估计的保真度、每次运行的损失 / 准确率 / 参与构成 / 耗时、一整天的学习曲线、HO-FL 三个种子的差值。
5. **失败与异常**：每个失败步骤的日志末尾。
6. **没有运行的部分与下一步**：哪些被跳过、为什么；下一次会话该开哪些开关。

之后浏览器下载结果包 `session3_bundle.zip`（每次运行的日志和 JSON），再释放 GPU。结果包的内容直接放在这一格的输出里，下载不需要再向运行时要数据：如果浏览器没有自动下载（例如拦截了自动下载），点输出里的“点这里下载”即可，GPU 释放之后也能点。释放之后页面上的输出仍然保留；关掉页面之前记得把结果包存下来。
''')

code(r'''
#@title Feedback · report, download, then release the GPU
print_feedback("Feedback of this session")
stage("Session summary", f"Time used: {(time.time() - SESSION_START) / 3600:.2f} h of {BUDGET_HOURS} h.")
print("| step | status |\n| --- | --- |")
for name, state in STATUS.items():
    print(f"| {name} | {state} |")
started = download_results()
print("\nresult bundle:", "download started (session3_bundle.zip); the link above works too, also after the GPU is released" if started
      else "NOT available; the report above is the record of this session")
release_gpu(15)
''')

md(r'''
## 带回来的内容

- **反馈报告**：上一节两条分隔线之间的全部文字，直接贴给我。
- **`session3_bundle.zip`**（浏览器自动下载，可选）：环境记录、测试结果、轨迹的元数据与统计、12 份划分的哈希、正确性检查的逐项结果、显存表、每次运行的 `config.json`、`summary.json`、`rounds.jsonl`、`ledger.jsonl`、`eval.jsonl` 和日志，以及基线回归的记录。需要查某次运行的细节时才用得上。
- 想留档的话，把整页输出复制到 `colab_results_feedback/` 下新建的目录里（例如 `v3_<日期>/`）。每一步以 `# 3 · ...` 这样的标题开始，表格是 Markdown 文本，复制后保持原样。
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
