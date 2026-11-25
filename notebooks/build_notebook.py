#!/usr/bin/env python
"""Source of ``notebooks/RHO-FFT_colab.ipynb``.

The notebook is generated: edit the cells below and run

    python notebooks/build_notebook.py            # rewrites the notebook
    python notebooks/build_notebook.py --check    # exit code 1 if the notebook on disk is not what this file gives

``tests/sat/test_sat_notebook.py`` checks that the notebook on disk is the generated one and executes its code cells
with every external effect stubbed (no GPU, no network, no Colab).

How the notebook is organised (the way the research lead's other Colab notebooks are): numbered steps, each a
Markdown heading and the cell or cells that do it. Step 0 gets the code, Step 1 installs the dependencies, Step 2
checks the files and the GPU, Step 3 runs the experiment, Step 4 shows the summary and the figures in the page,
Step 5 prints the feedback report, Step 6 hands over the result bundle and Step 7 releases the GPU. There are no
forms and no switches: what a session runs is decided by the code it clones (``experiments/sat/e0/e0.py``), not by
the notebook. There is no time budget either: nothing is skipped or stopped for taking long.

What belongs where: the notebook holds what must exist before the repository does (getting the code, building the
environment) and the order of the session; everything a step does lives in the pipeline it calls, which also writes
the feedback report, and the figures are drawn by ``experiments/sat/e0/plots.py``. Text for the research lead
(Markdown cells, headings of figures, the feedback report) is Chinese; what code prints is English.
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
# RHO-FFT — Google Colab

卫星场景的实验：120 颗卫星在轨微调同一个模型，强星用反向传播，其余卫星在分界以下只做前向（零阶）。

**实验内容完全由代码决定**（`experiments/sat/e0/e0.py`），本 notebook 不做任何覆写，也**不设时间上限**：不会因为跑得久而跳过或中止任何一次运行。

1. GPU：Runtime → Change runtime type → 选 GPU。建议 G4、L4 或 A100；T4 没有 BF16 加速，Qwen 的运行在上面要几个小时。
2. **Step 0** 取代码 → **Step 1** 安装依赖 → **Step 2** 检查环境。
3. **Step 3a** 显存 → **Step 3b** 短程运行 → **Step 3c** 一整天的运行。页面实时打印进度：训练每轮一行（带已用时间和预计剩余时间），任何命令 30 秒没有输出时补一行“仍在运行”。
4. **Step 4** 实验摘要与图：全部 inline 渲染，断开运行时后仍在页面上。
5. **Step 5** 反馈报告：把两条分隔线之间的内容复制回来。
6. **Step 6** 打包下载结果；**Step 7** 释放 GPU。

整次会话在 A100 上实测 54 分钟（2026-11-25）；其他 GPU 上未实测，实测值写在 Step 5 的报告里。

> **使用方式**：Run All → 等 Step 7 释放 GPU → 复制 Step 5 的报告。任何一格出错，也会先打印已有结果的报告、交出结果包，再释放 GPU。手动点停止只停掉当前这一步，不会释放 GPU：之后手动运行 Step 4–7，就能得到已完成部分的报告并释放。
''')

# --- Step 0 --------------------------------------------------------------------------------------------------------------

md(r'''
## Step 0: 获取代码

代码只从 GitHub 读取（公开仓库，不需要 token，不使用 Google Drive）。目录已存在时先同步到 `origin/main`，不会沿用旧版本。

第一格定义后面各步用到的几个函数：运行命令并把输出实时打印到页面、打印反馈报告、交出结果包、释放 GPU。
''')

code(r'''
# Session: paths, and the helpers the later steps use
import json, os, shutil, subprocess, sys, time
from pathlib import Path

REPO_URL = "https://github.com/GuangLun2000/RHO-FFT.git"
REPO_REF = "main"
REPO_DIR = "/content/RHO-FFT"
WORK = "/content/work/sat"       # results of this session (local disk of the runtime)
CACHE = "/content/cache"         # models, datasets, the trace and the partitions
VENV = "/content/venv"           # Python 3.11 with the pinned dependencies
SESSION_START = time.time()
SESSION = {"started_at": SESSION_START, "repo_ref": REPO_REF, "steps": []}   # what the notebook itself did; the report lists it
STATUS = {}                      # one line per step of the experiment, printed at the end


def sh(command, cwd=None, env=None, check=True):
    # Run a command and stream its output into the cell. Nothing is stopped for taking long.
    shown = command if isinstance(command, str) else " ".join(str(part) for part in command)
    print(f"$ {shown}", flush=True)
    process = subprocess.Popen(command, shell=isinstance(command, str), cwd=cwd, env={**(env or os.environ), "PYTHONUTF8": "1"},
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
    for line in process.stdout:
        print(line, end="", flush=True)
    code = process.wait()
    if check and code != 0:
        raise RuntimeError(f"Command failed with exit code {code}: {shown}")
    return code


def note_step(name, started, ok=True):
    # A step of the notebook itself (code, environment): no GPU work, but the GPU is held meanwhile.
    SESSION["steps"].append({"name": name, "minutes": (time.time() - started) / 60, "status": "ok" if ok else "attention", "gpu": False})
    Path(WORK).mkdir(parents=True, exist_ok=True)
    (Path(WORK) / "session.json").write_text(json.dumps(SESSION, indent=1))


def build_report():
    # The feedback stage of the pipeline: it collects what exists into feedback/feedback_E0.json and writes the
    # report, the summary card and the result bundle. Its own output is shown only when no report came out of it.
    report = Path(WORK) / "feedback" / "feedback_E0.md"
    if "PY" in globals():
        done = subprocess.run([PY, "experiments/sat/e0/e0.py", "feedback", "--quiet", "--root", WORK, "--cache-dir", CACHE, "--device", "cuda"],
                              cwd=REPO_DIR, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if not report.exists():
            print((done.stdout or "")[-3000:], (done.stderr or "")[-3000:])
    return report.exists()


def show_report():
    # Print the feedback report between two marker lines: that text is what goes back.
    build_report()
    path = Path(WORK) / "feedback" / "feedback_E0.md"
    print("\n" + "=" * 24 + " 反馈报告：从下一行开始复制 " + "=" * 24)
    print(path.read_text(encoding="utf-8") if path.exists() else "反馈报告没有生成；请把上面的整页输出带回来。")
    print("=" * 24 + " 反馈报告：复制到上一行为止 " + "=" * 24, flush=True)


def offer_bundle(auto=True):
    # Hand the result bundle (logs and JSON files of every run, the report, the figures) to the browser. Returns
    # whether it was handed over.
    #
    # google.colab.files.download cannot do this here. It only shows a script; the browser then asks the kernel for
    # the file over a comm, and the kernel answers such a request only once it is idle. A session that goes on to
    # release the runtime never is, so that download never starts. Instead the bytes travel inside this cell's
    # output: the browser builds the file from what it already has, and the link keeps working after the runtime
    # is gone. ``auto`` also starts the download.
    source = Path(WORK) / "feedback" / "feedback_E0_bundle.zip"
    if not source.exists():
        return False
    megabytes = source.stat().st_size / 1e6
    if megabytes > 20:
        print(f"The result bundle has {megabytes:.0f} MB, too much to pass through the page; the report is the record.")
        return False
    try:
        import base64
        import uuid
        from IPython.display import HTML, display
        name, link = "session_bundle.zip", "bundle-" + uuid.uuid4().hex
        display(HTML(
            f'<a id="{link}" download="{name}" style="font-size:15px">结果包 {name}（{megabytes:.1f} MB）：点这里下载</a>'
            f'<script>(function() {{ const bytes = Uint8Array.from(atob("{base64.b64encode(source.read_bytes()).decode("ascii")}"), c => c.charCodeAt(0));'
            f' const a = document.getElementById("{link}"); a.href = URL.createObjectURL(new Blob([bytes], {{type: "application/zip"}}));'
            f' {"a.click();" if auto else ""} }})();</script>'))
        return True
    except Exception as error:
        print("the result bundle could not be handed to the browser:", repr(error))
        return False


def release_gpu(delay=20):
    # Hand the GPU runtime back. Everything on its local disk is lost, so this comes after the report and the
    # bundle. The page keeps its output.
    print(f"Releasing the GPU runtime in {delay} seconds ...", flush=True)
    time.sleep(delay)
    try:
        from google.colab import runtime
        runtime.unassign()
    except Exception as error:
        print("not on Colab, or unassign is unavailable:", repr(error))
        print("release it by hand: Runtime -> Disconnect and delete runtime")


def _release_on_error(shell, etype, value, traceback, tb_offset=None):
    # An error in any later cell: show it, print the report of what exists, hand over the bundle, then release the
    # GPU instead of leaving it idle. A manual interruption (KeyboardInterrupt) is not an Exception and does not.
    shell.showtraceback((etype, value, traceback), tb_offset=tb_offset)
    try:
        print("\n# Feedback of what ran before the error\n", flush=True)
        show_report()
        offer_bundle()
    except Exception as error:
        print("the report could not be shown:", repr(error))
    release_gpu()


get_ipython().set_custom_exc((Exception,), _release_on_error)
print("session ready; an error in any later cell prints the report and releases the GPU")
''')

code(r'''
# Fetch the repository and check out the branch
started = time.time()
git_env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
if os.path.isdir(f"{REPO_DIR}/.git"):
    print(f"Using the existing clone at {REPO_DIR}; fetching origin ...")
    sh(["git", "-C", REPO_DIR, "fetch", "--quiet", "origin"], env=git_env)
else:
    print(f"Cloning {REPO_URL} ...")
    sh(["git", "clone", "--quiet", REPO_URL, REPO_DIR], env=git_env)
# A reused directory must not silently keep an older version of the code.
remote_ref = f"origin/{REPO_REF}"
has_remote_ref = subprocess.run(["git", "-C", REPO_DIR, "rev-parse", "--verify", "--quiet", remote_ref],
                                capture_output=True).returncode == 0
sh(["git", "-C", REPO_DIR, "checkout", "--quiet", "--detach", remote_ref if has_remote_ref else REPO_REF])
print("repository HEAD:")
sh(["git", "-C", REPO_DIR, "log", "--oneline", "-3"])
note_step("code", started)
''')

# --- Step 1 --------------------------------------------------------------------------------------------------------------

md(r'''
## Step 1: 安装依赖

建一个独立的 Python 3.11 虚拟环境，按 `pyproject.toml` 的固定版本安装依赖（PyTorch 2.6.0 / CUDA 12.4），外加卫星部分的可选依赖（陆地掩膜、Pillow）。所有运行都是这个环境里的子进程；仓库本身不安装，运行时通过 `PYTHONPATH` 指向源码树。

G4 是 Blackwell 架构的 RTX PRO 6000，固定版本的 PyTorch 2.6.0 不支持它。环境建好后先探测 GPU：固定版本用不了时，自动换成 `TORCH_FALLBACK` 指定的版本（CUDA 12.8 构建），实际使用的版本记在每个结果里。
''')

code(r'''
# Build the environment the experiment runs in
started = time.time()
TORCH_FALLBACK = "2.7.1"   # installed only if the pinned PyTorch 2.6.0 cannot use this GPU


def clean_environment():
    # Colab exports constraint files for its own package set (PIP_CONSTRAINT, UV_CONSTRAINT, ...). They must not
    # apply to the separate environment built here, which pins different versions.
    return {key: value for key, value in os.environ.items() if not key.startswith(("UV_", "PIP_"))}


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

sh([PY, "-c", "import torch, transformers, peft, datasets; "
              "print('torch', torch.__version__, '| cuda runtime', torch.version.cuda, '| cuda available', torch.cuda.is_available()); "
              "print('transformers', transformers.__version__, '| peft', peft.__version__, '| datasets', datasets.__version__)"])

# Probe the GPU with the interpreter that will train, in the precision of the backbones. The pinned PyTorch has no
# kernels for Blackwell GPUs (the G4 runtime); if it cannot use this GPU, the fallback build for CUDA 12.8 is
# installed and probed again. The versions actually used are recorded with every result.
GPU_PROBE = ("import torch; x = torch.randn(256, 256, device='cuda', dtype=torch.bfloat16); "
             "assert torch.isfinite((x @ x).float()).all(); "
             "print('GPU usable:', torch.cuda.get_device_name(0), '| torch', torch.__version__)")


def gpu_usable():
    return sh([PY, "-c", GPU_PROBE], check=False) == 0


if shutil.which("nvidia-smi") is None:
    raise RuntimeError("No GPU in this runtime. Choose a GPU runtime type and run again.")
if not gpu_usable():
    print(f"\nThe pinned PyTorch cannot use this GPU; installing torch {TORCH_FALLBACK} built for CUDA 12.8 instead.")
    fallback = f"{uv} pip install --python {VENV}/bin/python torch=={TORCH_FALLBACK}"
    if sh(fallback + " --torch-backend cu128", env=clean_environment(), check=False) != 0:
        sh(fallback + " --index-url https://download.pytorch.org/whl/cu128", env=clean_environment())
    if not gpu_usable():
        raise RuntimeError("PyTorch cannot use this GPU with either build. Send the output above back.")
    STATUS["environment note"] = f"torch {TORCH_FALLBACK} (CUDA 12.8) in place of the pinned 2.6.0, which does not support this GPU"
note_step("environment", started)
print("environment ready")
''')

# --- Step 2 --------------------------------------------------------------------------------------------------------------

md(r'''
## Step 2: 检查文件与 GPU

确认核心文件齐全，打印 GPU 型号、显存和 PyTorch 版本，并记录本次会话的环境（每个结果都引用它）。GPU 早于 Ampere 一代（例如 T4）时会提示：它没有 BF16 加速，Qwen 的运行要几个小时。
''')

code(r'''
# Check the files and the GPU, and record the environment of this session
required = ["experiments/sat/e0/e0.py", "experiments/sat/e0/digest.py", "experiments/sat/e0/plots.py",
            "ho_fl/sat/engine.py", "ho_fl/sat/cli.py", "pyproject.toml"]
missing = [name for name in required if not os.path.exists(f"{REPO_DIR}/{name}")]
print("missing files:", missing) if missing else print("core files present")


def sat(*stages):
    # Stages of experiments/sat/e0/e0.py, the pipeline of this session; 0 = every stage ok.
    return sh([PY, "experiments/sat/e0/e0.py", *stages, "--root", WORK, "--cache-dir", CACHE, "--device", "cuda"], cwd=REPO_DIR, check=False)


GPU_FACTS = ("import torch; p = torch.cuda.get_device_properties(0); major, minor = torch.cuda.get_device_capability(0); "
             "print(f'GPU: {p.name} | {p.total_memory / 1e9:.1f} GB | compute capability {major}.{minor} | torch {torch.__version__}'); "
             "print('NOTE: this GPU is older than the Ampere generation and has no fast path for BF16, the precision of the backbones. "
             "On a Tesla T4 (session of 2026-11-25) one evaluation of HumAID took 11.6 minutes: expect the Qwen runs of Step 3b "
             "to take hours. Nothing is stopped for taking long.') if major < 8 else None")
sh([PY, "-c", GPU_FACTS], check=False)
if sat("env") != 0:
    raise RuntimeError("The training interpreter does not see a CUDA device. Send the output above back.")
''')

# --- Step 3 --------------------------------------------------------------------------------------------------------------

md(r'''
## Step 3a: 显存——每个分界上一次参与的峰值

对每个分界做一次完整参与（5 步本地训练，批大小 16），记录峰值显存：Qwen 在分界 0、8、16、24、28，ViT 在分界 0、3、6、9、12。页面每测完一项打印一行。

**Step 3 的参数完全来自代码**，三步都没有时间上限。同一个运行时里重跑时，已经完成的测量和运行直接复用，不会重跑。
''')

code(r'''
# Step 3a: peak GPU memory of a participation per cut
def run_step(title, *stages):
    # One step of the experiment. Nothing here raises and nothing is stopped for taking long: the pipeline records
    # what failed, and the notebook goes on to the summary, the report and the release of the GPU.
    print(f"\n# {title}\n", flush=True)
    started = time.time()
    code = sat(*stages)
    STATUS[title] = "ok" if code == 0 else "ATTENTION: failed or incomplete (the feedback report says which and why)"
    print(f"\n{title}: {STATUS[title]} | {(time.time() - started) / 60:.1f} min\n", flush=True)


run_step("Step 3a · Peak GPU memory of a participation per cut", "memory")
''')

md(r'''
## Step 3b: 短程运行（7 次，各 2 个仿真小时）

一个仿真小时是 6 轮，每轮约 18 颗卫星受邀。

- **EuroSAT（ViT-B/16）**：完整方法、理想中心化、只让强星训练、纯零阶泛洪各一次。
- **HumAID（Qwen2.5-1.5B）**：完整方法、理想中心化、只让强星训练各一次。

第一次运行会先生成轨迹和数据划分，之后的运行直接复用。页面每轮打印一行，每次评估开始前和结束后各打印一行。

服务器步长都取 1，没有调过，所以这里的准确率不能用来给方法排名；评估用验证集，不碰测试集。
''')

code(r'''
# Step 3b: two simulated hours of every implemented method
run_step("Step 3b · Short runs: two simulated hours each", "smoke")
''')

md(r'''
## Step 3c: 一整天的运行（EuroSAT，4 次）

“一整天”是仿真里的 24 小时（144 轮、25 次评估），不是 GPU 上的时间。四种已实现的方法各一次：2 个仿真小时太短，看不出方法之间的差别（例如只让强星训练只用到十分之一的数据，12 轮里显不出来）。

它仍然不是正式结果（步长没选、单种子、验证集），用来回答四件事：四种方法在足够长的时间上各到哪里、EuroSAT 会不会太简单、一次完整运行要多久、同一配置跑两次是否逐位相同。
''')

code(r'''
# Step 3c: a whole simulated day of every implemented method on EuroSAT
run_step("Step 3c · Day-long runs on EuroSAT: every implemented method", "pilot")
''')

# --- Step 4 --------------------------------------------------------------------------------------------------------------

md(r'''
## Step 4: 实验摘要与图

- **实验摘要卡**：GPU、PyTorch、代码版本、各步骤的状态，以及每次运行的起止准确率和耗时。
- **图**（inline，同时存入结果包）：各方法的验证准确率随仿真时间的变化；一整天的运行另有每轮训练损失；完整方法里一阶、零阶两条通道对每一层更新的贡献；每个分界的峰值显存。

> 图里的文字是英文（Colab 的字体没有中文字形），图上方的标题是中文。图只画已经完成的运行；每个数字都在 Step 5 的反馈报告里。
''')

code(r'''
# Step 4: summary card and figures, shown in the page
import importlib.util
from IPython.display import Image, Markdown, display

build_report()
feedback = Path(WORK) / "feedback"
card = feedback / "summary_card.md"
display(Markdown("## 实验摘要\n\n" + (card.read_text(encoding="utf-8") if card.exists() else "摘要没有生成，见 Step 5 的反馈报告。")))
try:   # the figures are a convenience: the report of Step 5 has every number, so a failure here costs nothing else
    spec = importlib.util.spec_from_file_location("sat_plots", f"{REPO_DIR}/experiments/sat/e0/plots.py")
    plots = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plots)
    drawn = plots.figures(json.loads((feedback / "feedback_E0.json").read_text(encoding="utf-8")), feedback / "figures")
    for heading, path, error in drawn:
        display(Markdown(f"### {heading}"))
        if error is None:
            display(Image(filename=str(path), width=760))
        else:
            print("not drawn:", error)
    if not drawn:
        print("no finished measurement or run to draw yet")
except Exception as error:
    print("the figures could not be drawn:", repr(error))
''')

# --- Step 5 --------------------------------------------------------------------------------------------------------------

md(r'''
## Step 5: 反馈报告——这次实验说明了什么，代码该怎么改

整次会话汇总成一份报告，打印在两条分隔线之间。**把这段复制回来即可。** 报告依次是：结论和各个问题的答案；本次会话的范围；发现与建议的调整（必须先修的、建议修改的、需要你决定的）；数据表；失败步骤的日志末尾；没有运行的部分与下一步。
''')

code(r'''
# Step 5: the feedback report, between two marker lines
print(f"\n# Feedback of this session ({(time.time() - SESSION_START) / 3600:.2f} h)\n", flush=True)
show_report()
''')

# --- Step 6 --------------------------------------------------------------------------------------------------------------

md(r'''
## Step 6: 打包下载结果

结果包 `session_bundle.zip` 里是每次运行的日志和 JSON、反馈报告和 Step 4 的图，备查。它的内容直接放在这一格的输出里：没有自动下载时，点输出里的“点这里下载”即可，GPU 释放之后也能点。
''')

code(r'''
# Step 6: hand the result bundle to the browser
if not offer_bundle():
    print("result bundle: not available; the report of Step 5 is the record of this session")
''')

# --- Step 7 --------------------------------------------------------------------------------------------------------------

md(r'''
## Step 7: 释放 GPU 并断开 Colab 运行时

**务必运行这一格**（Run All 会自动运行到这里），避免长时间占用 GPU。释放后运行时上的文件全部消失，页面上的输出保留。
''')

code(r'''
# Step 7: release the GPU runtime
print(f"All steps done ({(time.time() - SESSION_START) / 3600:.2f} h).")
for title, state in STATUS.items():
    print(f"- {title}: {state}")
release_gpu(20)
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
            "metadata": {"accelerator": "GPU", "colab": {"name": "RHO-FFT_colab.ipynb", "provenance": [], "gpuType": "G4"},
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
