# P-S0 报告：基础修改与回归

日期：2026-11-24。分支 `sat`（从 `main` 的 `a9be57b` 建立，未推送）。依据：实现说明 v2 §2.2、§4.4。

验收标准是不变量 I1（新选项取默认值时，现有入口和测试的行为不变）和 I2（`ho-fl-replay`、`ho-fl-profile`、`ho-fl-memory` 行为不变）成立。结论：**在 CPU 上成立**；GPU 上的基线回归（notebook 的 C2）**未运行**。

## 1 本阶段改了什么

对现有文件只做了 §4.4 列出的修改，全部是带默认值的关键字参数或新增字段。

| 文件 | 修改 | 默认行为 |
|---|---|---|
| `ho_fl/algorithms.py` | `hybrid_step` 增加关键字参数 `cut=None`；`None` 时仍从前缀推出分界（改用 `model.block_index` 取层号）；前缀为空时不生成任何方向 | 不变。与冻结的上游副本逐位相同 |
| `ho_fl/train.py` | `Backend.local_step` 把已知的分界传给 `hybrid_step` | 不变。分界的值与原来推出的相同 |
| `ho_fl/model.py` | `block_index` 同时接受 `.layers.N.` 和 `.layer.N.`；新增 `register_head`：登记过的分类头参数映射为层号 L | Qwen、OPT、SmolLM3 的参数名结果不变；未登记的名字仍断言失败 |
| `ho_fl/config.py` | `Config` 增加 `scenario: str = ""`；非空时不再要求 `n == 30` | 默认值为空，检查与原来相同 |
| `pyproject.toml` | 增加 5 个 `ho-fl-sat-*` 入口；新增可选依赖组 `sat` | 原有入口和依赖不变 |
| `tests/test_config.py` | 在"新增字段登记表"里登记 `scenario`（该文件要求每个新字段都登记） | |
| `tests/sat/test_sat_base.py` | 16 项测试，每处修改至少一项，见第 2 节 | |
| `ho_fl/sat/__init__.py`、`ho_fl/sat/cli.py` | 新子包与命令行入口 | |

提交：`ecbc382`。notebook 的阶段 S（§4.4 第 6 项）要等引擎和运行队列写好才有内容，放在 `caa692e`，见 P-S3 报告。

## 2 测试

全部在本机 CPU 上运行（Apple M4，Python 3.12.7，torch 2.7.1，transformers 4.54.1，peft 0.18.1；与 `pyproject.toml` 的固定版本不同，Colab 上用固定版本）。

| 测试 | 结果 | 日志 |
|---|---|---|
| 修改前，`main` 上的全部 CPU 测试 | 267 通过，3 个 GPU 测试未选中 | 本次会话的终端输出 |
| 最后一个代码提交 `96b8e16` 上，全部 CPU 测试（原有 267 项加 `tests/sat/` 下的 133 项新测试） | **400 通过，0 失败**，3 个 GPU 测试未选中 | `outputs/sat/local/pytest.log` |
| 基线回归流水线的小模型预演：`experiments/p0/p0.py env golden regress --tiny`（提交 `01aa162`；此后只改了自检表格的文字） | 当前代码与上游原样提交相比，HO-FL、FedAvg、DeComFL 三个冒烟配置都是 `PASS_BITWISE`；`ho-fl-replay` 为 `PASS` | `outputs/sat/local/p0-tiny-regress.log` |

`tests/sat/test_sat_base.py` 的内容：

| 修改 | 测试 | 结果 |
|---|---|---|
| `hybrid_step` 的显式分界 | 玩具模型上分界 1、2、3，小模型上分界 3、6、9：显式传入分界、不传分界、冻结的上游副本，三者连续几步的损失、标量、参数、梯度逐位相同 | 通过 |
| 分界 0 | 梯度与全模型反向传播一致（1e-6）；不生成方向；小模型上与 FedAvg 的梯度一致 | 通过 |
| 前缀为空但没有给分界 | 与上游一样报错 | 通过 |
| 分界等于层数 | 各层得到的是零阶估计，分类头得到的是精确的一阶梯度（带可训练分类头的玩具模型；显式和推导两种方式） | 通过 |
| `local_step` 传分界 | 传给 `hybrid_step` 的分界与前缀对应 | 通过 |
| `block_index` | 上游的参数名结果不变；ViT 的 `.layer.N.` 可识别；登记的分类头映射为 L，并且对任何分界都在后缀里；未登记的名字断言失败 | 通过 |
| `Config.scenario` | 默认为空；`n=120` 不带 `scenario` 仍被拒绝；带 `scenario` 时通过，其余检查照常 | 通过 |
| 入口与依赖组 | 5 个新入口可导入；原有 4 个入口的指向不变；`sat` 组包含 `global-land-mask==1.0.0` | 通过 |

原有的等价性测试也全部通过：`tests/test_tiny_training.py`（整个训练循环与冻结的上游循环比较日志和最终参数哈希，8 个配置）、`tests/test_legacy_steps.py`（`hybrid_step` 与冻结的上游副本逐位比较）。

## 3 运行清单

| 运行 | 目录 | 要点 | 状态 |
|---|---|---|---|
| 小模型基线回归预演 | `outputs/sat/local/p0-tiny/` | 合成的小模型，CPU，3 轮；只说明流程和等价性，数字没有意义 | 完成 |
| G4 上的基线回归（真实的 OPT-125M / SST-2，对照上游提交） | notebook 的 C2 | 与前两次会话相同的流程 | **未运行** |

## 4 结果

| 结论 | 证据 | 判断 |
|---|---|---|
| I1：默认设置下，现有入口和测试的行为不变 | 原有 267 项测试全部通过；小模型上三个冒烟配置与上游提交逐位相同 | CPU 上支持；GPU 上无法判断（未运行） |
| I2：`ho-fl-replay` 行为不变 | 小模型上重放通过（672 项比较，最大误差 3.6e-12） | CPU 上支持 |
| I2：`ho-fl-profile`、`ho-fl-memory` 行为不变 | 这两个文件没有改动；它们调用的 `hybrid_step`（不传分界的路径）与上游副本逐位相同。两个工具本身需要 GPU，没有执行 | 间接支持 |

## 5 与实现说明不一致之处、未解决的问题、需要决定的事项

1. **`tests/golden/` 里没有黄金值。** 该目录的说明写着"尚未记录任何文件"。I1 实际由两处保证：测试里冻结的上游副本（`tests/legacy_loop.py`、`tests/legacy_steps.py`），以及 `experiments/p0/p0.py` 在同一块 GPU 上与上游提交的对比。实现说明 §2.2 提到的"黄金值不变"按后者理解。
2. **状态报告的位置。** 实现说明引用 `docs/reports/status-2026-11-22.md`，实际文件在 `docs/old_version/reports/` 下（工作区里旧的计划和报告都移到了 `docs/old_version/`，这一移动没有提交，我没有动它）。
3. **`Config.scenario` 只放宽 `n == 30`。** `Config.resolve()` 还会拒绝上游表里没有的模型和数据集（`vit`、`humaid`、`eurosat`），`verbalizers` 和 `max_length` 也是按上游数据集查表。所以卫星路径用一个 `Config` 的子类（`ho_fl/sat/config.py` 的 `TrainConfig`，由任务给出 verbalizer 和长度），并且不调用上游的 `resolve()`。`ho_fl/data.py` 的 manifest 检查没有改，卫星路径有自己的 manifest。
4. **可选依赖组 `sat` 多了一个 `pillow==11.3.0`。** EuroSAT 的图像以 JPEG 存储，解码需要 Pillow，而现有依赖里没有它。
5. **`ho-fl-sat-tune`、`ho-fl-sat-topology`、`ho-fl-sat-figures` 三个入口已登记，但还没有实现**，运行时会提示所属的里程碑（P-S4、P-S7）。
6. **分类头的登记是进程内的全局表。** 一个进程里加载过 ViT 之后，它的分类头参数名会一直被识别为第 L 层。对实验没有影响（每次运行一个进程、一个模型）。
7. 仓库是否公开：实现说明写"仓库保持私有"，但 notebook 是匿名克隆，只有公开仓库才行得通。请确认以哪个为准。

## 6 GPU 小时

已用 0。notebook 的 C2（基线回归）在前两次会话里约需 10 分钟。
