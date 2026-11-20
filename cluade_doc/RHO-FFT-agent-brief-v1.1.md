# RHO-FFT 实施说明：逐层一阶供给与韧性混合阶联邦微调

*Agent brief for RHO-FFT (Resilient Hybrid-Order Federated Fine-Tuning), built on the HO-FL codebase*

| 项目 | 内容 |
|---|---|
| 版本 | v1.1，2026-11-20（相对 v1.0 的改动见附录 E） |
| 项目名 | RHO-FFT（Resilient Hybrid-Order Federated Fine-Tuning）。既是仓库名，也是本研究提出的方法名 |
| 论文工作标题 | Resilient Hybrid-Order Federated Fine-Tuning of Foundation Models over Wireless Edge Networks |
| 投稿目标 | IEEE Network 专刊 "Distributed Intelligence and Collaborative Inference for Foundation Models over Wireless Edge"，截稿 2026-12-31。里程碑 P0 到 P3 服务于这次投稿，P4 和 P5 留给后续技术长文 |
| 仓库 | `GuangLun2000/RHO-FFT`，本地路径 `Desktop/Github/RHO-FFT`。本文件放在仓库的 `cluade_doc/` 下 |
| 代码基线 | 上游 `HKU-WILL-Lab/ho-fl`，commit `ca74bed900b1af133bea2412bfdeba45447116dc`（2026-09-26）。仓库工作区中的 `ho_fl/`、`pyproject.toml`、`README.md` 是该提交的原样拷贝，逐文件 sha256 一致 |
| 算力假设 | 单张 H200 级 GPU |
| 读者 | 负责实现的 coding agent；研究负责人 Hanlin Cai（Lance） |

标注约定：**[核]** 已对照论文或代码核对；**[算]** 由已核对的量推算；**[假]** 待实验检验的假设；**[理想]** 理想化模型下的预测，不是实验结果。

---

## 0 这份文件怎么用

你将在一个公开的研究代码库上做增量开发。目的有两个：检验一组关于混合阶（hybrid-order，HO）联邦微调的假设；实现由这些假设导出的调度、聚合与上行机制，并与基线公平比较。

阅读顺序：

1. §1 到 §3 说明为什么做、要回答什么问题、哪些结论算新。实现中遇到本文件没有覆盖的取舍时，请回到这三节，选择更能回答研究问题的做法。
2. §4 和 §5 是代码现状与不变量。§5 的每一条都写了原因，动手前请读完。
3. §6 是模块规格，§7 是实验矩阵，§8 是里程碑、验收标准与决策关卡。
4. 在 §8 标出的决策关卡处停下，提交报告，等研究负责人确认后再继续。

仓库现状（2026-11-20 核对）：`main` 分支只有一次提交，跟踪了 `.gitattributes` 和一个初始 `README.md`。工作区里有上游代码的原样拷贝（`ho_fl/` 下 19 个文件、`pyproject.toml`、上游的 `README.md`），尚未提交。仓库里没有 `.gitignore`。P0 的第一步是把这份拷贝原样提交为基线（见 §8）。

三条贯穿全程的约定：

- 本文件与代码不一致时以代码为准，并在报告里指出不一致之处。
- 报告中的每个数字都来自一次有日志的运行，并附运行目录。没有跑的实验写"未运行"，不估计，不外推。
- 你的产出是代码、测试、运行日志、汇总表和里程碑报告。论文文字由研究负责人撰写。

---

## 1 研究背景与动机

### 1.1 HO-FL 是什么 [核]

HO-FL（arXiv:2609.39074）让显存不同的设备共同微调同一个带 LoRA 的语言模型。客户端 $i$ 有一个分界 $c_i$：层号小于 $c_i$ 的层用零阶（ZO）更新，只需要前向计算；层号不小于 $c_i$ 的层用一阶（FO）更新，只对上半段做反向传播。显存越小，$c_i$ 越大。客户端上传 ZO 部分的标量（随机种子由服务器预先分配）和 FO 部分的参数增量；服务器重建每个客户端的整模型增量，对选中的 $K$ 个客户端逐坐标等权平均。采样概率为 $p_i=(1-\beta)/N+\beta r_i$，其中 $r$ 把概率集中到 ZO 维度最小的 $K$ 个客户端，默认 $\beta=0.75$。

代码默认设置：$N=30$，$K=6$，160 轮，每次参与做 $E=5$ 步本地 AdamW（学习率 $10^{-5}$，权重衰减 $5\times10^{-4}$，每次参与重置优化器），$q=2$ 个扰动方向，$\mu=10^{-3}$，批大小 16。LoRA 秩 8，作用于 `q_proj` 和 `v_proj`；主干为 BF16，LoRA 为 FP32。三档设备各 10 台，分界与标签无关：

| 模型 | 层数 | T1（强） | T2（中） | T3（弱） |
|---|---|---|---|---|
| OPT-125M | 12 | 3 | 6 | 9 |
| Qwen2.5-1.5B | 28 | 8 | 16 | 24 |
| SmolLM3-3B | 36 | 9 | 18 | 27 |

下文把 T1、T2、T3 的分界记为 $c_1<c_2<c_3$，层号从 0 开始。

### 1.2 观察：谁能给哪一层写一阶更新 [核][算]

一台设备只能对 $\ell\ge c_i$ 的层给出一阶报告。于是模型被三个分界切成四段，每段的一阶来源不同。以 Qwen2.5-1.5B 为例（OPT-125M 结构相同，段界为 3、6、9）：

| 段 | 层 | 能给一阶报告的设备 | 每轮一阶报告数的期望：均匀采样 / 默认 $\beta=0.75$ |
|---|---|---|---|
| S0 | 0 到 7 | 无 | 0 / 0 |
| S1 | 8 到 15 | T1 的 10 台 | 2.0 / 5.0 |
| S2 | 16 到 23 | T1 和 T2 的 20 台 | 4.0 / 5.5 |
| S3 | 24 到 27 | 全部 30 台 | 6.0 / 6.0 |

S1 是最稀缺的一段，下文称为稀缺段。用仓库自带的采样器模拟 200 个种子、每个 160 轮，得到 S1 的两种状态：

- 均匀采样（$\beta=0$）：69.2% 的轮次里 S1 至多收到 2 份一阶报告，26.5% 的轮次只有 1 份，6.4% 的轮次一份也没有。
- 默认采样（$\beta=0.75$）：S1 每轮平均收到 5.0 份，但 96% 来自固定的 6 台 T1 设备。这 6 台每台出现在 80% 的轮次里，其余 24 台每台只出现在 5% 的轮次里。

### 1.3 由机制直接推出的三件事

**(a) 零阶报告几乎不含方向信息，范数却与一阶报告相近。[算]** `hybrid_step` 的零阶估计是 $v_p=\langle\lambda,\,h(\theta+\mu u_p)-h(\theta)\rangle\approx\mu\langle\nabla_\theta L,u_p\rangle$，其中 $\lambda=\partial L/\partial h_{\mathrm{cut}}$ 由上半段的反向传播得到。因此 $\hat g=\frac{1}{q}\sum_p u_p v_p/\mu$ 是一个前向梯度估计，满足 $\mathbb{E}[\hat g]=\nabla L$ 和 $\cos(\hat g,\nabla L)\approx\sqrt{q/(d+q+1)}$，$d$ 为该客户端的零阶维度。代入默认设置：Qwen 三档分别约为 0.0025、0.0018、0.0015，OPT 约为 0.0052、0.0037、0.0030。AdamW 把每一步每个坐标的位移归一到学习率的量级，所以零阶报告的范数与一阶报告处于同一量级，方向却接近随机。

**(b) 等权平均稀释一阶信号。[算]** 设第 $t$ 轮第 $\ell$ 层收到 $n_{\ell,t}$ 份一阶报告，服务器实际施加的更新是
$$\Delta_{\ell,t}=\frac{n_{\ell,t}}{K}\,\bar\Delta^{\mathrm{FO}}_{\ell,t}+\frac{K-n_{\ell,t}}{K}\,\bar\Delta^{\mathrm{ZO}}_{\ell,t}.$$
均匀采样时 S1 的一阶信号平均只剩 $2/6$，默认采样时为 $5/6$。HO-FL 把 $\beta$ 的收益解释为采样加权零阶维度的下降；按这里的算术，同样的收益也可以来自一阶信号被稀释得更少。两种解释可以用实验区分（见 H1a）。

**(c) 一份不可靠的一阶报告能否被同层的其他报告否决，取决于同层有几份一阶报告。[算]** 一份不可靠报告在 $n$ 份一阶报告中占少数需要 $n\ge3$。如果聚合器不区分阶数，把 $K=6$ 份报告一起做稳健统计，那么正常的一阶报告需要至少 4 份才构成多数。均匀采样下 S1 只在约 31% 的轮次有 3 份以上的一阶报告，有 4 份以上的轮次不到 8%；默认采样下两个门槛基本都能达到，代价是同样的 6 台设备长期在场。

这三件事合在一起，就是本研究的出发点：在混合阶联邦微调里，决定学习进度和容错能力的量是每一层每一轮收到的一阶报告的数量、来源和到达情况。每轮选中的客户端数 $K$ 只是名义规模。

### 1.4 落脚点与范围

论文面向通用的无线边缘网络，用一个具体的落脚点贯穿全文：一组显存分档的终端与边缘设备，经一个边缘服务器共同微调同一个语言模型。强档是少数大内存的边缘网关或工作站，中档是笔记本和平板一类设备，弱档是 12 GiB 级终端。HO-FL 报告全反向传播需要 40.26 GiB，分界在第 24 层时为 9.08 GiB [核]；三档各对应哪类设备是本文的读法，HO-FL 没有逐档指定。实验的默认设置（$N=30$，三档各 10 台，$K=6$）就是这个落脚点的实例，不需要额外的系统建模。

四个供给性质在这个落脚点上的含义：宽度来自显存分档；多样性来自服务器的调度；及时性来自拥塞或小区边缘上行下的轮次截止时间；可靠性对应标注流程或配置出错的设备。

本研究的结论限定在两条假设之内：服务器可以选择每轮参与的设备；设备按显存分档。低轨卫星等由可见性决定参与者的场景不满足第一条假设，留给后续工作，本阶段不为它建模。

---

## 2 核心概念与假设

### 2.1 定义

- **报告**：客户端 $i$ 在第 $t$ 轮对整模型的增量 $\Delta_{i,t}$。它在第 $\ell$ 层的分量是一阶的（$\ell\ge c_i$）或零阶的（$\ell<c_i$）。服务器知道每个分量的阶数：真实系统中分界由客户端申报并用于重建，代码中分界来自 manifest。
- **一阶表决集合** $\mathcal{F}_{\ell,t}$：第 $t$ 轮在第 $\ell$ 层给出一阶分量并按时到达的客户端。
- **逐层一阶供给**（layer-wise first-order supply）：$\mathcal{F}_{\ell,t}$ 随层和轮次变化的整体情况，用四个量刻画：
  - 宽度 $n_{\ell,t}=|\mathcal{F}_{\ell,t}|$；
  - 多样性 $N_{\mathrm{eff},\ell}=1/\sum_i s_{i,\ell}^2$，$s_{i,\ell}$ 为设备 $i$ 在第 $\ell$ 层全部一阶报告中的份额；
  - 及时性：一阶分量在截止时间前到达的比例；
  - 可靠性：来源是否持续给出有问题的报告。
- **法定数**（quorum）：对一层启用"只在一阶报告之间做稳健统计"所需的最小宽度。本研究提出的配置取 3；配置字段 `fo_quorum` 的默认值是 1，以保持基线行为。
- **不可靠来源**：在一段时间内持续给出有问题报告的设备。成因可以是标注流程出错、超参数配置错误、本地模型过期，也可以是设备被控制。本研究不区分成因，只看报告。

### 2.2 假设

每条假设都给出检验方式和被否定时的处理。请按实际结果报告，不要为了支持假设而调参。

**H1（零阶报告的信息量）[假]** 在 HO-FL 默认预算下，客户端在某层的零阶分量与同一客户端用相同起点、相同批次算出的一阶分量之间的余弦在 $10^{-3}$ 到 $10^{-2}$ 量级，两者范数之比在 0.3 到 3 之间。模型层面的推论：关闭零阶更新对最终准确率的影响小于种子间标准差。
- 已有的旁证 [核]：HO-FL 表 9 中 $q$ 取 1、2、4、8 时 OPT/SST-2 的准确率为 86.51、86.39、86.47、86.47；纯零阶的 DeComFL 在 OPT/SST-2 上为 51.95%，在 OPT/AG News 上为 23.38%。
- 检验：E1b 的逐层探针，E1c 的 A 臂与 B 臂。
- 若被否定：零阶更新在多轮累积后有可测的贡献。此时保留零阶分量参与表决，给它一个较低的权重，叙述从"一阶供给"调整为"按阶加权的供给"。

**H1a（稀释）[假]** 把聚合改成"每层只在一阶报告之间取平均"之后，均匀采样（$\beta=0$）就能取得偏向采样（$\beta=0.75$）在等权平均下的大部分收益；只把全局步长放大 2 到 3 倍做不到这一点。
- 检验：E1c 的 C 臂、A 臂，以及步长对照。判为支持的标准：$\beta=0$ 时 C 臂相对 A 臂的提升不低于"A 臂从 $\beta=0$ 到 $\beta=0.75$ 的提升"的 70%，且步长对照的提升不到 C 臂提升的一半。
- 若成立：HO-FL 用偏向采样换来的效率，可以在不集中参与的前提下由聚合取得。这是贡献 C1 中最有分量的一条。
- 若被否定：$\beta$ 的收益不来自稀释，集中参与无法单靠聚合规避，调度在设计中的比重相应提高。

**H2（法定数交叉）[假]** 稀缺段上有一个不可靠的一阶来源、另有 $m$ 份正常一阶报告时：
- (i) 只在一阶报告之间做稳健统计，在 $m\ge2$ 时有效，在 $m\le1$ 时比不区分阶数的稳健统计更差，因为此时没有可用的多数，一阶报告又不再被稀释；
- (ii) 不区分阶数的稳健统计对任意 $m$ 都把影响限制在有界范围内，零阶报告在其中起压舱作用，但它的学习进度停留在被稀释的水平；
- (iii) 两者在 $m=2$ 附近交叉。按法定数切换的聚合在每个 $m$ 上都取较好的一支。
- 检验：E2 的剂量响应实验。附录 B 给出理想化模型下的预测形状。
- 若被否定（例如正常一阶报告之间差异太大，3 份报告的稳健统计不起作用）：把法定数提高到 4，配额相应改为 `4,1,1`，并报告正常一阶报告之间的余弦分布。

**H3（采样旋钮的两种失效）[假]** 在只用等权平均的前提下，不可靠设备属于偏向集合时默认采样受损最重（80% 的轮次在场），均匀采样次之（20% 在场，但稀缺段常常没有其他一阶报告与之抗衡），不可靠设备不在偏向集合时默认采样受损最轻（5% 在场）。配额轮转加法定数切换聚合在三种角色上的最坏受损最小，且无故障时的准确率与 HO-FL 默认设置相差不超过 0.5%。
- 检验：E3。
- 若被否定：按实际排序报告，并检查是哪一环（宽度、多样性或聚合规则）没有起作用。

**H4（截止时间先削薄稀缺段）[假]** T1 设备的一阶上行载荷是 T3 的 5 倍（Qwen：3.11 MB 对 0.62 MB）[算]。在链路速率与档位无关、各档共用一个截止时间的条件下，按时到达的一阶供给在稀缺段下降最快。把一阶分量自底向上发送（稀缺段优先）可以保住这部分供给。
- 检验：E4。
- 若被否定：无线部分只作为系统约束出现在论文里，不作为贡献。

**H5（结构性稀缺与前向裁判，可选）[假]** 当强设备本来就很少（例如三档为 3、9、18 台）时，轮转无法增加来源，需要由只做前向计算的弱设备在留出数据上评估"去掉某个来源在某段的贡献后损失是否下降"。这一条留给 P4，不在本次投稿范围内。

### 2.3 一个贯穿设计的判断

零阶报告作为方向几乎没有用，作为参照却有用：它们给稳健统计提供压舱（H2 的 (ii)），它们的范数给出"一份正常报告应该有多大"的参照，产生它们的前向计算能力还可以用来评估别人的更新（H5）。设计上因此让一阶报告负责写入，让零阶一侧负责约束。

---

## 3 研究问题、预期贡献与新颖性边界

### 3.1 研究问题

- **RQ1（测量）** 混合阶联邦微调中，每一层的有效更新来自哪里？显存分档、调度和截止时间怎样决定逐层一阶供给的宽度、多样性和及时性？零阶报告和一阶报告各自贡献多少？
- **RQ2（韧性）** 一个持续不可靠的一阶来源造成的损害和恢复过程，怎样随所在层的一阶供给变化？不区分阶数的稳健聚合在混合阶场景下得到什么、失去什么？
- **RQ3（设计）** 在设备资源不变（相同的 $K$、$E$、$q$、显存分档和截止时间）的条件下，面向供给的调度、聚合和上行顺序能否同时改善效率与韧性？

### 3.2 预期贡献（以实验结果为准）

- **C1 概念与测量。** 提出逐层一阶供给，并在 HO-FL 公开代码上给出独立测量（就目前的检索所见，尚无同类测量）：供给图、零阶报告的信息量、稀释效应，以及"偏向采样的收益可以由聚合取得"是否成立。
- **C2 韧性分析。** 用受控的剂量响应实验给出损害随一阶供给宽度变化的关系；检验采样旋钮的两种失效是否存在；检验不区分阶数的稳健聚合是否只能得到有界但被稀释的结果。
- **C3 设计。** RHO-FFT（Resilient Hybrid-Order Federated Fine-Tuning），一个面向供给的混合阶联邦微调框架：按法定数约束的分档配额加档内轮转；按法定数切换的逐层稳健聚合；稀缺段优先的上行顺序。在相同设备资源下与 HO-FL、通用稳健聚合、参与偏差校正等基线比较。

### 3.3 新颖性边界

下表列出与本工作最接近的已有工作。写代码和写报告时请守住右栏的界线。

| 已有工作 | 它做了什么 | 本工作不同之处，以及不应声称的内容 |
|---|---|---|
| HO-FL（arXiv:2609.39074） | 客户端内混合阶；按零阶维度偏向采样；等权平均；指出 $\beta$ 在更新质量与数据代表性之间折中 | 没有逐层来源视角，没有可靠性分析。HO-FL 是我们的基线和代码来源 |
| HO-SFL（arXiv:2603.14773）、HERON-SFL（arXiv:2601.09076） | 分割式混合阶，服务器端一阶 | 架构不同，作为背景 |
| ElasticZO（arXiv:2501.04287） | 单设备上末几层反向传播、其余层零阶 | 不涉及联邦和多来源 |
| CyBeR-0（arXiv:2406.14362，2502.00193） | 全零阶联邦学习的拜占庭稳健聚合 | 所有客户端同阶；不存在"同一层混有两种阶的报告"的问题 |
| SALF（arXiv:2403.18375）、FedRA（arXiv:2311.11227）、ZorBA（arXiv:2603.04436） | 不同客户端训练不同的层；每层由不同的贡献者集合更新 | "每层只在贡献者之间聚合"是已有做法。`fo_mean` 是它在阶异构上的实例，不作为新方法声称 |
| LASA（arXiv:2409.01435） | 逐层的稳健聚合 | 假设所有客户端更新所有层；不区分阶数 |
| Karimireddy 等（arXiv:2006.09365） | 正常客户端的数据异构会削弱稳健聚合，用分桶缓解 | 那里的异构是统计性的、未标注的；阶异构是结构性的，服务器已知每个分量的阶数，所以可以直接限定表决集合 |
| Malinovsky 等（arXiv:2311.14127） | 部分参与下被采样的一批客户端可能被不可靠者主导，用裁剪缓解 | 我们指出逐层的表决集合比名义批次更薄，并用法定数切换处理 |
| FedAuto（arXiv:2512.22035） | 从聚合角度处理连接不可靠和数据不均衡下的联邦微调，自适应调整聚合 | 按摘要，它处理的是连接不可靠造成的报告缺失，未涉及混合阶。我们处理的是报告本身不可靠，以及同一层混有两种阶的报告。E4 的截止时间实验需要与它对照定位 |
| Ma 等（Electronics 15(20):4746, 2026） | 按可用性做逆倾向加权和加权几何中位数 | 客户端级、由可用性驱动。在混合阶下，逆倾向加权会放大很少被选中的弱设备，而它们的报告以零阶分量为主（E3 中作为可选基线检验） |
| TiFL（arXiv:2001.09249） | 按训练速度分档，每轮从一档中选客户端 | 分档选择是已有做法。我们的配额由逐层一阶法定数推出 |
| BaFFLe（arXiv:2011.02167）、Zeno | 用验证数据评估更新 | 与 H5 相关，P4 再比较 |

2026-11-19 至 20 日的检索没有发现针对混合阶联邦学习的容错或稳健聚合工作。检索不保证穷尽，论文定稿前研究负责人会再查一次。

### 3.4 本研究不做的事

- 不设计针对某个聚合器优化的偏离策略。故障注入限于 §6.6 列出的良性成因故障和稳健聚合文献中的标准非自适应压力条件。
- 不为任何方法单独调超参数。
- 不改变 HO-FL 的本地训练算法本身（`hybrid_step` 的数值路径在默认配置下保持不变）。
- 不为低轨卫星或其他由可见性决定参与者的场景建模（见 §1.4）。

---

## 4 代码库现状 [核]

### 4.1 文件地图

| 文件 | 作用 |
|---|---|
| `ho_fl/train.py` | `Backend`（模型、数据、分界、本地一步、评估）、`Sampler`、`logical_uplink`、`run`（主循环）。入口 `ho-fl-train` |
| `ho_fl/algorithms.py` | `optimizer_for`、`task_loss`、`restore`、`perturb`、`hybrid_step` |
| `ho_fl/config.py` | `SPECS`（模型、修订号、层数、三档分界）、`Config` 数据类、`resolve()`、`parse_config` |
| `ho_fl/sampling.py` | `probabilities`、`dep_round`、`DepRoundSampler` |
| `ho_fl/model.py` | `load_model`（LoRA 注入、精度设置）、`block_index`、`partition_parameters` |
| `ho_fl/split.py` | `Partition` 及 `QwenPartition`、`OPTPartition`、`SmolLM3Partition`：`forward_prefix`、`forward_suffix`、`forward` |
| `ho_fl/data.py` | 数据集表、`load_tokens`、`generate_manifest`、`load_manifest` |
| `ho_fl/partitioning.py` | `allocate`（Dirichlet 划分）及三种 manifest 生成函数 |
| `ho_fl/batches.py` | `collate`、`ClientStream`（每个客户端一个带种子的数据流） |
| `ho_fl/randomness.py` | `stable_seed`、`RoundDirections`、`client_directions` |
| `ho_fl/classification.py` | `evaluate`（准确率与损失） |
| `ho_fl/decomfl.py` | DeComFL 基线（文件头带 Apache 许可文本） |
| `ho_fl/replay.py` | 种子加标量重放的数值校验。入口 `ho-fl-replay` |
| `ho_fl/profile.py`、`ho_fl/memory.py` | 显存测量与批次显存模拟。入口 `ho-fl-profile`、`ho-fl-memory` |
| `ho_fl/squad.py`、`ho_fl/tokenization.py`、`ho_fl/io.py` | SQuAD 路径、分词缓存、JSON 与哈希工具 |

### 4.2 一轮训练在代码里怎样进行

`run(cfg)` 的每一轮：

1. `selected = sampler.sample()` 得到 $K$ 个客户端编号。
2. 对每个客户端按 `selected` 的顺序：`restore(b.params, state)`；新建 AdamW；做 $E$ 次 `b.local_step(cid, streams[cid].next(), optimizer, directions, e, state)`；把 `param - state` 累加进 `delta_sum`；写一行 `ledger.jsonl`。
3. `state[name].add_(delta_sum[name], alpha=1 / cfg.k)`，再 `restore(b.params, state)`。
4. 写 `rounds.jsonl`；按 `evaluate_every` 评估并写 `curves.jsonl`。

`hybrid_step` 的一步：对前缀做无梯度前向得到 `hidden`；在 `h_cut` 上建图，对后缀前向并反向，得到后缀参数的梯度和 `lam`；对 $q$ 个方向逐个扰动前缀、前向、计算标量 `v` 并累加估计；把估计写入前缀参数的 `.grad`；对全部可训练张量做一次 `optimizer.step()`。

### 4.3 约束设计的事实

- **F1 单模型顺序模拟。** 所有客户端共用一份模型，靠 `restore` 回到全局状态。客户端增量是 `param - state`，在模拟里直接可得，不需要走种子重放。
- **F2 聚合的算术形式。** 先按 `selected` 顺序逐个 `add_`，再一次 `add_(…, alpha=1/k)`。要逐位复现基线，需要保持这个顺序和形式。
- **F3 采样器。** `DepRoundSampler` 以 `backend.dimensions[cut]` 为输入，偏向集合是 `d/q` 最小的 $K$ 个客户端，平局用种子 `40000+seed` 打破，逐轮随机数种子为 `30000+seed`。三档各 10 台时，偏向集合是 10 台 T1 中的 6 台。
- **F4 分界与 manifest。** 每个客户端的分界存在 manifest 的 `mixed_L_zo` 字段，由 `torch.randperm(30)`（种子 `20000+seed`）分配，与标签无关。`load_manifest` 断言 `n==30`、各客户端样本数相等、每个分界恰好 10 台。`--manifest` 可以传入外部文件。缓存路径只由模型、数据集、划分和种子决定。
- **F5 种子体系。** 划分 `10000+seed`，分界 `20000+seed`，采样 `30000+seed`，平局 `40000+seed`；客户端数据流 `stable_seed(seed, model, cid, "data")`；零阶方向 `stable_seed(seed, model, round, cid, "zo_master")`。各随机源互相独立，HO-FL 路径的训练循环不消耗全局随机数（全局种子只在 `load_model` 中设置，影响 LoRA 初始化）。
- **F6 输出。** `status.json`、`config.json`、`manifest.json.gz`、`ledger.jsonl`、`rounds.jsonl`、`curves.jsonl`、`summary.json`。`run` 拒绝写入非空目录。JSON 写入使用 `allow_nan=False`。`summary.json` 目前只有初始参数哈希，没有最终参数哈希。
- **F7 评估。** 默认每 `max(1, 50 // E)` 轮（即 10 轮）在完整评估集上评估一次。AG News 用官方测试集 7,600 条，SST-2 用验证集 872 条。
- **F8 配置解析。** `parse_config` 按数据类字段生成命令行参数（下划线变连字符），字段类型限于 `str`、`int`、`float`、`bool`；`--config` 读 JSON，未知字段报错。
- **F9 `resolve()` 的限制。** `n` 只能是 30；非 IID 划分只支持 AG News；采样器名单固定；非 `hofl` 方法的 `beta` 被置 0；`beta>0` 时只允许 `dep_round_v1`。
- **F10 其他入口依赖现有签名。** `replay.py` 构造 `Backend` 并按位置参数调用 `b.local_step(...)`，还读取前缀参数的 `.grad`；`profile.py` 用未经 `resolve()` 的 `Config` 按位置参数调用 `hybrid_step(...)`；`memory.py` 读取 `config.json` 中的 `actual_cuts`、`bottom_dimensions`、`sampler_metadata`。
- **F11 分界取值。** `hybrid_step` 从前缀参数名推断分界，要求前缀非空。`Backend.bottoms` 只为 `SPECS[model][3]` 中的三个分界预先建好前缀。
- **F12 确定性。** `load_model` 调用 `torch.use_deterministic_algorithms(False)`，所以同一配置两次运行是否逐位相同需要在目标 GPU 上实测。
- **F13 工程现状。** 上游没有测试、没有 CI、没有 LICENSE 文件。上游的 `.gitignore` 忽略 `outputs/`、`cache/`、`data/`、`models/`、`local/`、`checkpoints/`、`*.pt`、`proxy*`、`mirror*`、`*token*.txt` 等；本仓库目前没有 `.gitignore`，P0 需要补上。
- **F14 模型范围。** 只支持三种因果语言模型，分类任务通过最后位置的 verbalizer logits 完成。没有视觉语言模型的支持。

---

## 5 不变量与工作规则

- **I1 基线等价。** 所有新选项取默认值时，`ho-fl-train` 的结果与改动前一致：三个冒烟配置（见 P0）的逐轮 `task_loss`、评估指标和最终参数哈希相同。原因：后面所有比较都以"我们没有改坏基线"为前提。如果未改动的代码在目标 GPU 上本身不能逐位复现，改用 `replay.py` 的容差（最大绝对误差 $10^{-7}$），并在报告中写明。
- **I2 现有入口可用。** `ho-fl-replay` 通过；`ho-fl-profile` 与 `ho-fl-memory` 行为不变。原因：它们是 HO-FL 论文数值的校验工具，也是我们对代码理解是否正确的检查。
- **I3 增量式修改。** 新功能放在新模块里。对现有文件的修改尽量小，新增参数用带默认值的关键字参数。不重排、不格式化现有代码。`decomfl.py` 只在不可避免时改动。原因：便于审阅差异，也便于将来与上游合并。
- **I4 配对比较。** 同一种子下被比较的方法共享数据划分、分界分配、初始参数、客户端数据流（每台设备按参与次序取到相同的批次序列）和零阶方向。不可靠设备的身份由种子决定，与方法无关。原因：效应量可能只有 1% 左右，不配对就会被种子间方差淹没。
- **I5 不单独调参。** 学习率、$E$、$q$、$K$、$\mu$、批大小、轮数保持 HO-FL 默认值，除非某个实验明确写了别的值。原因：公平，也避免把调参收益误认为方法收益。
- **I6 结果可追溯。** 报告中的数字都能追到运行目录。失败和发散的运行要报告，不能丢弃。
- **I7 故障范围。** 只实现 §6.6 列出的故障类型。原因见 §3.4。
- **I8 仓库卫生。** 在 `main` 之外的工作分支上开发，提交粒度小，提交信息、代码和注释用英文。未经研究负责人指示不推送到远端，也不改写已有提交。上游没有 LICENSE 文件，在许可问题澄清之前仓库保持私有；导入上游代码的提交要在提交信息里写明来源和 commit。

---

## 6 增量设计

### 6.1 把一轮拆成显式的阶段

把 `run` 中每轮的循环体整理成下面的阶段。各阶段内部的运算顺序与现有代码一致。

```python
for r in range(1, cfg.rounds + 1):
    plan = scheduler.sample(r)                    # 客户端名单，可带分界覆盖
    reports = []
    for cid in plan.clients:
        ctx = faults.before_local(r, cid)         # 训练级故障：学习率倍数、标签映射、过期起点；通常为空
        rep = train_client(b, state, streams, r, cid, plan.cut(cid), ctx, cfg)   # 返回 ClientReport
        rep = faults.after_local(r, rep)          # 报告级故障；通常原样返回
        reports.append(rep)
    arrivals = links.arrivals(r, reports)         # 未设截止时间时：全部按时
    probes.maybe_run(r, b, state, reports)        # 只读，不改变任何状态
    update, agg_log = aggregation.aggregate(reports, arrivals, cfg)
    aggregation.apply(state, update, cfg.server_lr)
    restore(b.params, state)
    scheduler.observe(r, arrivals)
    # 日志与评估
```

`train_client` 就是现有内层循环的函数化：`client_directions` → `restore` → `b.optimizer()` → $E$ 次 `b.local_step` → 计算 `param - state`。`ClientReport`（放在 `ho_fl/reports.py`）包含：`cid`、`round`、`cut`（实际使用的分界）、`tier`、`delta`（名字到张量的字典）、`scalars`、`batches`（本次用到的样本下标，供探针复用）、`train_loss`、`fault`（故障类型或空）。

### 6.2 模块一览

| 模块 | 新建或修改 | 内容 |
|---|---|---|
| `ho_fl/reports.py` | 新建 | `ClientReport`；按层和按段给张量名分组的工具；判断某份报告在某层的阶数 |
| `ho_fl/scheduling.py` | 新建 | 调度器接口和四种实现（§6.3） |
| `ho_fl/aggregation.py` | 新建 | 表决集合、统计量、法定数切换（§6.4） |
| `ho_fl/faults.py` | 新建 | 故障规格解析和三类钩子（§6.6） |
| `ho_fl/links.py` | 新建 | 链路与时间模型、到达判定（§6.7） |
| `ho_fl/supply.py` | 新建 | 供给指标，解析计算与从日志计算。入口 `ho-fl-supply`（§6.8） |
| `ho_fl/probes.py` | 新建 | 逐层统计与反事实一阶探针（§6.9） |
| `ho_fl/dose.py` | 新建 | 剂量响应实验的编排。入口 `ho-fl-dose`（§6.10） |
| `ho_fl/analysis.py` | 新建 | 从运行目录生成整洁的 CSV 和速览图。入口 `ho-fl-analyze`（§6.11） |
| `ho_fl/train.py` | 修改 | 按 §6.1 重排主循环；`Backend.local_step` 增加关键字参数 `cut=None`、`label_map=None`、`zo=True`，`Backend.optimizer` 增加 `lr=None`；最终参数哈希；状态保存与加载；非有限值处理 |
| `ho_fl/algorithms.py` | 修改 | `hybrid_step` 增加 `zo=True` 关键字参数；`optimizer_for` 增加 `lr=None` 关键字参数 |
| `ho_fl/config.py` | 修改 | 新字段（§6.5）和 `resolve()` 的相应检查 |
| `ho_fl/classification.py` | 修改 | 可选的分类别准确率 |
| `ho_fl/data.py` | 修改（P4） | 放宽"每档恰好 10 台"的断言，由配置给出各档台数 |
| `tests/` | 新建 | 不需要 GPU 的单元测试，以及需要 GPU 的回归测试（用 pytest 标记区分） |
| `experiments/` | 新建 | 各实验的 JSON 配置和网格运行脚本 |
| `pyproject.toml` | 修改 | 新入口；`dev` 可选依赖（pytest）；`analysis` 可选依赖（matplotlib） |

### 6.3 调度（`scheduling.py`）

接口：`sample(round_id)` 返回名单和可选的分界覆盖；`observe(round_id, arrivals)` 接收到达情况；`inclusion()` 返回每台设备的边际入选概率（确定性规则返回长期频率）；`metadata()` 返回写入 `config.json` 的描述。`sample` 另接受一个可选的 `available` 参数（本轮可达设备的集合，默认全部可达），为以后接入可达性轨迹留出接口，本阶段不使用。

| `sampler` 取值 | 行为 |
|---|---|
| `dep_round_v1`、`historical_torch`、`historical_numpy_uniform` | 现有实现，原样包装。`config.json` 中的 `sampler_metadata` 键保持不变 |
| `tier_equalised` | 取现有采样器在当前 $\beta$ 下的入选概率，在每个档位内取平均，再用 `dep_round` 抽样（随机数种子 `30000+seed`）。各档的期望名额与 HO-FL 相同，档内不再偏向固定设备。用途：把"集中"与"宽度"分开 |
| `tier_quota` | 每轮从 T1、T2、T3 各取固定名额，名额由 `tier_quota` 给出（如 `3,1,2`），和为 $K$。档内规则由 `in_tier` 决定：`rotate` 选最近 `rotate_window` 轮内被选次数最少的设备，平局用带种子的随机数打破；`random` 档内均匀无放回；`fixed` 始终选同一批设备（T1 取现有偏向集合的前几台），仅用于对照 |
| `schedule_file` | 从 JSONL 文件读每轮的名单，格式为 `{"round": r, "clients": [...], "cut_override": {"cid": cut}}`。供剂量响应实验使用。分界覆盖只能取 `SPECS[model][3]` 中的值 |

关于轮转依据：`rotate_key` 默认为 `selected`（按被选次数）。另一个取值 `delivered`（按按时送达次数）会让长期迟到的设备被反复选中，占用名额却不提供供给；实现它是为了对照，不作为默认。

名额怎样定：稀缺段要达到法定数 3，T1 的名额至少为 3。`3,1,2` 在满足法定数的前提下给弱设备留了与均匀采样相同的参与率（20%），是主配置；`4,1,1` 作为敏感性对照；`2,2,2` 是分层的均匀采样，用来区分"平均宽度不足"和"宽度随机波动"。

各调度器的解析指标见附录 A，可直接用作单元测试的期望值。

### 6.4 聚合（`aggregation.py`）

对每个聚合单元 $u$（默认是一层，即同一层的四个 LoRA 张量），记 $\mathcal{A}_u$ 为在 $u$ 上有分量到达的报告，$\mathcal{F}_u\subseteq\mathcal{A}_u$ 为其中分量是一阶的报告。`zo_mode=off` 时零阶分量视为不存在，此时 $\mathcal{A}_u=\mathcal{F}_u$。

**表决集合** $\mathcal{E}_u$ 由 `agg_electorate` 决定：

- `all`：$\mathcal{E}_u=\mathcal{A}_u$。
- `fo`：若 $|\mathcal{F}_u|\ge$ `fo_quorum`，则 $\mathcal{E}_u=\mathcal{F}_u$；否则按 `thin_policy` 处理：`all` 退回 $\mathcal{A}_u$ 并使用同一统计量；`dilute` 对 $\mathcal{F}_u$ 取和后除以 $K$；`hold` 本轮不更新该单元。
- 表决集合为空时，该单元本轮不更新。

**统计量**由 `agg_stat` 决定：

- `mean`：按报告顺序逐个累加，再乘以 $1/K$（`agg_weight=cohort`，默认）或 $1/|\mathcal{E}_u|$（`agg_weight=count`）。
- `geomed`：几何中位数，平滑 Weiszfeld 迭代，从均值出发，至多 100 次，迭代相对变化小于 $10^{-6}$ 时停止；$|\mathcal{E}_u|\le2$ 时返回均值。在单元内所有张量拼接成的向量上计算。
- `cwmed`：逐坐标中位数。报告数为偶数时取中间两个值的平均。
- `trimmed`：逐坐标去掉最大值和最小值后取平均；报告数小于 3 时返回均值。

**可选项**：`norm_clip=c`（$c>0$）在统计之前把每个分量的范数裁剪到 $c$ 乘以该单元上到达分量范数的中位数；`server_lr` 乘在最终更新上；`agg_unit` 可取 `layer`（默认）、`segment`（按 §1.2 的四段）、`whole`（整个模型作为一个单元，只与 `agg_electorate=all` 搭配，对应稳健聚合的经典用法）。

**默认配置需要逐位等于现有实现**：`agg_electorate=all`、`agg_stat=mean`、`agg_weight=cohort`、无截止时间时，对每个张量按 `selected` 顺序 `add_`，再 `state.add_(sum, alpha=cfg.server_lr / cfg.k)`。请为这条路径保留与现有代码相同的算术形式，不要用 `torch.stack(...).mean(0)` 之类的等价写法替换。

实验中使用的具名配置：

| 名称 | `agg_electorate` | `agg_stat` | `agg_weight` | `fo_quorum` | `thin_policy` | 说明 |
|---|---|---|---|---|---|---|
| `mean_all` | `all` | `mean` | `cohort` | 不适用 | 不适用 | HO-FL 现有做法 |
| `fo_mean` | `fo` | `mean` | `count` | 1 | `all` | 有一阶报告的层只在一阶报告之间平均 |
| `fo_diluted` | `fo` | `mean` | `cohort` | 1 | `all` | 丢掉零阶分量但保留 $1/K$ 的权重，用于分离零阶更新的净贡献 |
| `robust_all` | `all` | `geomed` | 不适用 | 不适用 | 不适用 | 不区分阶数的逐层稳健聚合 |
| `robust_whole` | `all` | `geomed` | 不适用 | 不适用 | 不适用 | `agg_unit=whole`，经典的整向量稳健聚合 |
| `fo_robust` | `fo` | `geomed` | 不适用 | 1 | `all` | 只在一阶报告之间做稳健统计，没有法定数保护 |
| `quorum_robust` | `fo` | `geomed` | 不适用 | 3 | `all` | 本研究提出的配置：达到法定数时在一阶报告之间做稳健统计，否则退回全部报告 |

`cwmed` 和 `trimmed` 作为 `geomed` 的替换项，用于检查结论是否依赖具体统计量。

### 6.5 新增配置字段

全部加在 `Config` 数据类里，默认值复现现有行为。命令行参数由 `parse_config` 自动生成。

| 字段 | 类型与默认值 | 取值 | 含义 |
|---|---|---|---|
| `sampler`（已有） | `str`，`dep_round_v1` | 新增 `tier_equalised`、`tier_quota`、`schedule_file` | 调度器 |
| `tier_quota` | `str`，`""` | 如 `3,1,2` | T1、T2、T3 每轮名额 |
| `in_tier` | `str`，`rotate` | `rotate`、`random`、`fixed` | 档内选择规则 |
| `rotate_window` | `int`，20 | 正整数 | 轮转统计窗口 |
| `rotate_key` | `str`，`selected` | `selected`、`delivered` | 轮转依据 |
| `schedule_file` | `str`，`""` | 路径 | 显式名单 |
| `agg_electorate` | `str`，`all` | `all`、`fo` | 表决集合 |
| `agg_stat` | `str`，`mean` | `mean`、`geomed`、`cwmed`、`trimmed` | 统计量 |
| `agg_weight` | `str`，`cohort` | `cohort`、`count` | `mean` 的分母 |
| `agg_unit` | `str`，`layer` | `layer`、`segment`、`whole` | 聚合单元 |
| `fo_quorum` | `int`，1 | 正整数 | 一阶表决集合的最小规模 |
| `thin_policy` | `str`，`all` | `all`、`dilute`、`hold` | 不足法定数时的处理 |
| `norm_clip` | `float`，0.0 | 0 表示关闭 | 范数裁剪倍数 |
| `server_lr` | `float`，1.0 | 正数 | 服务器步长 |
| `zo_mode` | `str`，`on` | `on`、`off` | 是否执行零阶更新 |
| `fault_spec` | `str`，`""` | 路径 | 故障规格 |
| `link_spec` | `str`，`""` | 路径 | 链路规格 |
| `deadline_s` | `float`，0.0 | 0 表示关闭 | 每轮截止时间（模拟秒） |
| `upload_policy` | `str`，`all_or_nothing` | `all_or_nothing`、`top_first`、`scarcity_first` | 上行顺序 |
| `layer_stats` | `bool`，`False` | | 每轮记录逐层统计 |
| `probe_rounds` | `str`，`""` | 如 `1,20,40,80,160` | 运行反事实探针的轮次 |
| `save_state_rounds` | `str`，`""` | 如 `40` | 保存全局状态的轮次 |
| `init_state` | `str`，`""` | 路径 | 从保存的状态开始 |
| `on_nonfinite` | `str`，`raise` | `raise`、`stop` | 出现非有限值时报错，或记为发散并正常结束 |
| `per_class_eval` | `bool`，`False` | | 评估时输出分类别准确率 |

`resolve()` 需要相应放宽和补充：允许新的 `sampler` 取值；`tier_quota` 的三个数之和等于 `k`；`tier_equalised` 允许 `beta>0`，`tier_quota` 和 `schedule_file` 不使用 `beta`；`agg_unit=whole` 只与 `agg_electorate=all` 搭配；非 `hofl` 方法下把报告的阶数按 `Backend.cuts` 处理（`fedavg` 全为一阶，`decomfl` 全为零阶），聚合器不需要特判。`method_label` 在非默认调度或聚合时追加相应名称。

`zo_mode=off` 的实现：`hybrid_step(..., zo=False)` 跳过扰动循环，也不给前缀参数的 `.grad` 赋值。AdamW 会跳过 `.grad` 为空的参数（包括权重衰减），前缀因此保持不动。`logical_uplink` 在该模式下把标量字节记为 0。

### 6.6 故障注入（`faults.py`）

规格文件是 JSON，包含若干条目，每条给出设备选择、起止轮次、类型和参数：

```json
{"faults": [{"select": {"role": "preferred_t1", "count": 1}, "rounds": [41, 80],
             "kind": "lr_scale", "params": {"scale": 20}}]}
```

- **设备选择**：`by_id`，或按角色 `preferred_t1`（T1 中属于偏向集合的设备）、`ordinary_t1`（T1 中不属于偏向集合的设备）、`t2`、`t3`，配合 `count` 和种子。偏向集合一律按同一种子下 `dep_round_v1`、$\beta=0.75$ 的定义计算，与当前使用的调度器无关，这样各方法面对的是同一台设备（I4）。
- **类型**：

| 类别 | `kind` | 参数 | 含义 |
|---|---|---|---|
| 训练级，良性成因 | `label_noise` | `rate`，`map`（`shift` 或 `random`） | 本地标签按比例被替换，对应标注流程出错 |
| 训练级，良性成因 | `lr_scale` | `scale` | 本地学习率被放大或缩小，对应配置错误 |
| 训练级，良性成因 | `stale` | `lag` | 从若干轮之前的全局状态开始本地训练，对应模型未及时更新 |
| 报告级，良性成因 | `noop` | 无 | 按时返回全零增量 |
| 时间类 | `late` | `extra_s` 或 `drop` | 报告迟到或缺席，用作对照 |
| 报告级，标准压力条件 | `sign_flip` | `scale`，可选 `layers` | 增量取反并缩放 |
| 报告级，标准压力条件 | `gauss` | `scale`，可选 `layers` | 用范数为原范数若干倍的高斯噪声替换增量 |

后两种是稳健聚合文献里常用的非自适应测试条件。`layers` 可以把报告级故障限制在某个层段，剂量响应实验用它把影响限定在稀缺段。范围以此表为限，不再增加类型。

- **钩子**：`before_local` 返回训练级的覆盖项（学习率倍数、标签映射、过期起点）；`after_local` 变换报告；时间类故障交给 `links`。为此 `Backend.local_step` 增加关键字参数 `label_map=None`，`Backend.optimizer` 增加 `lr=None`，训练循环保留最近若干轮的全局状态供 `stale` 使用。
- **干净孪生**：为了计算聚合层面的影响，故障设备的本地训练在故障版本之外再跑一遍无故障版本，使用相同的起点和相同的批次下标，得到同一轮的干净报告。这一遍不调用 `ClientStream.next()`，不改动任何计数器。

### 6.7 链路与截止时间（`links.py`）

目的是回答 H4，不追求信道模型的精细程度。规格文件给出：

- 每台设备的上行速率中位数，默认在 0.5 到 8 Mbps 之间按对数均匀抽取（对应拥塞或小区边缘到中等质量的上行）（种子 `stable_seed(seed, "link", cid)`），与档位无关；逐轮速率在中位数上乘以对数正态衰落（默认标准差 4 dB）。也可以给出逐轮逐设备的速率轨迹文件（CSV：`round, client, uplink_mbps`）。
- 计算时间：以"一次完整前向"为单位计数。按现有 `hybrid_step`，每个本地步是一次完整前向、后缀的反向和 $q$ 次前缀前向。反向与前向的耗时比默认取 2，此时 HO-FL 每轮各档都是 15.0 个前向当量；`zo_mode=off` 时 Qwen 三档为 12.14、9.29、6.43 [算]。每档一次前向的秒数由规格给出，默认各档相同。
- 上行顺序：先发头部和零阶标量（58 字节），再按 `upload_policy` 逐层发送一阶分量。`top_first` 从最高层往下发；`scarcity_first` 从分界层往上发。由于层越低能提供一阶报告的设备越少，自底向上就是按稀缺程度优先。每层一阶分量的到达时间等于计算时间加累计字节数除以速率。
- 到达判定：`all_or_nothing` 要求最后一个字节在截止时间前到达，否则整份报告（含零阶部分）作废；另外两种策略按层判定，已到达的层参与聚合。
- 每轮的模拟墙钟时间：所有报告都按时到达时取最晚到达时间，否则取截止时间。累计值写入日志，供"准确率对墙钟时间"的曲线使用。

另提供一个不训练的命令（可并入 `ho-fl-supply`）：给定链路规格和截止时间，输出各档的按时比例和各层按时一阶供给的期望，供研究负责人选定截止时间。默认参数是占位值，实验结果按"T1 整份报告的按时比例"这类无量纲量来组织，以降低对绝对常数的依赖。

### 6.8 供给指标（`supply.py`）

两种计算方式给出同一组逐层指标：

- 解析：输入各设备的分界和调度器的入选概率，输出每层的一阶可用设备数、期望宽度、单设备最大份额、份额最高的 $K$ 台设备的合计份额、$N_{\mathrm{eff}}$。
- 经验：输入运行目录（`ledger.jsonl`，以及启用时的 `layers.jsonl`），输出实际宽度的均值和分布、宽度低于法定数的轮次比例、按时宽度。

### 6.9 探针（`probes.py`）

- **逐层统计**（`layer_stats`，开销很小）：每轮每层记录一阶和零阶分量的份数、各自均值的范数、一阶之间、零阶之间以及两类之间的平均两两余弦、最终更新的范数、本轮采用的表决集合。
- **反事实一阶探针**（`probe_rounds`，每个被选客户端多做一遍完整反向的本地训练）：从同一全局状态出发，用 `ClientReport.batches` 中记录的相同批次，对全部 LoRA 参数做一阶本地训练，得到反事实增量。逐层记录实际分量与反事实分量的余弦和范数比。客户端的一阶层应接近 1，可作为探针实现的自检；零阶层给出 H1 需要的量。

探针需要是只读的：不调用 `ClientStream.next()`，不改动 `processed_tokens` 等计数器，结束后恢复参数。验收标准是开启探针与关闭探针的两次运行最终参数哈希相同。

### 6.10 剂量响应实验（`dose.py`）

目的：在其他条件不变时，只改变稀缺段上正常一阶报告的份数 $m$，观察不同聚合规则下的损害。

- 起点：某次基线运行保存的第 40 轮状态（`save_state_rounds=40`，再用 `init_state` 载入）。
- 名单：每轮固定包含一台指定的 T1 设备 $f$，另外 5 台从其余 29 台中按种子抽取。所有实验臂共用同一份名单序列。
- 分界覆盖：5 台中的前 $m$ 台分界设为 $c_1$（在 S1 上是一阶），其余设为 $c_2$（在 S1 上是零阶，在 S2 和 S3 上仍是一阶）。这样不同的 $m$ 之间只有 S1 的阶数组成不同，设备和数据都相同。分界覆盖是受控实验的手段，不对应真实设备的显存。
- 故障：对 $f$ 施加 `sign_flip`，`layers` 限定在 S1，`scale` 取 1（范数不变、方向相反）和 20（大范数）两档。
- 实验臂：$m\in\{0,\dots,5\}$ 乘以聚合配置 `mean_all`、`fo_mean`、`robust_all`、`fo_robust`、`quorum_robust`，每个臂配一个无故障的配对运行。
- 长度：20 轮，结束时评估。
- 指标：结束时测试损失和准确率相对配对运行的差；S1 上的相对影响，定义为 $\lVert a(\text{含故障报告})-a(\text{含干净孪生})\rVert\,/\,(\lVert f-f_0\rVert/K)$，其中 $a(\cdot)$ 是聚合结果，$f$ 和 $f_0$ 是故障报告和干净孪生在 S1 上的分量。报告级故障的干净孪生就是变换前的报告，不需要重新训练。按定义 `mean_all` 的相对影响为 1。S1 上的一阶表决集合规模是 $m+1$，因为 $f$ 本身也是一阶来源。

各臂之间相互配对；它们与原始运行第 40 轮之后的轨迹不配对，因为数据流从头开始。

### 6.11 汇总（`analysis.py`）

输入若干运行目录，输出：

- `summary.csv`：每次运行一行，含配置要点、最终指标和韧性指标；
- `curves.csv`、`supply.csv`、`probe.csv`、`dose.csv`：长表格式，便于直接作图；
- 速览用的 PNG 图。

资源指标：每轮上行字节数（沿用 `logical_uplink` 的口径）、每轮前向当量、各档参与台数。

韧性指标以"同一配置、同一种子、无故障"的运行为参照：窗口内平均下降（故障窗口内各评估点上准确率差的平均）、最大下降、恢复轮数（窗口结束后差距首次不超过 0.5% 并连续保持两个评估点所需的轮数，未恢复则记为截尾）、最终差距。准确率的差统一写成百分数之差。

### 6.12 日志的增补

只增加键和文件，现有键的含义和取值不变。

- `ledger.jsonl` 每行增加：`tier`、实际分界、`fault`、各层到达情况、计算时间、上行时间、是否按时。对没有入选概率定义的调度器，`p` 与 `inclusion_probability` 写调度器给出的边际值。
- `layers.jsonl`（启用 `layer_stats` 时）：§6.9 的逐层统计。
- `probe.jsonl`（启用探针时）。
- `rounds.jsonl` 增加累计模拟时间和按时报告数；`curves.jsonl` 增加累计模拟时间和可选的分类别准确率。
- `summary.json` 增加 `final_parameter_sha256`；`status` 可取 `DIVERGED`。
- `config.json` 增加调度、聚合、故障、链路的描述。

---

## 7 实验矩阵

模型与数据：先在 OPT-125M / AG News（Dirichlet $\alpha=0.5$）上做，运行快，且 HO-FL 与 FedAvg 的差距大（59.58 对 78.24 [核]），效应容易看清；再在 Qwen2.5-1.5B / AG News（$\alpha=0.5$）上确认，这是论文的主设置。种子取 42、43、44；OPT/AG News 的种子间标准差较大（论文中为 2.24 到 6.52），在 OPT 上请用 5 个种子（42 到 46）。除特别说明外使用 HO-FL 默认超参数。比较两种配置时，报告逐种子的配对差及其均值和标准差，不只报告各自的均值。请先测一次完整运行的耗时，据此估算每组实验的 GPU 小时数，超过 20 GPU 小时的网格在启动前写进报告。

| 编号 | 目的 | 内容 | 对应假设 | 论文中的位置 |
|---|---|---|---|---|
| E0 | 复现 | OPT/SST-2（$\beta=0.75$）；OPT/AG News 和 Qwen/AG News 的 FedAvg、DeComFL、HO-FL（$\beta=0$ 和 0.75） | 无 | 不出图，作为可信度依据 |
| E1a | 供给图 | 不训练。对 `dep_round_v1`（$\beta=0$ 和 0.75）、`tier_equalised`、`tier_quota`（`3,1,2`、`4,1,1`、`2,2,2`）计算逐层供给指标 | RQ1 | 图：供给图 |
| E1b | 各阶信息量 | 在 HO-FL（$\beta=0$ 和 0.75）运行中开启 `layer_stats`，并在第 1、20、40、80、160 轮运行反事实探针。探针只读，可以直接加在 E1c 的 A 臂上 | H1 | 与 E1c 合为一图 |
| E1c | 消融阶梯 | 见下表 | H1、H1a | 同上 |
| E2 | 剂量响应 | §6.10 | H2 | 图：剂量响应 |
| E3 | 端到端韧性 | 见下文 | H3 | 图：故障窗口内的准确率曲线；表：汇总 |
| E4 | 截止时间 | 见下文 | H4 | 图：按时供给与准确率对墙钟时间 |
| E5 | 结构性稀缺与裁判 | P4，获批后再做 | H5 | 技术长文 |
| E6 | 视觉语言模型 | P5，获批后再做 | 无 | 技术长文 |

IEEE Network 限制图表合计不超过 6 个，所以上表右栏的四到五个位置就是全部的结果展示。每个实验的输出请以能直接画出对应图表为目标。

**E0 的验收**：各种子的均值与 HO-FL 表 2 的差不超过 $\max(1.0\%,\,2\times\text{论文标准差})$。我们的种子未必与论文相同，所以这是合理性检查，不要求逐位一致。参考值 [核]：OPT/SST-2 为 86.54（0.24）；OPT/AG News 的 FedAvg、DeComFL、HO-FL（$\beta=0$）、HO-FL（$\beta=0.75$）为 78.24（2.97）、23.38（0.08）、59.58（2.24）、62.85（6.52）；Qwen/AG News 为 88.15（0.21）、55.83（0.08）、85.51（0.43）、86.69（0.56）。超出范围时停下报告。E0 的 HO-FL 运行同时保存第 40 轮状态，供 E2 使用。

**E1c 的实验臂**（每个臂在 $\beta=0$ 和 $\beta=0.75$ 下各跑一次，采样器为 `dep_round_v1`）：

| 臂 | `zo_mode` | 聚合配置 | 它回答什么 |
|---|---|---|---|
| A | `on` | `mean_all` | HO-FL 基线 |
| B | `off` | `fo_diluted` | A 减 B 是零阶更新的净贡献（H1） |
| C | `on` | `fo_mean` | C 减 A 是去掉稀释的收益（H1a） |
| D | `off` | `fo_mean` | 只做部分层一阶训练能达到的水平；C 减 D 是零阶更新在没有一阶报告的层上的贡献 |
| A×2、A×3 | `on` | `mean_all`，`server_lr` 取 2 和 3，仅 $\beta=0$ | 全局放大步长能否复现 C 的收益 |
| FedAvg、DeComFL | 不适用 | `mean_all` | 上界与下界，直接取自 E0 |

同时记录每个臂每轮的前向当量（§6.7），以便给出"准确率对计算量"的关系。D 臂的计算量最低（Qwen 三档为 12.14、9.29、6.43，HO-FL 为 15.0 [算]）。

**E3 的设置**：160 轮，故障窗口为第 41 到 80 轮，`evaluate_every=4`，开启 `per_class_eval`。每个配置同时汇报资源指标（每轮上行字节数、前向当量、各档参与台数），因为 RQ3 的比较以设备资源不增加为前提。系统配置：

| 代号 | 调度 | 聚合 | 角色 |
|---|---|---|---|
| B1 | `dep_round_v1`，$\beta=0$ | `mean_all` | HO-FL 均匀采样 |
| B2 | `dep_round_v1`，$\beta=0.75$ | `mean_all` | HO-FL 默认 |
| B3 | `dep_round_v1`，$\beta=0.75$ | `robust_all` | 在 HO-FL 上直接套用通用稳健聚合 |
| B4 | `dep_round_v1`，$\beta=0$ | `robust_all` | 同上，均匀采样 |
| B5 | `tier_equalised`，$\beta=0.75$ | `mean_all` | 只做档内均衡 |
| B6 | `dep_round_v1`，$\beta=0$ | `fo_mean` | 高效但没有保护 |
| P | `tier_quota 3,1,2`，`rotate` | `quorum_robust` | 本研究提出的配置（RHO-FFT） |
| P 的消融 | 把 `rotate` 换成 `fixed`；把 `fo_quorum` 设为 1；把 `geomed` 换成 `mean` | | 各组件的作用 |

故障条件：无故障；`label_noise`（`rate=1`，`map=shift`）；`lr_scale`（`scale=20`）；`sign_flip`（`scale=1`）。不可靠设备的角色：`preferred_t1`、`ordinary_t1`、`t3`，各 1 台。在 OPT 上跑完整网格；在 Qwen 上跑 B1、B2、B3、B6、P 乘以无故障和两种故障（`lr_scale`、`label_noise`），角色取 `preferred_t1`，其余组合视 OPT 的结果由研究负责人决定。可选基线：按入选概率做逆倾向加权的几何中位数，作为 Ma 等方法思路的简化实现，报告中须如此标注，不称为复现。

**E4 的设置**：在 B2 和 P 两种配置上，比较 `all_or_nothing`、`top_first`、`scarcity_first` 三种上行策略。截止时间取三档，使 T1 整份报告的按时比例约为 90%、60%、30%。先看无故障情形，再加一种故障（`lr_scale`）。输出各层的按时一阶宽度，以及准确率对累计模拟时间的曲线。有截止时间时，`mean_all` 的含义是除以 $K$（缺失分量按零计），同时报告除以到达数（`agg_weight=count`）的版本作为常见做法的对照。

---

## 8 里程碑、验收标准与决策关卡

| 阶段 | 内容 | 验收标准 | 关卡 |
|---|---|---|---|
| P0 基础设施与回归 | 提交上游基线并补 `.gitignore`，建工作分支；随后的一次提交只做插桩（最终参数哈希、状态保存与加载）；在目标 GPU 上记录黄金值；按 §6.1 重排主循环；`reports`、`aggregation`（默认路径）、调度器包装；测试框架；E0 | I1 和 I2 成立；E0 在容差内；不需要 GPU 的测试全部通过 | **G0**：通过则继续；任何一项不成立则停下报告 |
| P1 测量 | `supply`、`probes`、`zo_mode`、`tier_equalised` 与 `tier_quota` 调度器、均值类聚合配置（`fo_mean`、`fo_diluted`）、`server_lr`；E1a、E1b、E1c | 附录 A 的数值被测试复现；探针的只读性通过；E1 的表和 CSV 齐全 | **G1**：停下，提交 H1 与 H1a 的结果。论文的叙述取决于这一步，等研究负责人确认后再继续 |
| P2 故障与韧性 | `faults`、稳健统计量、法定数切换、`schedule_file`、`dose`；附录 B 的合成测试；E2（先 OPT），E3（先 OPT） | 合成测试通过；E2、E3 的 OPT 结果齐全 | **G2**：停下，提交 OPT 结果和 Qwen 网格的 GPU 小时估算，等确认后跑 Qwen |
| P3 截止时间 | `links`、上行策略、模拟时间；E4；`analysis` 完成 | 链路模块的单元测试通过；E4 结果齐全；全部图表的 CSV 可一键重新生成 | 提交总报告 |
| P4 结构性稀缺与裁判 | 放宽各档台数；留出数据；窗口内的留一来源评估 | 另行约定 | **G3**：开始前需要研究负责人批准 |
| P5 视觉语言模型 | 新的 `Partition` 子类和数据管线 | 另行约定 | 同上 |

**P0 的黄金值流程**：

0. 环境按仓库 README 搭建（Python 3.11，PyTorch 2.6.0，CUDA 12.4，依赖版本以 `pyproject.toml` 为准）。模型和数据集按 `SPECS` 与 `DATASETS` 中固定的修订号自动下载。
1. 仓库准备。先用 sha256 确认工作区中的上游拷贝未被改动，再把它原样提交为基线，提交信息写明来源（`HKU-WILL-Lab/ho-fl@ca74bed9`）。补一个 `.gitignore`，内容取自上游（忽略 `__pycache__/`、`.venv/`、`outputs/`、`cache/`、`data/`、`models/`、`checkpoints/`、`*.pt`、`*.safetensors` 等），并加上 `.DS_Store`。上游的 `README.md` 保留，在开头加一段说明本仓库的来源和用途。之后建工作分支。
2. 插桩提交。只加两样东西：`summary.json` 中的 `final_parameter_sha256`（对 `state` 调用 `tensor_hash`），以及默认关闭的状态保存。
3. 在目标 GPU 上，对下面三个冒烟配置各运行两次，并运行一次 `ho-fl-replay`：
   ```bash
   ho-fl-train --model opt --dataset sst2 --method hofl --beta 0.75 --rounds 3 --evaluate-every 3 --seed 42 --output outputs/golden/hofl-a
   ho-fl-train --model opt --dataset sst2 --method fedavg --rounds 3 --evaluate-every 3 --seed 42 --output outputs/golden/fedavg-a
   ho-fl-train --model opt --dataset sst2 --method decomfl --rounds 3 --evaluate-every 3 --seed 42 --output outputs/golden/decomfl-a
   ```
4. 两次运行逐位相同时，把逐轮 `task_loss`、评估指标和最终哈希存为黄金值（放在 `tests/golden/`，并注明 GPU 型号和驱动版本，因为黄金值与硬件有关），此后每次重构提交都要与之相同。两次运行不相同时，记录最终参数的最大绝对差，I1 改为容差模式并在报告中说明。
5. 没有 GPU 的环境下，请完成所有不需要 GPU 的部分，准备好脚本，并在报告中列出尚未执行的 GPU 步骤。不要用推测的数值代替。训练路径只通过 `cfg.device` 使用设备，可以尝试用 `--device cpu` 在 OPT-125M 上跑同样的三个冒烟配置，在同一台机器上比较重构前后的结果，作为临时回归；这条路径未经验证，跑不通就在报告中说明原因。它不能替代目标 GPU 上的黄金值。

**常用命令示例**（P1 之后）：

```bash
# C 臂：均匀采样，只在一阶报告之间平均
ho-fl-train --model qwen --dataset ag_news --partition dirichlet_05 --method hofl --beta 0 \
  --agg-electorate fo --agg-weight count --fo-quorum 1 --layer-stats --seed 42 --output outputs/e1c/qwen-C-b0-s42

# RHO-FFT：配额轮转加法定数切换的稳健聚合，带一个故障规格
ho-fl-train --model qwen --dataset ag_news --partition dirichlet_05 --method hofl \
  --sampler tier_quota --tier-quota 3,1,2 --in-tier rotate \
  --agg-electorate fo --agg-stat geomed --fo-quorum 3 --thin-policy all \
  --fault-spec experiments/faults/lr_scale_preferred_t1.json --evaluate-every 4 \
  --seed 42 --output outputs/e3/qwen-P-lrscale-s42
```

---

## 9 交付物与报告格式

- 代码：一个分支，提交粒度小。新模块带单元测试。
- 配置：`experiments/` 下每个实验一个目录，含 JSON 配置、故障与链路规格、网格脚本。网格脚本为每次运行生成唯一的输出目录，并跳过状态为 `COMPLETED` 的运行。
- 报告：每个里程碑一份 Markdown，放在 `docs/reports/`，用中文撰写，术语和标识符保留英文。内容依次为：
  1. 本阶段改了什么（文件与提交）；
  2. 运行了哪些测试，结果如何；
  3. 运行清单（目录、配置要点、种子、状态）；
  4. 结果表，每条假设对应的结论写"支持""不支持"或"无法判断"，并给出数字；
  5. 与本文件不一致之处、未解决的问题、需要研究负责人决定的事项；
  6. 已用 GPU 小时数和下一阶段的估算。

---

## 10 已知陷阱

1. `torch.median` 在元素个数为偶数时返回中间两个值里较小的一个。$K=6$ 时这会给零均值输入带来系统性的负偏移。`cwmed` 请取中间两个值的平均，并写测试覆盖偶数情形。
2. `zo_mode=off` 时不要把前缀参数的 `.grad` 设为零张量。那样 AdamW 仍会对前缀施加权重衰减，`replay.py` 的轨迹也会改变。保持 `.grad` 为空即可。
3. `hybrid_step` 要求前缀非空，分界覆盖只能取 `SPECS[model][3]` 中的值。
4. `ClientStream.next()` 会推进客户端的数据流。探针、干净孪生和任何重复训练都要复用记录下来的批次下标。
5. `RoundDirections` 会缓存方向张量。现有代码在每个客户端结束后释放，重构后请保持。
6. `append_jsonl` 和 `write_json` 拒绝 NaN 与无穷大，`classification.evaluate` 断言 logits 有限。压力条件下模型可能发散。`on_nonfinite=stop` 时请在写日志之前检测，记录发散轮次，把状态记为 `DIVERGED`，并保留最后一次有限的评估结果。发散是一种结果，汇总时按"崩溃"处理。
7. `run` 拒绝写入非空目录。网格脚本需要处理中断后的续跑。
8. manifest 的缓存路径不包含各档台数和分界分配方式。任何变体都请生成独立的 manifest 文件并通过 `--manifest` 传入，避免与默认缓存混用。
9. 不要把代码放在名为 `data/`、`models/`、`local/`、`checkpoints/` 的目录下，也不要用 `proxy` 或 `mirror` 开头的文件名，补上 `.gitignore` 之后它们会被忽略。保存的状态放在 `outputs/` 下。
10. 默认每 10 轮评估一次，对 40 轮的故障窗口太粗。E3 和 E4 使用 `evaluate_every=4`。
11. `selected` 的顺序影响浮点累加的结果。默认路径保持采样器返回的顺序。
12. `run` 中 `sampler.dr.p[cid]` 被用来写 `ledger.jsonl`，包装调度器后需要保留这一信息。
13. 非 `hofl` 方法的 `Backend.cuts` 分别是全 0（`fedavg`、`fedprox`）和层数（`decomfl`）。按 §6.5 统一处理阶数，聚合器即可同时支持这些方法。
14. 聚合单元在 LoRA 下是"同一层的四个张量"，用 `model.block_index(name)` 分组。
15. SQuAD 路径不在本研究范围内。请保证它没有被改坏（能通过导入和一次极短的冒烟运行即可），不为它实现新功能。

---

## 附录 A 已核对的参考数值（供单元测试使用）[算]

Qwen2.5-1.5B，三档各 10 台，$K=6$，$q=2$。解析量与分界落在哪些设备上无关。分布量来自仓库采样器 200 个种子乘以 160 轮的模拟，测试时请留出容差（概率值正负 0.02）。`tier_quota` 两列对 `rotate` 和 `random` 两种档内规则都成立。

| 指标 | `dep_round_v1` $\beta=0$ | `dep_round_v1` $\beta=0.75$ | `tier_equalised` $\beta=0.75$ | `tier_quota 3,1,2` | `tier_quota 4,1,1` |
|---|---|---|---|---|---|
| S1 期望宽度 | 2.0 | 5.0 | 5.0 | 3（每轮） | 4（每轮） |
| S2 期望宽度 | 4.0 | 5.5 | 5.5 | 4 | 5 |
| S3 期望宽度 | 6.0 | 6.0 | 6.0 | 6 | 6 |
| S1 单设备最大份额 | 10% | 16% | 10% | 10% | 10% |
| S1 份额最高 6 台合计 | 60% | 96% | 60% | 60% | 60% |
| S1 的 $N_{\mathrm{eff}}$ | 10 | 6.49 | 10 | 10 | 10 |
| S2 的 $N_{\mathrm{eff}}$ | 20 | 7.81 | 11.98 | 16 | 14.71 |
| S3 的 $N_{\mathrm{eff}}$ | 30 | 9.23 | 14.12 | 25.71 | 20 |
| 单设备最大在场率 | 20% | 80% | 50% | 30% | 40% |
| S1 宽度为 0 的轮次比例 | 6.4% | 0 | 0 | 0 | 0 |
| S1 宽度为 1 的轮次比例 | 26.5% | 0.01% | 0.02% | 0 | 0 |
| S1 宽度不超过 2 的轮次比例 | 69.2% | 0.4% | 0.4% | 0 | 0 |
| HO-FL 界中的偏差项 $N\sum_i(p_i-1/N)^2$ | 0 | 2.25 | 1.125 | 0.167 | 0.5 |
| 采样加权的 $d_{\mathrm{ZO}}/q$ | 311,296 | 194,560 | 194,560 | 285,355 | 233,472 |

均匀采样下 T1 入选台数的分布与超几何分布一致（0 到 6 台的概率依次为 0.0653、0.2611、0.3672、0.2304、0.0672、0.0085、0.0004），可作为解析参照。OPT-125M 的份额、在场率、宽度和概率与上表相同，只有最后一行不同（73,728、46,080、46,080、67,584、55,296）。

LoRA 规模与载荷（由 LoRA 形状推算，运行时请以 `config.json` 的 `bottom_dimensions` 为准；不一致时报告）：

| 量 | Qwen2.5-1.5B | OPT-125M |
|---|---|---|
| 每层可训练参数 | 38,912 | 24,576 |
| 全部可训练参数 | 1,089,536 | 294,912 |
| 三档的零阶维度 | 311,296 / 622,592 / 933,888 | 73,728 / 147,456 / 221,184 |
| 三档的一阶上行字节数 | 3,112,960 / 1,867,776 / 622,592 | 884,736 / 589,824 / 294,912 |
| T1 设备在 S1 上的一阶字节数 | 1,245,184 | 294,912 |
| 头部与零阶标量字节数 | 18 + 40 | 18 + 40 |
| 每轮前向当量，HO-FL（$E=5$，$q=2$，反向与前向耗时比 2） | 15.0 / 15.0 / 15.0 | 15.0 / 15.0 / 15.0 |
| 每轮前向当量，`zo_mode=off` | 12.14 / 9.29 / 6.43 | 12.5 / 10.0 / 7.5 |
| $\sqrt{q/d}$ | 0.0025 / 0.0018 / 0.0015 | 0.0052 / 0.0037 / 0.0030 |

HO-FL 论文中的参考结果 [核]：表 4（Qwen/AG News）在 IID、$\alpha=1$、$\alpha=0.5$、$\alpha=0.1$ 下，$\beta=0$ 为 86.75、86.22、85.51、83.39，$\beta=0.75$ 为 87.91、87.26、86.69、82.66；$\alpha=0.1$ 时 $\beta=0.75$ 的标准差为 3.07。

---

## 附录 B 理想化模型下的聚合行为（供合成测试使用）[理想]

模型：维度 20,000；$K=6$；正常一阶分量为 $\sqrt{c}\,\hat s+\sqrt{1-c}\,\xi_j$，$\hat s$ 为单位信号方向，$\xi_j$ 为相互独立的单位随机方向，$c=0.4$；零阶分量为范数 $r=0.7$ 的独立随机方向；不可靠分量为一份正常分量乘以 $-1$（记为 `flip1`，对应 `sign_flip`、`scale=1`）或 $-20$（记为 `flip20`）。"进度"定义为聚合结果在 $\hat s$ 上的投影除以 $\sqrt{c}$，1.0 表示一份正常一阶报告的信号强度。下表是 20 次重复的平均，几何中位数作为稳健统计量。

无故障，$n$ 份一阶分量加 $6-n$ 份零阶分量：

| $n$ | `mean_all` | `fo_mean` | `robust_all` | `quorum_robust` |
|---|---|---|---|---|
| 1 | 0.17 | 1.00 | 0.12 | 0.12 |
| 2 | 0.33 | 1.00 | 0.28 | 0.28 |
| 3 | 0.50 | 1.00 | 0.47 | 1.00 |
| 4 | 0.67 | 1.00 | 0.68 | 1.00 |
| 5 | 0.83 | 1.00 | 0.86 | 1.00 |

`robust_all` 的进度随零阶分量的范数变化：$n=2$ 时，$r=1.0$ 为 0.36，$r=0.5$ 为 0.21。

一份不可靠的一阶分量加 $m$ 份正常一阶分量加 $5-m$ 份零阶分量：

| 故障 | $m$ | `mean_all` | `fo_mean` | `robust_all` | `fo_robust` | `quorum_robust` |
|---|---|---|---|---|---|---|
| `flip1` | 0 | −0.17 | −1.00 | −0.12 | −1.00 | −0.12 |
| `flip1` | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `flip1` | 2 | 0.17 | 0.33 | 0.16 | 0.60 | 0.60 |
| `flip1` | 3 | 0.33 | 0.50 | 0.37 | 0.71 | 0.71 |
| `flip1` | 4 | 0.50 | 0.60 | 0.62 | 0.77 | 0.77 |
| `flip20` | 0 | −3.33 | −19.99 | −0.13 | −19.99 | −0.13 |
| `flip20` | 1 | −3.16 | −9.49 | 0.00 | −9.49 | 0.00 |
| `flip20` | 2 | −3.01 | −6.02 | 0.18 | 0.67 | 0.67 |
| `flip20` | 3 | −2.83 | −4.24 | 0.41 | 0.77 | 0.77 |
| `flip20` | 4 | −2.67 | −3.20 | 0.66 | 0.82 | 0.82 |

读法：不区分阶数的稳健统计把影响限制住了，但进度停留在被稀释的水平；只看一阶分量的稳健统计在 $m=2$ 和 $m=3$ 时进度约为前者的 2 到 4 倍，$m$ 更大时差距缩小，在 $m\le1$ 时完全失去保护；按法定数切换在每个 $m$ 上取较好的一支。

合成测试的要求：按上述模型生成报告（固定随机种子），调用 `aggregation.aggregate`，断言 (1) 无故障时 `mean_all` 的进度与 $n/K$ 相差不超过 0.03，`fo_mean` 在 $n\ge1$ 时与 1 相差不超过 0.05；(2) `flip20` 下 `mean_all` 的进度小于 −2，`robust_all` 的进度绝对值小于 1，`fo_robust` 在 $m\le1$ 时小于 −5；(3) `quorum_robust` 在 $m\le1$ 时与 `robust_all` 的结果相同，在 $m\ge2$ 时与 `fo_robust` 的结果相同。这些数字描述的是聚合规则在理想输入上的行为，不是对真实训练结果的预测；真实数据上的结论以 E2 为准。

---

## 附录 C 与 2026-11-19 草案的差异

研究负责人此前的草案（`resilient-hybrid-order-federated-fine-tuning.md`）与本文件不一致的地方，以本文件为准：

1. 草案的主线是"轮换加重点复核"。本文件把主线前移到逐层一阶供给的测量，并把聚合规则纳入设计。原因：轮换会消除触发复核所依赖的集中度信号，两步放在一起自相抵消；单轮单段的复核统计功效低。复核保留为 P4 的可选扩展。
2. 草案中"96% 的一阶更新来自 6 台设备"需要配上参照值：任何对 T1 设备一视同仁的调度下这个比例也有 60%。本文件把依赖分成结构性的部分（显存分档决定）和调度引入的部分。
3. 草案的轮转依据是"最近 20 轮按时贡献最少"。本文件默认按被选次数轮转，理由见 §6.3。
4. 草案的良性故障是"跳过本地更新但按时返回"，在等权平均下只相当于把步长缩小到 $5/6$，过于温和。本文件补充了 `label_noise`、`lr_scale`、`stale`。
5. 草案中的无线部分只作为轨迹回放出现。本文件给出一条可检验的机制（H4）。
6. 草案的视觉语言模型实验推迟到 P5。代码库目前只支持因果语言模型。

---

## 附录 D 参考文献与链接

- HO-FL：<https://arxiv.org/abs/2609.39074>；代码：<https://github.com/HKU-WILL-Lab/ho-fl>
- HO-SFL：<https://arxiv.org/abs/2603.14773>
- HERON-SFL：<https://arxiv.org/abs/2601.09076>
- ElasticZO：<https://arxiv.org/abs/2501.04287>
- CyBeR-0：<https://arxiv.org/abs/2406.14362>，<https://arxiv.org/abs/2502.00193>
- SALF：<https://arxiv.org/abs/2403.18375>
- FedRA：<https://arxiv.org/abs/2311.11227>
- ZorBA：<https://arxiv.org/abs/2603.04436>
- LASA：<https://arxiv.org/abs/2409.01435>
- Karimireddy 等，Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing：<https://arxiv.org/abs/2006.09365>
- Malinovsky 等，Byzantine Robustness and Partial Participation Can Be Achieved at Once：<https://arxiv.org/abs/2311.14127>
- Ma 等，Participation Bias Correction for Byzantine-Robust Federated Learning Under Heterogeneous Client Availability，Electronics 15(20):4746，2026
- TiFL：<https://arxiv.org/abs/2001.09249>
- BaFFLe：<https://arxiv.org/abs/2011.02167>
- FedAuto：<https://arxiv.org/abs/2512.22035>

---

## 附录 E v1.1 相对 v1.0 的改动

1. 项目和方法定名为 RHO-FFT，取代工作代号 SA-HO。
2. 代码改在新仓库 `GuangLun2000/RHO-FFT` 中开发。§0 增加仓库现状，§8 的 P0 增加"提交上游基线并补 `.gitignore`"一步，I8 和 F13 相应调整。
3. 新增 §1.4，写明落脚点（显存分档的终端与边缘设备）和结论适用的两条假设；低轨卫星场景留给后续工作。
4. 调度器接口预留 `available` 参数。
5. P0 增加无 GPU 环境下的临时回归做法（`--device cpu`）。
6. 新颖性边界表增加 FedAuto。
