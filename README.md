# 稳境 RobustTwin

面向蚂蚁灵波 VLA 挑战赛的 **clean 数据训练、视觉泛化对照实验与可追溯评测工程**。

当前状态：工程已建立；模型尚未训练，真实成功率尚未测得。用户正在申请 AMD 赛事算力。本项目不会把流程测试、官方论文指标或其他队伍的模型成绩作为自己的参赛成绩。

## 项目方法

从赛事指定的 `robbyant/lingbot-vla-v2-6b` 基础模型开始，使用 Aloha-AgileX 的 50 个任务、每个任务 50 条 clean 数据做全参数 SFT。保留 current/future depth 与 DINO-video teacher、FSDP2 和 gradient checkpointing。

提供两个可控配方：

| 配方 | 数据 | 训练 | 图像增强 |
|---|---|---|---|
| `baseline` | 50 × 50 clean | 完整 teacher 全参数 SFT | 关闭 |
| `robust` | 相同数据 | 相同训练设置 | 开启官方多相机、当前/未来帧一致增强 |

核心实验问题：仅从 clean 演示学习时，一致的视觉增强能否提升 randomized 自测效果，同时保持 clean 效果。增强实现来自 LingBot；本项目贡献在于 clean-only 统计、受控实验、预算与环境校验、恢复失败检查、模型与评测证据绑定。效果和创新性评价需要真实实验支持。

## 先完成算力申请

1. 在 [天池赛事入口](https://tianchi.aliyun.com/competition/entrance/532514) 报名并建立队伍，取得队伍 ID。
2. 完成 [AMD 开发者计划注册](https://developer.amd.com.cn/login?source=sSEUZ7cAA)。
3. 加入官方赛事群，按 [AMD 申请入口](https://developer.amd.com.cn/contestgpu_apply01) 提交。当前官方指南要求每位队员填写，队伍指定一名队长。
4. 优先申请 **8 × W7900 48GB**。同一队伍只保留一条排队记录；4 卡也有配置，但 micro batch 16 的显存余量较小。

申请文本与时间安排见 [docs/AMD_APPLICATION.md](docs/AMD_APPLICATION.md)。官方规则与待核实项见 [docs/COMPETITION.md](docs/COMPETITION.md)。报名和申请尚未代为提交。

## 本地立即可用

Windows / PowerShell 7，项目自带的 `.venv` 已装好 PyYAML：

```powershell
$ErrorActionPreference = 'Stop'
.\.venv\Scripts\python.exe -m robustwin plan
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m robustwin train --run robust-10k --recipe robust
```

最后一条只预览参数。`prepare / train / export / evaluate` 加上 `--execute` 才在云端执行。项目不会在本地下载模型权重或自动消耗云端 GPU credits。

重新部署时，在 Python 3.12 环境安装 `requirements.txt`；运行全部 Parquet 集成测试时安装 `requirements-test.txt`。始终从项目根目录运行 `python -m robustwin`；模型训练依赖由赛事 AMD 镜像提供，PyYAML 只用于本项目的配置工具。

## 云端运行

使用 Radeon Cloud 的赛事 `robotwin` 镜像、`Dev` 资源池、`Persistent /workspace` 和 `Devzone` 模型挂载。参考 [AMD 指定的部署指南](https://github.com/ZiguanWang/Robotwin-radeon-cloud/blob/27b2f6fcb6c715e2cbb6db4431381f87c94b86ed/Reproduce_Guide.md)。实例内已有 `/RoboTwin`，无需另建嵌套 Docker。

将本项目的 `robustwin/`、`configs/`、`docs/`、`tests/` 和根目录说明文件放到 `/workspace/RobustTwin`。本地 `vendor/`、`sources/`、`.venv/` 无需上传。源码交付包 `dist/RobustTwin-source.zip` 可用于上传；它不包含模型和数据。

下面的命令在项目根目录运行，`/opt/robotwin-env/bin/python` 是镜像内 Python。它们也可直接在云端 JupyterLab Terminal 中使用。

### 1. 自检与数据准备

```text
/opt/robotwin-env/bin/python -m robustwin doctor --gpus 8
/opt/robotwin-env/bin/python -m robustwin prepare --gpus 8 --execute
```

自检核对 AMD 参考脚本哈希、三个 Git revision、50 个任务名、ROCm、模型与 teacher 文件。数据准备遍历全部 action/state parquet 行，检查 50 条 episode、14 维输入输出、有限数值及帧序号；检查三路视频文件存在，然后仅用通过检查的 clean 清单调用官方脚本重算归一化统计。视频内容不会逐帧解码，数据来源仍以官方挂载为依据。

输出目录：`/workspace/runtime/robustwin/prepared`。已有准备结果不会被覆盖；失败后的半成品需检查后处理，再重新准备。

### 2. 先做短训练，量测实际耗时和显存

```text
/opt/robotwin-env/bin/python -m robustwin train --run smoke-10 --recipe baseline --steps 10 --save-every 10 --gpus 8 --execute
```

即使只训练 10 步，官方训练器仍保存完整 DCP，约 70GB。该步骤检验 forward、backward、优化器与 checkpoint 写入；其模型不能作为有效成绩。完成后保留日志，核实短训练输出的用途，再人工处理不需要的模型文件，为正式训练留出空间。本项目不自动删除 checkpoint。

### 3. 正式训练

```text
/opt/robotwin-env/bin/python -m robustwin train --run robust-10k --recipe robust --steps 10000 --save-every 10000 --gpus 8 --execute
```

默认 global batch = 16 × 8 × 2 = 256。4 卡使用 `--gpus 4`，梯度累积自动改为 4。数据、模型、teacher、配方、版本以及状态会记录在训练目录。`baseline` 与 `robust` 在相同步数、种子和环境下构成增强开关的对照实验。

在不删除旧文件的前提下，每个计划保存的 DCP 预留 70GiB，导出与日志另留 20GiB。默认 10k 末尾保存一次，要求至少 90GiB 可用空间；**这意味着中途故障可能损失未保存的进度**。若取得更大持久盘，可用 `--save-every 2500` 定期保存，10k 训练预计需 300GiB。100GB 持久盘通常无法同时容纳多个训练实验和导出模型，必须先规划下载、备份或扩容。

有中间 checkpoint 时，可按原配方恢复，例如原训练使用 `--save-every 2500`：

```text
/opt/robotwin-env/bin/python -m robustwin train --run robust-10k --recipe robust --steps 10000 --save-every 2500 --gpus 8 --resume-step 2500 --execute
```

恢复要求配方、步数、卡数、保存间隔、数据和统计保持一致。训练入口在内存中修正官方“恢复失败后重新开始”的分支：加载失败立即报错。延长学习率计划需要单独设计实验，当前入口不允许把修改后的计划伪装成原实验恢复。

### 4. 导出可迁移模型

```text
/opt/robotwin-env/bin/python -m robustwin export --run robust-10k --gpus 8 --execute
```

输出：`/workspace/runtime/robustwin/models/robust-10k`，包含完整 HF 权重、训练配置、clean 归一化统计、机器人映射、训练记录和逐文件 SHA-256。导出后仍需真实推理验证；导出成功不等于任务成功。

### 5. 自测与报告

先跑每任务每 setting 2 次的初筛，用于检查全链路与大范围失败，不用于可靠比较模型：

```text
/opt/robotwin-env/bin/python -m robustwin evaluate --run robust-10k-screen --model /workspace/runtime/robustwin/models/robust-10k --episodes 2 --gpus 8 --execute
```

方向确认、剩余额度充足后，使用新的运行名，将 `--episodes` 改为 `10` 或 `20` 做候选比较。同一评测中断时追加 `--resume`；模型哈希或协议不同则拒绝复用旧结果。100 次入口也可执行，但其实际资源需求应单独测算，官方统一评测由赛事组安排。

评测固定使用 AMD 参考协议：**ROCm + MPLib、BF16、use_length=25、seed=0、expert_check=true、accept_expert_info_on_failure=true**。该协议与 LingBot CUDA/FP32 发布指标存在差异，不得直接横向比较。最终协议以赛事组要求为准。

结果位于 `runtime/evaluation/outputs/<run-name>/`：

- `report.html`：50 个任务的 clean/randomized 对照表。
- `results.csv`：成功次数、总次数、成功率、逐任务 Wilson 区间。
- `report.json`：完整性、两种 setting 的宏平均、性能差、模型与协议身份。
- `logs/`、`episode_info/`、`done/`：原始证据。

缺失任务不会填 0，也不会从分母中静默排除。只有某 setting 的 50 项全部完成，才计算其宏平均。未核实官方评分公式前，不生成所谓“官方总分”。

本地读取从云端下载的评测目录：

```powershell
.\.venv\Scripts\python.exe -m robustwin report --run-dir .\runtime\evaluation\outputs\robust-10k-screen --episodes 2
```

中途检查可加 `--allow-incomplete`，报告会保留缺失和无效项；不完整报告不能打包交付。

### 6. 打包候选材料

```text
/opt/robotwin-env/bin/python -m robustwin bundle --model /workspace/runtime/robustwin/models/robust-10k --evaluation /workspace/runtime/robustwin/evaluation/outputs/robust-10k-screen --out /workspace/runtime/robustwin/RobustTwin-candidate.zip
```

打包要求真实训练记录、完整模型分片、文件哈希一致、100 个任务/setting 自测条目齐全且绑定同一模型。包含代码、模型、评测原始日志与清单。低次数自测会如实标记，工程压缩包不代表已满足天池最终命名、上传体积或表单要求，也不会自动上传。

## 验收边界

本地验证覆盖数据元信息校验、配方生成、预算计算、恢复失败行为、结果解析、缺项拦截、模型分片与哈希、提交材料绑定。GPU 容量、全链路模型训练、仿真渲染、模型可加载性和真实成功率，必须在获批的 AMD 实例上验证。

完整实验记录表见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。代码依赖与归属见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

相关论文的官方 PDF、校验值、关键结果及对本项目的适用边界见 [docs/PAPERS.md](docs/PAPERS.md)。
