# 赛事依据与待确认项

核对日期：2026-10-03，时区 Asia/Shanghai。

| 来源 | 已确认信息 | 工程中的处理 |
|---|---|---|
| [用户提供的公众号](https://mp.weixin.qq.com/s/rYRzx9p0urcbej37e533sA) | LingBot-VLA 2.0；Aloha-AgileX 50 个任务；每任务 50 条 clean 训练数据；clean/randomized 各 100 次测试；提交代码、权重、结果、说明 | 固定 50 项清单和 50 条 episode；拒绝 randomized 数据清单；保留 100 次评测入口 |
| [AMD 算力规则](https://github.com/AMD-DEV-CONTEST/Embodied-AI-Challenge-AMD-Platform-2026-09/blob/main/Radeon-Cloud-User-Guide/AMD_VLA_Contest_Compute_Rules.md) | 指定基础模型全参数训练；4/8 GPU；第一阶段 600 credits、第二阶段 1,600 credits；最终统一评测 | 默认 full SFT；LoRA 不作为本项目参赛方案；自测与官方成绩分开 |
| [AMD 指定的 ROCm 复现仓库](https://github.com/ZiguanWang/Robotwin-radeon-cloud/tree/27b2f6fcb6c715e2cbb6db4431381f87c94b86ed) | 固定源码、镜像和数据版本；原生 clean 数据挂载；完整 teacher；MPLib 评测 | 对源码 revision 与脚本哈希做显式校验 |
| [LingBot 官方代码](https://github.com/Robbyant/lingbot-vla-v2/tree/951475ae1b1d87553e7dc47c97b53a3d695c0d13) | 通用示例使用 clean + randomized；有多视角/时间一致增强和统计计算接口 | 重新构建 clean-only 清单和归一化统计；复用增强实现 |

公众号信息属于赛事介绍。天池详情页的正文未能完整读取，因此本工程不声称已经核实所有竞赛条款。

## 时间

- 初赛提交：2026-10-26 24:00，即 2026-10-27 00:00。
- AMD 算力窗口结束：2026-10-26 20:00。应在此之前导出、下载、校验模型，保留提交缓冲。
- 入围公布：11 月 6 日；上海决赛：11 月 13—15 日。按公众号安排，最终以官方通知为准。

## 资料之间的差异

公众号描述选手执行各任务 100 次并提交；较新的 AMD 算力规则描述最终 `100 task × 100 episodes` 由赛事组统一执行。因此目前优先保留可复现的候选模型和低成本自测证据，最终自测提交次数、文件格式需要报名后核实。

AMD 额度规划给出的 8 卡 10k 训练约 61.4 小时、100 项 × 10 次评测约 5 小时；固定版本复现指南的容量测试约 24.682 秒/step，完整自测实测约 15 小时 12 分。`plan` 使用后者作偏保守的估算；实际吞吐需要在自己的实例上测量，不能直接按官方估算消耗到额度上限。

## 获得账号后需要核实

1. 天池最新完整规则、训练数据允许范围及图像增强限制。
2. 是否必须提交选手侧 100 次结果，还是提交候选模型等待统一评测。
3. 官方计分方式、成功率聚合方式、seed、动作长度、精度、expert seed 筛选与规划器设置。
4. 权重与材料的命名、文件体积、托管方式、许可证及披露要求。
5. AMD credits 实际生效时刻、过期时刻、存储上限和当前镜像版本。

在以上字段确认前，所有评测只标记为自测。无需等它们全部确认才能运行本地工程测试，但不能据此宣称已完成正式提交。
