# 实验与提交记录

状态：等待 AMD 算力。尚无训练模型、任务成功率或排名。

## 待验证假设

在仅使用 clean 演示的约束下，三路相机及当前/未来帧共享增强参数，可以降低视觉扰动导致的策略退化。复用官方 `data.image_augment`，保留完整 depth/video teacher。增强可能降低颜色敏感任务的效果，需观察 `blocks_ranking_rgb` 等任务，不能先验认定一定提升。

## 受控实验

| 项目 | baseline | robust |
|---|---|---|
| 初始化 | 指定基础模型 | 相同 |
| 数据 | clean，50 × 50 | 相同 |
| 归一化 | 仅 clean 重算 | 同一统计 |
| teacher / optimizer | 完整 / AdamW | 相同 |
| steps / batch / seed | 10k / 256 / 42 | 相同 |
| 图像增强 | false | true |
| 自测 | 同一环境与协议 | 相同 |

先检查全部任务能否运行，再逐步增加 episode 数。2 次/任务仅用于粗筛；不能凭 1 次成败决定方案。候选比较应报告成功次数、分母、每任务区间、clean 与 randomized 宏平均和两者的差，不只展示表现好的任务。

这是工程配方实验，尚不能宣称算法优于基线。若预算只允许一次训练，完成 robust 配方和完整复现记录，不虚构消融实验。

## 实验记录字段

`run / recipe / base_model_revision / source_revisions / data_fingerprint / norm_sha256 / steps / batch / optimizer / seed / precision / planner / expert_check / episodes / credits / wall_time / peak_vram / checkpoint_identity / result_path`。

工具自动记录其已观测的信息。credits、真实峰值显存与官方 score 需要从实际平台和评测取得，未观测字段保留为空；资源估算不作为实测数据。

## 提交前验收

- [ ] 天池报名、组队与参赛资格确认。
- [ ] AMD 算力获批；镜像、版本与挂载通过自检。
- [ ] 50 × 50 clean 数据及归一化统计完成检查。
- [ ] 完成真实训练并保留日志。
- [ ] 导出完整权重，加载及推理通过。
- [ ] 每个任务在两种 setting 的自测证据完整。
- [ ] 根据最新规则确认是否需要选手侧各 100 次结果。
- [ ] 效果、消融、失败案例只填写实测数据。
- [ ] 按官方最终协议和格式完成复现说明。
- [ ] 权重下载、校验和与持久化备份完成。
- [ ] 核对第三方代码、模型许可和参赛披露要求。
- [ ] 天池提交成功回执保存。
