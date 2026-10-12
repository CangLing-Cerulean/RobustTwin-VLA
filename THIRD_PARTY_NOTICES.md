# 来源与归属

本项目的训练与推理依赖 LingBot-VLA 2.0，AMD 适配与仿真运行依赖赛事算力指南指定的 RoboTwin ROCm 镜像。

- [LingBot-VLA 2.0](https://github.com/Robbyant/lingbot-vla-v2)，本工程针对 AMD 指定版本 `951475ae1b1d87553e7dc47c97b53a3d695c0d13`。仓库提供 Apache-2.0 许可。模型权重使用须另核对对应 model card。
- [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin)，版本 `266f3aadf505a4f7fe9af0faa41a20f5f47cd123`。
- XPolicyLab：镜像指定 revision `c37109c500be67d0dea6b36bf7337bbd26e763cd`；以 RoboTwin 子模块实际地址和许可为准。
- [Robotwin-radeon-cloud](https://github.com/ZiguanWang/Robotwin-radeon-cloud)，版本 `27b2f6fcb6c715e2cbb6db4431381f87c94b86ed`。训练 teacher 配置与接口参数以该指南及脚本为依据。该参考仓库根目录未见明确 LICENSE，使用与再分发应核实许可；本项目的代码包不捆绑该仓库全部源码。
- PyYAML 6.0.2：配置解析依赖，沿用 AMD 参考版本。

`vendor/` 和 `sources/` 是本地调查用的公开源码与原文快照，不进入默认源码包和候选模型包。模型文件和完整仿真依赖需通过官方渠道取得。训练入口只在内存中对已校验哈希的官方训练代码做一次局部变换，使 checkpoint 恢复失败明确中止，云端原文件保持不变。

配置参数与底层增强实现不宣称为本项目原创算法；自有工程负责数据校验、实验控制、资源估算、错误拦截、证据汇总与交付校验。
