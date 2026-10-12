# 本地验收记录

日期：2026-10-03。环境：Windows、PowerShell 7、Python 3.12、PyYAML 6.0.2、PyArrow 21.0.0。

执行：`.venv/Scripts/python.exe -m unittest discover -s tests -v`

结果：**28 项测试通过，0 失败、0 跳过**。

覆盖范围：

- 50 个任务、每项 50 条 episode、三相机和数据文件完整性；重复任务、缺项和 randomized 名称被拒绝。
- 实际 Parquet 格式的合成数据扫描；NaN、重复帧、缺失 episode、总帧数不符被拒绝。
- 4/8 卡 global batch、磁盘估算与实验配方；增强对照保持其他配置一致。
- clean-only 归一化统计同时绑定机器人 YAML 和推理配置。
- 指定恢复失败时明确抛错，预览命令不启动 GPU 工作。
- 真实格式的评测日志解析；重复最终结果、错误分母和不一致百分比被拒绝。
- 缺失项不参与完整宏平均；clean/randomized 分开报告；逐任务 Wilson 区间保留小样本不确定性。
- Safetensors 文件结构、权重索引路径、分片存在性、模型哈希及模型与评测身份匹配。
- 候选压缩包包含模型、源码与评测证据，排除 vendor、虚拟环境。
- 本地已下载的 AMD 参考脚本及打过 AMD 补丁的 LingBot 源码与锁定哈希一致；训练入口可编译。

测试数据全部是临时目录中的明确合成夹具；未写入真实实验目录，不能作为机器人训练效果证据。

尚未验证：AMD GPU 训练、分布式通信、teacher 显存、SAPIEN 渲染、仿真交互、模型实际可加载性、真实成功率和天池提交。需要获批的云端实例、官方模型与数据完成这些验证。

源码包不含调查用 vendor，因此在另一台机器仅运行源码包测试时，来源快照校验项会跳过。安装 `requirements-test.txt` 才能运行 Parquet 集成测试。
