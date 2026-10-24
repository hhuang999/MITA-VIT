# 实验日志

尚未开展 RQ 实验。环境验证单独记录于 `experiments/2026-10-24_environment/`；这不构成模型准确率或研究结论。

## 2026-10-24 environment

- 目的：P0 环境验收，检查预训练 DeiT-S 能否用本地缓存完成 CUDA 推理。
- 设置：`scripts/check_environment.py`，timm 默认 eval 预处理，Imagenette 验证集第一张图，batch=1。
- 在哪跑：本地 RTX 4060 Laptop GPU；GPU 推理未单独计时。
- 结果：输出 1×1000 且全部有限，预测与标签同为 ImageNet 类别 0，峰值 CUDA 已分配内存约 97.2 MiB；结构/CPU 前向测试 1 项通过（pytest 总计 45.36 秒，含导入）。
- 异常：初次安装因 Windows 沙箱拒绝写入可执行启动器中断，已在沙箱外从缓存重装并通过依赖校验；验证未发现运行错误。ImageNet 权限 403 单独留作待办。
- 下一步：检查点 1 确认后进入 P1；不把单图预测作为模型准确率估计。
