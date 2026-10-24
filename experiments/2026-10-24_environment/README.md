# P0 环境验收（2026-10-24）

- `environment.json`：`HF_HUB_OFFLINE=1` 时运行 `python scripts/check_environment.py` 的实际输出。
- `data.json`：数据集数量、Pet 测试集图片/trimap 配对检查及 ImageNet 访问状态。
- `requirements-windows.txt`：验收环境的安装快照，包含 Windows CUDA 包；跨平台安装使用仓库根目录的依赖文件。

`python -m pytest -q`：1 passed in 45.36s，退出码 0。
`uv pip check --python .venv/Scripts/python.exe`：57 个包兼容，退出码 0。

测试覆盖 timm DeiT-S 的层数、head 数、patch/CLS 数和 CPU 前向；真实图片脚本覆盖预训练缓存加载、默认预处理及 CUDA 前向。本阶段尚未实现或检验注意力探针，没有研究结果。
