# 实验日志

## 2026-10-25 hpc_gpu_queue

- 目的：执行用户已审阅并确认的首次 GPU 验收，通过后运行正式 RQ1。
- 设置：复用 `mita-p3-env`、原镜像和持久化挂载；资源改为已批准的 CPU 4 核 + A800 MIG 10 GB 单实例。应用接受配置，生成作业 `97895`。
- 结果：环境由“启动中”转为“等待”；平台返回 `Job's MIG GPU resource requirements not satisfied: 1 host`。尚无 SSH 连接，尚未运行实验；这不是探针或 CUDA 测试失败。
- 记录：`experiments/2026-10-25_hpc_queue/status.json` 保存平台返回的资源配置和排队原因，不含凭据。
- 下一步：保留原申请等待；已询问用户是否允许尝试其他单 GPU 规格（优先 MIG 20 GB），尚未改变资源。首轮验收通过前不会启动正式 RQ1。

## 2026-10-25 p3_adaptation

- 目的：迁移到按需 HPC 开发环境，准备首次 GPU 作业审阅。未启动远程 GPU 推理或正式 RQ1。
- 分支：`feature/hpc-adaptation`；锁定已接受的数据与模型 revision，支持持久化数据路径，新增固定 5,000 张前缀的 50 张验收配置。
- 本地验证：29 项 pytest 通过，10.42 秒；Git Bash shell 语法检查通过。
- 初次测试：27 通过、2 项 tmp_path fixture 错误，原因是沙盒临时目录不可写；使用项目忽略目录 `--basetemp=data/pytest-p3-20261025-1` 后全部通过。首次 Bash 语法检查被 Windows 命名对象权限阻止，在授权环境中执行只读语法检查通过。
- HPC 实测：CPU 2 核环境启动；镜像 Python 3.10.13 / torch 2.3.1+cu121，准备独立 Python 3.12；持久化 SSD 可用约 41 TB；GitHub、Hugging Face、PyTorch 下载站可达。
- 本地端到端预演：代码 `8d02692`；`configs/rq1_hpc_smoke.yaml`，ImageNet 固定子集前 50 张，全部 35 组，19.9 秒；k=197 预测一致率 100%、最大读取/恒等误差均为 0；没有非有限输出。记录在 `experiments/2026-10-25_hpc_local_smoke/`，明确标为 local rehearsal，不视为远程验收或正式研究结果。
- 本地 ready preflight：通过，权重 SHA-256 `1e747b4a8d0df2cfbd3c450e8c97685d867448ab0c2ddbfb34b6885f5cb23e5b`；记录在 `experiments/2026-10-25_hpc_local_preflight/`。
- 代码已推送到私有仓库；以同一分支的 Git bundle 经 SSH 同步，远程保留完整提交历史并将 origin 设为私有仓库，避免在计算节点存放 GitHub 凭据。
- 安装调整：PyTorch 源下载未锁定的 Pillow 12.3 极慢，且随后会被项目要求的 Pillow 13.0 替换；主动停止该安装，保留日志与已下载缓存。改为先从 PyTorch 源安装 torch/torchvision wheel（`--no-deps`），再从 PyPI 一并解析其运行依赖和项目锁定依赖，最后执行 `pip check`。版本不变。
- 网络调整：HPC 对 PyPI 主站同一 simple 页面约 4.09 秒，对清华镜像约 0.44 秒；通用依赖使用 `PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`，CUDA wheel 仍通过显式 PyTorch 官方索引。前两次慢速安装日志保留在远程工作目录，没有修改锁定版本。
- 一次 CUDA 运行库下载触发 ReadTimeout，已保留日志，使用 `PIP_DEFAULT_TIMEOUT=120` 复用缓存重试；安装完成，`pip check` 无依赖冲突。CPU 环境初次 preflight 遇到 `nvidia-smi` 占位文件的 Exec format error；已将此类可选驱动信息采集失败记为 unavailable，GPU 作业仍必须通过 CUDA 可见性检查和完整测试。
- 网络异常：HPC 直连 Hugging Face 的 curl、urllib 和 hub 客户端均出现连接重置，下载流程停止。使用临时、仅监听远程 loopback 的 SSH 转发连接本机已有代理后，官方源 HTTP 200，固定 revision 权重已缓存；ImageNet 也由 HPC 端通过此连接从官方源下载，未上传本地数据副本。作业阶段使用离线模式。
- 下载调整：Xet 分块传输过慢，保留未完成缓存到备份目录后，设置 `HF_HUB_DISABLE_XET=1`，用固定 revision 的 `snapshot_download(max_workers=14)` 预取 validation 分片。API 通过临时代理、官方 `.hf.co` CDN 直连；随后项目下载入口复用缓存生成数据集与索引。临时隧道曾断开一次，恢复后下载完成，未修改数据版本或内容。
- 远程最终验收：代码 `8946e29271f227138e2e15ff0f8e72c817686038`；Python 3.12.15，torch 2.7.0+cu128，torchvision 0.22.0+cu128，timm 1.0.30；环境 preflight、assets preflight 和 pip check 全部通过，保存 72 项依赖清单。
- 数据结果：50,000 张、1,000 类各 50 张；14 个官方分片 SHA-256 全部一致，固定 seed=0 的 5,000 张全部解码。模型权重、标签与索引等 12 项跨机器核对全部通过。记录见 `experiments/2026-10-25_hpc_environment/`；`loader_smoke.performed=false`，未执行远程模型推理。
- 收尾：临时 SSH 代理隧道已关闭。原浏览器任务空间消失后，经用户明确同意重新打开平台，仅停止 CPU 环境 `mita-p3-env`；平台显示“退出”，随后关闭任务空间。代码、独立解释器、数据与缓存均位于持久化 SSD。
- 下一步：用户审阅 `docs/hpc_first_job_review.md` 与 `hpc/first_job.sh`；确认后执行首次 GPU 验收，通过后再进行正式 5,000 张 RQ1。

P0/P1 实现验收、P2 试跑与正式研究结果分别记录。

## 2026-10-25 imagenet_download

- 目的：下载 ImageNet validation，为 P4 的固定 5,000 张子集准备数据。
- 设置：HF revision `49e2ee26f3810fb5a7536bbf732a7b07389a47b5`；仅 14 个 validation 分片，共 6,693,093,726 字节。
- 在哪跑：本地；下载原始日志保留在忽略的 `data/imagenet_download.log` 和 `data/imagenet_prefetch.log`。
- 异常与处理：先启动 datasets 下载，随后并行预取。Windows 无 symlink 的缓存流程中观察到已完成分片被另一进程重复下载；已核对进程身份并终止原始下载器（退出码 1，为主动终止），保留已下载文件及预取进程。入口改为先下载快照，再直接从本地 Parquet 生成数据集，避免重复传输。
- 本地整理首次因生成的锁文件名过长报 WinError 206；缓存改为较短的 `data/hf/` 后重试，未重新下载分片。首次日志 `data/imagenet_prepare.log`、重试日志 `data/imagenet_prepare_retry.log` 均保留。
- 结果：50,000 张已保存；14 个源 SHA-256 全部一致；1,000 类各 50 张，类别顺序与 timm 一致（HF 的第 517 类名称 crane2 对应 timm 的 crane）。seed=0 的 5,000 张索引无重复且全部可解码，实际覆盖 994 类；32 张图片通过 CUDA 前向，输出为有限的 32×1000 张量。
- 验收代码提交：`8616cf2a2aba82e3d1efc9a2202e2417e655d835`；记录见 `experiments/2026-10-25_imagenet_download/`。原始分片 6.69 GB，保存后的数据集 6.71 GB；下载、缓存兼容性修复、整理和验收合计约 29 分钟。
- 下一步：停在检查点 3，待用户确认 P3 / P4。未开展 ImageNet 正式 RQ1，也未把数据加载检查计作正式恒等验收。

## 2026-10-25 rq1_pilot

- 目的：P2 完整 RQ1 试跑，检查曲线、随机对照和 K-means，准备检查点 3。
- 设置：`configs/rq1_pilot.yaml`；Imagenette 500 张，seed=0，DeiT-S，m=16，k={4,8,16,32,64}，随机种子 0–4；全层严格执行固定原模型说明书；FP32，batch=32。运行代码提交 `b436890859e2b95b15fac3f9cf037addac27e6aa`。
- 在哪跑：本地 RTX 4060；完整 35 组加恒等验收耗时 170.7 秒，不含启动/模型加载。
- 结果：grid 的五点一致率为 24.6%、46.4%、68.0%、78.4%、88.2%；随机均值为 0.32%、0.56%、2.96%、19.72%、51.88%；K-means 为 14.2%、40.4%、67.4%、79.6%、86.8%。grid 相对随机的 95% 配对区间全部大于 0。
- 可信度：原模型准确率 82.4%；500 张 k=197 预测完全一致，最大输出误差为 0；所有设置读取输出与原模型完全一致，无非有限值。
- 异常：无运行异常。k≤64 未达到 90% 的样本一致率，如实保留；未扩大 k 或改变分组。绘图首次标题被布局裁切，修正布局后重新导出并查看 PNG；未重跑模型。
- 记录：`experiments/2026-10-25_rq1_pilot/` 保存 17,500 条逐图结果、35 行设置结果、汇总、配对区间、索引、配置、日志、元数据及 PNG/PDF。
- 下一步：检查点 3 讨论，见 `docs/checkpoint_3.md`。建议保持设计进入 P3 / P4，ImageNet 5,000 张正式结果不由本次试跑替代。

## 2026-10-25 p2_unit

- 目的：检查 RQ1 配置继承、35 组设置、逐图配对汇总与置信区间，并回归核心探针。
- 设置：`python -m pytest -q --junitxml=experiments/2026-10-25_p2_unit/results.xml`。
- 在哪跑：本地 CPU / RTX 4060；22.71 秒。
- 结果：27 项通过；汇总测试覆盖乱序图片、种子标准差，以及重复、缺失、不同图片集合的拒绝。
- 异常：无。
- 下一步：运行 Imagenette 500 张的完整 RQ1 试跑。

## 2026-10-24 p1_validation

- 目的：P1 检查读取不改变原模型、k=197 全层执行恒等，以及原模型准确率。
- 设置：`configs/validate_p1.yaml`，DeiT-S 预训练权重、timm 默认 eval 预处理；Imagenette 160px 全部 3,925 张验证图，seed=0 排序，batch=32；说明书从原模型读取后固定。代码提交 `2b17f53`。
- 在哪跑：本地 RTX 4060；约 39.0 秒，不含模型加载。
- 结果：读取与执行均 3,925/3,925 预测一致，最大 logit 误差均为 0，KL=0；原模型及执行模型准确率均为 80.28%（3,151/3,925），按完整 1,000 类输出计算。
- 异常：无运行异常。Imagenette 的 10 类标签已逐项核对 timm ImageNet 元数据。此验收未覆盖仍待授权的 ImageNet 5,000 张。
- 记录：`experiments/2026-10-24_p1_validation/` 内保存配置、日志、结果 CSV、逐图预测、索引及元数据。
- 下一步：停在检查点 2，用四 token 手算例子讲解；确认后才进行 P2 小 k 试跑。

## 2026-10-24 p1_unit

- 目的：在真实数据验收前检查核心探针、模型接入、随机对照、指标与标签映射。
- 设置：`python -m pytest -q`，测试固定种子 0，模型集成测试不下载权重；含 CUDA 控制组。
- 在哪跑：本地 CPU / RTX 4060；最终运行 8.73 秒。
- 结果：22 项通过，详见 `experiments/2026-10-24_p1_unit/results.xml`。
- 异常：首次 18 项通过、1 项 fixture 建立失败，原因是沙箱不能建立 pytest 临时目录。该标签映射测试本来就不访问磁盘，移除了不需要的临时目录 fixture；首次记录保留在同目录 `initial.xml`。随后补充 CUDA 控制组测试并全部通过。
- 下一步：在预训练权重与 Imagenette 上验证只读无扰动及 k=197 全层恒等。

## 2026-10-24 environment

- 目的：P0 环境验收，检查预训练 DeiT-S 能否用本地缓存完成 CUDA 推理。
- 设置：`scripts/check_environment.py`，timm 默认 eval 预处理，Imagenette 验证集第一张图，batch=1。
- 在哪跑：本地 RTX 4060 Laptop GPU；GPU 推理未单独计时。
- 结果：输出 1×1000 且全部有限，预测与标签同为 ImageNet 类别 0，峰值 CUDA 已分配内存约 97.2 MiB；结构/CPU 前向测试 1 项通过（pytest 总计 45.36 秒，含导入）。
- 异常：初次安装因 Windows 沙箱拒绝写入可执行启动器中断，已在沙箱外从缓存重装并通过依赖校验；验证未发现运行错误。ImageNet 权限 403 单独留作待办。
- 下一步：检查点 1 确认后进入 P1；不把单图预测作为模型准确率估计。
