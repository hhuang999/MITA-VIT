# Colab CLI 接入（2026-10-27）

最新一次运行是已批准的V00–V03：单张L4 `mita-validation-20261027`，正式136设置全部完成并取回，实例已停止；当前服务器活动实例0、费率0、余额24.06 CU。结果见 [夜间补充验证总结](analysis/overnight_validation_2026-10-27.md)。下方是此前RQ1复核的记录。

本地CPU隔离必须使用CUDA_VISIBLE_DEVICES=-1，并先确认torch.cuda.is_available()为False；空字符串在本机Windows没有生效，曾导致3个CUDA单测误跑，已记录并纠正。本地测试使用 `.venv/Scripts/python.exe scripts/test_cpu.py -q`，Colab完整测试仍用普通pytest以覆盖CUDA。

## RQ1复核结束时的状态

全部42组RQ1已在同一张NVIDIA L4完成，结果已取回并核验。会话 `mita-rq1-l4-20261027` 已停止，服务器确认0活动实例、0 assignments、0.00 CU/小时，余额24.95 CU。下文A100和分块传输段落为此前尝试的历史记录，不是当前资源状态。

用户后来明确授权临时传输HF_TOKEN；据此使用HF-only文件在用户自己的L4实例下载固定ImageNet版本，校验14个源分片、模型和5000张索引哈希。令牌没有进入Git或日志；完成后停止实例。项目Python3.12.3、PyTorch2.7.0+cu128、timm1.0.30，45项测试通过；两次5000张k197一致率均100%、输出误差0。正式科学代码来源737f7d4，参数与原配置一致。

运行入口为 `colab/run_rq1.py`，依次执行 `rq1.yaml` 和 `rq1_k128.yaml`；环境目录通过 `MITA_ENV_RUN=2026-10-27_colab_environment_retry` 指定。Windows控制使用项目Python调用 `colab/cli.py`，内部采用WSL `--exec` 并过滤URL查询凭据。运行时可用 `colab/snapshot_run.py` 只读取回检查点；完成后用 `colab/export_results.py` 导出实验小文件。

CPU分析命令如下，输出必须是尚不存在的新目录：

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
.venv/Scripts/python.exe scripts/analyze_rq1.py experiments/2026-10-27_rq1_colab experiments/2026-10-27_rq1_colab_k128 experiments/2026-10-27_rq1_colab_combined
.venv/Scripts/python.exe scripts/make_figures.py experiments/2026-10-27_rq1_colab_combined --confidence-bands --title 'RQ1 | ImageNet, 5,000 images | NVIDIA L4'
.venv/Scripts/python.exe scripts/analyze_colab_rq1.py
```

结果及复核报告见 [rq1_colab_verification.md](analysis/rq1_colab_verification.md)。原A800/4060结果保留，本地GPU未启用，本轮未启动HPC。

## 安装与首次尝试（历史记录）

用户要求改用 Colab，并亲自完成 Google 授权。官方 CLI 当前支持 Linux/macOS，因此安装在已有 Ubuntu WSL 中，不修改项目 `.venv`。

- 官方项目：https://github.com/googlecolab/google-colab-cli
- 安装版本：`google-colab-cli==0.7.5`
- 环境：Ubuntu WSL，Python 3.12.3。
- 独立安装路径：`/home/hhuan/.local/share/google-colab-cli-venv`。
- 验证：`colab version` 输出 0.7.5；`pip check` 无依赖错误；`sessions --help` 正常。

在 PowerShell 中授权或查询会话：

```powershell
wsl -d Ubuntu -- /home/hhuan/.local/share/google-colab-cli-venv/bin/colab --auth oauth2 sessions
```

第一次执行会打印 Google 授权网址。用户在浏览器登录并同意授权，将 Google 显示的授权码直接粘贴回终端。该命令只查询会话，不创建计算实例。CLI 令牌由工具保存在 WSL 用户目录，不纳入项目或 Git。

OAuth 已验证，实例 `mita-rq1-20261027` 已分配 NVIDIA A100-SXM4-40GB，费率5.30 CU/小时，初始余额28.21 CU。Colab内核为Python3.13，项目另建Python3.12.3 venv；PyTorch2.7.0+cu128、TorchVision0.22.0+cu128、timm1.0.30已核验，45项pytest通过（10.62秒）。记录在 `experiments/2026-10-27_colab_environment`。

运行代码来自已提交Git bundle（`9397db1`）。使用 `colab/prepare_runtime.py` 准备环境；本次调度副本仅将输入bundle路径改为 `/content/mita-source-current.bundle`。`colab/package_assets.py` 对明确白名单中的20个本地数据/模型文件生成哈希和136个分块；`colab/upload_assets.py` 通过CLI上传并记录完成块；`colab/restore_assets.py` 复原并逐文件验证SHA-256；`colab/run_rq1.py` 运行原35设置与k128七设置。

自动审批拒绝HF令牌传输，当前采用已有本地资产上传，没有向Colab传输HF凭据。上传数据无需远程网络认证，正式推理采用离线模式。

本地GPU禁用约束继续有效。已有A800、RTX4060结果及HPC脚本保留；完整42设置统一在新A100上执行并另存。结束后必须先取回结果再运行 `colab --auth oauth2 stop -s mita-rq1-20261027`，核对服务器无本轮实例。

## 21:10 中断与恢复准备

首次实例在96个分块确认后消失，CLI服务器查询为0个活动实例、0计费，余额26.20 CU。没有正式RQ1结果。前述45项测试和环境记录保留，不能把新实例当作已有旧分块。

已向用户请求明确HF凭据传输授权；收到之前不上传、不重新申请计费实例。直接下载路径已准备为 `colab/download_assets.py`，它要求已授权的HF-only文件，比较14个源Parquet哈希、5000张抽样文件及模型哈希与旧实验一致。恢复环境通过 `MITA_ENV_RUN=2026-10-27_colab_environment_retry` 另存，保留首次环境记录；该脚本尚未运行。
