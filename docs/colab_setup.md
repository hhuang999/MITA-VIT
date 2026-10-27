# Colab CLI 接入（2026-10-27）

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
