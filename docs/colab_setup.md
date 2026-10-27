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

授权后再验证账户和远程硬件，准备数据与环境。当前没有启动 Colab 作业。本地 GPU 禁用约束继续有效。已有 A800、RTX 4060 结果及 HPC 脚本保留；若 Colab GPU 不同于原 A800，不能将两者合并声称为同硬件实验。正式执行方案及硬件来源须单独记录。
