Param(
  [int]$Duration = 0
)

# Windows PowerShell 一键运行：通过 WSL 调用 run.sh
# 用法示例：
#   powershell -ExecutionPolicy Bypass -File run.ps1
#   powershell -ExecutionPolicy Bypass -File run.ps1 -Duration 10

$ErrorActionPreference = 'Stop'

# 脚本所在目录（应为 第五部分/rm_ws）
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

# 检查 wsl 可用
if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
  Write-Error '未找到 wsl.exe，请在 Windows 上安装并启用 WSL 后重试。'
}

# 将 Windows 路径转换为 WSL 路径
$WslPath = & wsl.exe wslpath -a "$ScriptDir"
$WslPath = $WslPath.Trim()

# 组装 run.sh 调用参数
$Arg = ''
if ($Duration -gt 0) { $Arg = " --duration $Duration" }

# 在 WSL 中切换目录并执行 run.sh
& wsl.exe bash -lc "cd '$WslPath' && bash run.sh$Arg"


