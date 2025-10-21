param(
  [string]$VenvPath = "$PSScriptRoot\..\..\.venv",
  [switch]$SkipPip = $false
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Write-Host "Project root: $root"

# 1) 创建虚拟环境
if (!(Test-Path $VenvPath)) {
  Write-Host "Creating venv at $VenvPath ..."
  python -m venv $VenvPath
}
& "$VenvPath\Scripts\Activate.ps1"
python -V

# 2) 安装依赖
if (-not $SkipPip) {
  Write-Host "Installing requirements ... (retry if network is unstable)"
  pip install --upgrade pip
  pip install -r "$root\requirements.txt" --timeout 120
}

# 3) 环境检测
& "$root\scripts\sanity_checks.ps1"

# 4) 一键构建（根据 config.yaml 的 roots 扫描）
Write-Host "Running full pipeline: ingest -> embed -> imgindex -> charindex -> graph"
python -m src.cli all

Write-Host "Done. Try a query:"
Write-Host 'python -m src.query_cli --mode hybrid --smart-query "莫维芬的所有记录"'
