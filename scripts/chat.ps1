# 一键启动多轮 Chat（每轮自动 RAG）
param(
  [string]$Session = "",
  [string]$Config = "chat_config.yaml"
)
$cmd = @("python","-m","src.chat_cli","--config",$Config)
if ($Session -ne "") { $cmd += @("--session",$Session) }
Write-Host "Running: $($cmd -join ' ')"
& $cmd
