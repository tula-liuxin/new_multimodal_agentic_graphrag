Param(
  [string]$Session = ""
)

$ErrorActionPreference = "Stop"

# 允许用户在项目根任意位置执行
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projRoot = Resolve-Path (Join-Path $scriptDir "..")

# 优先使用项目根的 Python
$python = "python"

# 切到项目根执行，避免相对路径错乱
Push-Location $projRoot

Write-Host "Launching agent chat (RAG_A + RAG_B orchestrator) ..."

if ($Session -ne "") {
  & $python -m src.agent_chat --config "chat_orchestrator.yaml" --session $Session
} else {
  & $python -m src.agent_chat --config "chat_orchestrator.yaml"
}

Pop-Location
