Param(
    [ValidateSet("dev","install","run")]
    [string]$Mode = "dev"
)
$ErrorActionPreference = "Stop"

# Resolve project root (repo root = parent of scripts/)
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $Here
Write-Host "Project root: $ProjectRoot"
Set-Location $ProjectRoot

# Prefer venv python if present
$Py = "$env:VIRTUAL_ENV\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }

function Invoke-McpCli {
    param([string[]]$Args)
    $mcpCmd = Get-Command mcp -ErrorAction SilentlyContinue
    if ($mcpCmd) {
        # Use the installed CLI entrypoint
        & mcp @Args
        return $LASTEXITCODE
    } else {
        # Fallback to python -m mcp.cli
        & $Py -m mcp.cli @Args
        return $LASTEXITCODE
    }
}

if ($Mode -eq "dev") {
    Write-Host "Starting MCP server in DEV mode..."
    Invoke-McpCli @("dev", "mcp/graphrag_server.py")
}
elseif ($Mode -eq "install") {
    Write-Host "Installing MCP server into client (e.g., Claude Desktop)..."
    Invoke-McpCli @("install", "mcp/graphrag_server.py", "--name", "GraphRAG (Local)")
}
elseif ($Mode -eq "run") {
    Write-Host "Running MCP server directly (stdio)..."
    & $Py "mcp/graphrag_server.py"
} else {
    throw "Unknown mode: $Mode"
}
