param(
  [string]$ProjectRoot = "."
)
$ErrorActionPreference = "Stop"

$utils = Join-Path $ProjectRoot "src/utils.py"
if (!(Test-Path $utils)) {
  Write-Error "Not found: $utils"
  exit 1
}
# backup
Copy-Item $utils "$utils.bak" -Force

# read
$content = Get-Content $utils -Raw

if ($content -match "def\s+load_html_text\s*\(") {
  Write-Host "== load_html_text already exists. No change."
} else {
  $alias = @'
  
# --- compat alias for html_cleaner ---
def load_html_text(path: str) -> str:
    # keep behavior identical to load_file_text
    return load_file_text(path)
'@

  Add-Content -Path $utils -Value $alias -Encoding UTF8
  Write-Host "== appended compat alias: load_html_text -> load_file_text"
}

# clear pyc caches
$pyc = Join-Path $ProjectRoot "src/__pycache__"
if (Test-Path $pyc) {
  Remove-Item $pyc -Recurse -Force -ErrorAction SilentlyContinue
  Write-Host "== cleared: $pyc"
}

Write-Host "== done =="
