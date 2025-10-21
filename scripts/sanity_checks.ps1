$ErrorActionPreference = "Continue"
Write-Host "=== Sanity Checks ==="

# Python & CUDA
python -V
try { python -c "import torch, sys; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())" } catch { Write-Host $_ }

# Ollama
try {
  Write-Host "`nOllama version:"
  Invoke-WebRequest -UseBasicParsing http://127.0.0.1:11434/api/version | Select-Object -Expand Content
  Write-Host "`nOllama ps:"
  Invoke-WebRequest -UseBasicParsing http://127.0.0.1:11434/api/ps | Select-Object -Expand Content
} catch {
  Write-Warning "Ollama not reachable at 127.0.0.1:11434"
}
try { ollama list } catch { Write-Host "ollama CLI not found (optional)" }

# Tesseract (robust regex from config.yaml)
$configPath = Join-Path $PSScriptRoot "..\config.yaml"
$tesseractPath = $null
if (Test-Path $configPath) {
  $raw = Get-Content $configPath -Raw
  $m = [regex]::Match($raw, 'ocr_tesseract_cmd:\s*"?([^"\r\n]+)"?')
  if ($m.Success) { $tesseractPath = $m.Groups[1].Value.Trim() }
}
if ($tesseractPath -and (Test-Path $tesseractPath)) {
  & "$tesseractPath" --version
} else {
  Write-Host "Tesseract not configured or not found (OCR is optional)."
}

# Long path policy
try {
  $reg = Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -ErrorAction SilentlyContinue
  if ($reg.LongPathsEnabled -ne 1) {
    Write-Warning "Windows long paths not enabled. Consider enabling to avoid path issues."
  } else {
    Write-Host "Long paths: Enabled"
  }
} catch { }

Write-Host "=== Checks done ==="
