Param()

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$ingest = Join-Path $here "..\src\ingest.py"

if (-not (Test-Path $ingest)) {
  Write-Error "Not found: $ingest (run this from repo root; scripts folder should be alongside src)"
  exit 1
}

# Backup
Copy-Item $ingest "$ingest.bak" -Force

# Patch text (append compatibility wrapper)
$patch = @'
# --- HOTFIX (appended by scripts\hotfix_ingest_kw.ps1) ---
# Accept legacy CLI kwargs without failing (e.g., incremental)
try:
    _ingest_orig = ingest  # type: ignore[name-defined]
    def ingest(*args, **kwargs):  # type: ignore[no-redef]
        # Tolerate legacy flags sent by older/newer CLI
        for _k in ("incremental",):
            if _k in kwargs:
                kwargs.pop(_k, None)
        return _ingest_orig(*args, **kwargs)
except Exception as _e:
    # Do not break import even if symbol names differ
    pass
# --- END HOTFIX ---
'@

Add-Content -Path $ingest -Value $patch -Encoding UTF8

# Clear caches so Python doesn't read stale bytecode
Get-ChildItem (Join-Path $here "..\src") -Recurse -Include "__pycache__" -Directory | ForEach-Object {
  try { Remove-Item $_.FullName -Recurse -Force -ErrorAction SilentlyContinue } catch {}
}

Write-Host "== appended ingest() legacy-kwargs hotfix =="
Write-Host "== backup: $ingest.bak =="