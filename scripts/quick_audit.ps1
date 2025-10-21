\
    Param(
      [Parameter(Mandatory=$true)][string]$Root,
      [Parameter(Mandatory=$true)][string]$OutDir,
      [Parameter(Mandatory=$true)][string]$Dbg,
      [string]$Keyword = ""
    )

    Write-Host "== audit start =="
    New-Item -ItemType Directory -Force -Path $Dbg | Out-Null

    $chunks = Join-Path $OutDir "chunks.jsonl"
    $covout = Join-Path $Dbg "audit_coverage.json"
    python -m tools.coverage --root "$Root" --chunks "$chunks" --out "$covout"

    if ($Keyword -ne "") {
      $hits = Select-String -Path $chunks -Pattern $Keyword -SimpleMatch | Select-Object -ExpandProperty Path -Unique
      Write-Host "[grep] keyword='$Keyword' unique_files=$($hits.Count)"
      $hits | ForEach-Object { Write-Host "  $_" }
    }

    Write-Host "== done. see $Dbg =="
