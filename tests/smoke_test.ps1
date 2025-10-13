param(
    [string]$ConfigPath = "config.yaml"
)

Write-Host "== 生成样例数据 ==" -ForegroundColor Green
python .\tests\generate_samples.py

Write-Host "== ingest ==" -ForegroundColor Green
python -m src.cli ingest --config $ConfigPath

Write-Host "== embed ==" -ForegroundColor Green
python -m src.cli embed --config $ConfigPath

Write-Host "== imgindex ==" -ForegroundColor Green
python -m src.cli imgindex --config $ConfigPath

Write-Host "== charindex ==" -ForegroundColor Green
python -m src.cli charindex --config $ConfigPath

Write-Host "== links ==" -ForegroundColor Green
python -m src.cli links --config $ConfigPath

Write-Host "== graph ==" -ForegroundColor Green
python -m src.cli graph --config $ConfigPath

Write-Host "== query (dry) ==" -ForegroundColor Green
python -m src.query_cli --mode hybrid --wimg 0.25 --top 2 --smart-query --num-ctx 1024 "罗建祥 联系方式"
