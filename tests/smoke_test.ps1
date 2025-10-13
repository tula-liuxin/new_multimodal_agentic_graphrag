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

Write-Host "== query (AGI) ==" -ForegroundColor Green
python -m src.query_cli --mode hybrid --alpha 0.4 --beta 0.6 --gamma 1.0 --wimg 0.10 --wbool 0.8 --top 5 --smart-query --num-ctx 2048 "ASI/AGI 路线与控制"
