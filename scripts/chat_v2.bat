\
    @echo off
    setlocal
    REM Simple launcher avoiding PowerShell execution policy issues
    if exist ".venv\Scripts\python.exe" (
      ".venv\Scripts\python.exe" -m src.chat2.cli --config "chat2_config.yaml" --session "my-session"
    ) else (
      python -m src.chat2.cli --config "chat2_config.yaml" --session "my-session"
    )
    endlocal
