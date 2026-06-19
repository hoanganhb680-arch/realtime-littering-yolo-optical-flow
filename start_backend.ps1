$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonRoot = "C:\Users\hoang\.cache\codex-runtimes\codex-primary-runtime\dependencies\python"
$Python = Join-Path $PythonRoot "python.exe"
$SitePackages = Join-Path $Root ".venv312\Lib\site-packages"
$Src = Join-Path $Root "src"

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "$SitePackages;$Src"
$env:PATH = "$PythonRoot;$SitePackages;$(Join-Path $SitePackages 'torch\lib');$(Join-Path $SitePackages 'cv2');$env:PATH"

Set-Location $Src
& $Python -m uvicorn server:app --host 127.0.0.1 --port 8000 --log-level info
