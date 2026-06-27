param(
    [ValidateSet("file", "ip_camera")]
    [string]$SourceMode = "file",

    [string]$VideoFile = "",

    [switch]$NoStopExisting
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvCandidates = @(
    (Join-Path $Root ".venv312\Scripts\python.exe"),
    (Join-Path $Root ".venv\Scripts\python.exe")
)
$Python = $VenvCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $Python) {
    $Python = (Get-Command python -ErrorAction SilentlyContinue).Source
}
if (-not $Python) {
    throw "Khong tim thay Python. Hay tao virtualenv va cai: pip install -r requirements.txt"
}
$SitePackagesRoot = Split-Path -Parent (Split-Path -Parent $Python)
$SitePackages = Join-Path $SitePackagesRoot "Lib\site-packages"
$Src = Join-Path $Root "src"

if ([string]::IsNullOrWhiteSpace($VideoFile)) {
    $VideoFile = Join-Path $Root "video\di_bo_17.mp4"
}

function Stop-ExistingBackendOnPort {
    param([int]$Port)

    $listeners = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
    foreach ($listener in $listeners) {
        $pidOnPort = [int]$listener.OwningProcess
        if ($pidOnPort -eq $PID) {
            continue
        }

        $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $pidOnPort" -ErrorAction SilentlyContinue
        $cmd = if ($proc) { [string]$proc.CommandLine } else { "" }

        if ($cmd -like "*uvicorn server:app*") {
            if ($NoStopExisting) {
                throw "Port $Port dang duoc backend cu dung boi PID $pidOnPort. Hay tat no truoc hoac bo -NoStopExisting."
            }
            Write-Host "[Backend] Stop backend cu tren port $Port, PID $pidOnPort"
            Stop-Process -Id $pidOnPort -Force
            Start-Sleep -Seconds 1
            continue
        }

        throw "Port $Port dang bi process PID $pidOnPort chiem. CommandLine: $cmd"
    }
}

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = "$SitePackages;$Src;$env:PYTHONPATH"
$env:PATH = "$(Split-Path -Parent $Python);$SitePackages;$(Join-Path $SitePackages 'torch\lib');$(Join-Path $SitePackages 'cv2');$env:PATH"
$env:DETECTOR_SOURCE_MODE = $SourceMode
$env:DETECTOR_VIDEO_FILE = $VideoFile

Write-Host "[Backend] Source mode : $SourceMode"
if ($SourceMode -eq "file") {
    Write-Host "[Backend] Video file  : $VideoFile"
}
Write-Host "[Backend] API         : http://127.0.0.1:8000"

Stop-ExistingBackendOnPort -Port 8000

Set-Location $Src
& $Python -m uvicorn server:app --host 127.0.0.1 --port 8000 --log-level info
