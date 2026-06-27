param(
    [int]$BackendWaitSeconds = 60,
    [switch]$NoBackendWait
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Frontend = Join-Path $Root "frontend"
$Node = (Get-Command node -ErrorAction SilentlyContinue).Source
if (-not $Node) {
    $Node = "C:\Users\hoang\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
}
$Vite = Join-Path $Frontend "node_modules\vite\bin\vite.js"

function Stop-ExistingFrontendOnPort {
    param([int]$Port)

    $listeners = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
    foreach ($listener in $listeners) {
        $pidOnPort = [int]$listener.OwningProcess
        if ($pidOnPort -eq $PID) {
            continue
        }

        $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $pidOnPort" -ErrorAction SilentlyContinue
        $cmd = if ($proc) { [string]$proc.CommandLine } else { "" }

        if ($cmd -like "*vite*") {
            Write-Host "[Frontend] Stop frontend cu tren port $Port, PID $pidOnPort"
            Stop-Process -Id $pidOnPort -Force
            Start-Sleep -Seconds 1
            continue
        }

        throw "Port $Port dang bi process PID $pidOnPort chiem. CommandLine: $cmd"
    }
}

function Wait-BackendReady {
    param(
        [int]$TimeoutSeconds,
        [string]$Url = "http://127.0.0.1:8000/api/v1/violations"
    )

    if ($NoBackendWait) {
        return
    }

    Write-Host "[Frontend] Dang cho Backend san sang o http://127.0.0.1:8000 ..."
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2 | Out-Null
            Write-Host "[Frontend] Backend da san sang."
            return
        } catch {
            Start-Sleep -Seconds 2
        }
    }

    throw "Backend chua chay o port 8000. Hay mo terminal khac va chay: cd E:\TGMTTTT; .\start_backend.ps1"
}

if (-not (Test-Path -LiteralPath $Node)) {
    throw "Khong tim thay Node bundled: $Node"
}

if (-not (Test-Path -LiteralPath $Vite)) {
    throw "Khong tim thay Vite. Kiem tra lai thu muc frontend\node_modules."
}

Write-Host "[Frontend] URL: http://127.0.0.1:5173"

Stop-ExistingFrontendOnPort -Port 5173
Wait-BackendReady -TimeoutSeconds $BackendWaitSeconds

Set-Location $Frontend
& $Node $Vite --host 127.0.0.1 --port 5173 --strictPort
