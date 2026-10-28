# Start the FusionC live dashboard: Flask API (port 5000) + Vite dev server (port 5173).
#   .\scripts\start_demo.ps1
# Open http://localhost:5173 in a browser. Close the two windows to stop everything.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

if (-not (Test-Path "$root\build\fusionc.jar")) {
    throw 'build/fusionc.jar is missing -- run .\scripts\build.ps1 first'
}

Write-Host '[1/2] Flask API      -> http://127.0.0.1:5000  (window 1)'
Start-Process powershell -ArgumentList '-NoExit', '-Command', "Set-Location '$root'; python app/server.py --port 5000"

Write-Host '[2/2] React frontend -> http://localhost:5173  (window 2)'
Start-Process powershell -ArgumentList '-NoExit', '-Command', "Set-Location '$root\app'; npm run dev"

Write-Host ''
Write-Host 'dashboard: http://localhost:5173'
