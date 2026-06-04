# Start FunASR Runtime via Docker, then local gateway (Windows)
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Deploy = Join-Path $Root "deploy"

if (-not (Test-Path (Join-Path $Deploy "runtime.env"))) {
    Copy-Item (Join-Path $Deploy "runtime.env.example") (Join-Path $Deploy "runtime.env")
    Write-Host "Created deploy/runtime.env — edit model paths before production."
}

Push-Location $Deploy
docker compose --env-file runtime.env -f docker-compose.yml up -d funasr-runtime
Pop-Location

Start-Sleep -Seconds 10
python (Join-Path $Root "scripts\run_server.py") --config (Join-Path $Root "config.enterprise.yaml")
