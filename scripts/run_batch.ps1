# 录音转写 CLI（无需启动 Web 服务）
# 用法: .\scripts\run_batch.ps1 -Input "录音.wav" -Output "out"
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Input,
    [string]$Output = "out"
)
Set-Location $PSScriptRoot\..
python scripts/run_batch.py $Input -o $Output
