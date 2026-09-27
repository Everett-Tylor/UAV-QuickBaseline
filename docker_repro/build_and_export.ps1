param([string]$OutputTar = (Join-Path $PSScriptRoot 'uav-round2-image.tar'))
$ErrorActionPreference = 'Stop'
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw 'Install and start Docker with Linux GPU containers first.' }
docker build -t uav-round2:74.22 $PSScriptRoot
if ($LASTEXITCODE -ne 0) { throw 'Docker build failed' }
docker run --rm --gpus all uav-round2:74.22 check
if ($LASTEXITCODE -ne 0) { throw 'GPU container check failed; image export stopped' }
docker save --output $OutputTar uav-round2:74.22
if ($LASTEXITCODE -ne 0) { throw 'Docker export failed' }
Write-Output "Image exported: $OutputTar"
