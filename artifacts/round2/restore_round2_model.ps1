param(
    [string]$PartsDirectory = $PSScriptRoot,
    [string]$Destination = $PSScriptRoot
)
$ErrorActionPreference = 'Stop'
$expected = 'b2c5316650c7811b42c79190a55f54dd2f79223e25668c7c65bb2ad7f71c0763'
$parts = 1..7 | ForEach-Object { 'round2_model.zip.part{0:D2}' -f $_ }
foreach ($part in $parts) {
    if (-not (Test-Path -LiteralPath (Join-Path $PartsDirectory $part) -PathType Leaf)) {
        throw "Missing model part: $part. Download all seven parts into PartsDirectory."
    }
}
[System.IO.Directory]::CreateDirectory($Destination) | Out-Null
$zipPath = Join-Path $Destination 'round2_model.zip'
if (-not (Test-Path -LiteralPath $zipPath)) {
    $outputStream = [System.IO.File]::Open($zipPath, [System.IO.FileMode]::CreateNew)
    try {
        foreach ($part in $parts) {
            $inputStream = [System.IO.File]::OpenRead((Join-Path $PartsDirectory $part))
            try { $inputStream.CopyTo($outputStream) } finally { $inputStream.Dispose() }
        }
    } finally { $outputStream.Dispose() }
}
if ((Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) {
    throw 'Model ZIP checksum mismatch. No extraction performed.'
}
Expand-Archive -LiteralPath $zipPath -DestinationPath $Destination
Write-Output "Verified and extracted model: $(Join-Path $Destination 'round2_model')"
