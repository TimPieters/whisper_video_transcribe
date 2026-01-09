param(
  [Parameter(Mandatory=$true)][string]$ModuleManifestPath,
  [Parameter(Mandatory=$true)][string]$ModelPath,
  [Parameter(Mandatory=$true)][string]$InputWavPath,
  [Parameter(Mandatory=$true)][string]$OutDir,
  [string]$Language,
  [Parameter(Mandatory=$true)][ValidateSet("txt","srt","vtt","json")][string]$OutFormat,
  [string]$AdapterName
)

$ErrorActionPreference = "Stop"

try {
  if (-not (Test-Path $ModuleManifestPath)) {
    throw "Module manifest not found at $ModuleManifestPath"
  }

  Import-Module $ModuleManifestPath -Force

  $adapterArgs = @{}
  if ($AdapterName) {
    $adapterArgs["adapter"] = $AdapterName
  }

  $model = Import-WhisperModel -path $ModelPath @adapterArgs

  $transcribeArgs = @{ model = $model; path = $InputWavPath }
  if ($Language -and $Language -ne "auto") {
    $transcribeArgs["language"] = $Language
  }

  $res = Transcribe-File @transcribeArgs

  $outDirResolved = (Resolve-Path $OutDir).Path
  if (-not (Test-Path $outDirResolved)) {
    New-Item -ItemType Directory -Path $outDirResolved -Force | Out-Null
  }

  $formats = @("txt", "srt", "vtt", $OutFormat, "json") | Select-Object -Unique

  foreach ($fmt in $formats) {
    switch ($fmt) {
      "txt" {
        $res | Export-Text -path (Join-Path $outDirResolved "transcript.txt") -timestamps
      }
      "srt" {
        $res | Export-SubRip -path (Join-Path $outDirResolved "transcript.srt")
      }
      "vtt" {
        $res | Export-WebVTT -path (Join-Path $outDirResolved "transcript.vtt")
      }
      "json" {
        $segments = Format-Segments $res
        $textContent = ($segments | ForEach-Object { $_.text }) -join ""
        $payload = @{
          language = if ($Language -and $Language -ne "auto") { $Language } else { $null }
          segments = $segments
          text = $textContent
        }
        $jsonPath = Join-Path $outDirResolved "transcript.json"
        $payload | ConvertTo-Json -Depth 6 | Out-File -FilePath $jsonPath -Encoding utf8
      }
    }
  }
}
catch {
  Write-Error $_
  exit 1
}
