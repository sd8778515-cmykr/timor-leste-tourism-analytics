param(
    [switch]$KeepServer
)

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$expectedRoot = (Resolve-Path "D:\turismo_project (1)").Path
if ($projectRoot -ne $expectedRoot) {
    throw "Reset dibatalkan: lokasi proyek tidak sesuai. Ditemukan: $projectRoot"
}

if (-not $KeepServer) {
    $listeners = netstat -ano | Select-String ":8501\s+.*LISTENING\s+(\d+)$"
    foreach ($listener in $listeners) {
        $processId = [int]$listener.Matches[0].Groups[1].Value
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    }
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupRoot = Join-Path $projectRoot "demo_backup\$stamp"
$targets = @(
    "data\processed\reviews_processed.csv",
    "data\features\review_features.csv",
    "data\output\reviews_mined.csv",
    "reports\data_collection_log.json",
    "reports\eda_summary.json",
    "reports\model_evaluation.json",
    "reports\EVALUASI_DAN_INTERPRETASI.md",
    "reports\tables\category_sentiment.csv",
    "reports\tables\correlation_matrix.csv",
    "reports\tables\descriptive_statistics.csv",
    "reports\tables\model_metrics.csv",
    "reports\tables\sample_validasaun_manual.csv"
)

$targets += Get-ChildItem (Join-Path $projectRoot "reports\figures") -File -ErrorAction SilentlyContinue |
    ForEach-Object { $_.FullName.Substring($projectRoot.Length + 1) }
$targets += Get-ChildItem (Join-Path $projectRoot "powerbi") -File -Filter "*.csv" -ErrorAction SilentlyContinue |
    ForEach-Object { $_.FullName.Substring($projectRoot.Length + 1) }

$moved = 0
foreach ($relativePath in $targets | Select-Object -Unique) {
    $source = Join-Path $projectRoot $relativePath
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        continue
    }

    $resolvedSource = (Resolve-Path -LiteralPath $source).Path
    if (-not $resolvedSource.StartsWith($projectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Reset dibatalkan: target berada di luar proyek: $resolvedSource"
    }

    $destination = Join-Path $backupRoot $relativePath
    $destinationDirectory = Split-Path -Parent $destination
    New-Item -ItemType Directory -Path $destinationDirectory -Force | Out-Null
    Move-Item -LiteralPath $resolvedSource -Destination $destination
    $moved++
}

Write-Host "Reset demo selesai. $moved file hasil dipindahkan."
Write-Host "Backup: $backupRoot"
Write-Host "Dataset mentah, label manual, source code, dan dokumentasi tetap tersedia."

