# Video Data Processing Model Downloader
Write-Host "=== Video Data Processing Model Downloader ===" -ForegroundColor Cyan
Write-Host "This script will set up the environment and download required models." -ForegroundColor Cyan

# Parse command line arguments
$modelType = $null
$forceDownload = $false
$dryRun = $false

for ($i = 0; $i -lt $args.Count; $i++) {
    switch ($args[$i]) {
        "--model-type" {
            $i++
            $modelType = $args[$i]
        }
        "--force" {
            $forceDownload = $true
        }
        "--dry-run" {
            $dryRun = $true
        }
        "--help" {
            Write-Host "`nUsage: .\run_download_models.ps1 [options]" -ForegroundColor Yellow
            Write-Host ""
            Write-Host "Options:" -ForegroundColor Yellow
            Write-Host "  --model-type TYPE    Download only models of specified type (audio, video, image)"
            Write-Host "  --force              Force re-download even if model directory exists"
            Write-Host "  --dry-run            Show what would be downloaded without actually downloading"
            Write-Host "  --help               Show this help message"
            exit 0
        }
    }
}

# Check if virtual environment exists
if (-not (Test-Path ".venv")) {
    Write-Host "`n[1/5] Creating virtual environment..." -ForegroundColor Green
    python -m venv .venv
} else {
    Write-Host "`n[1/5] Virtual environment already exists." -ForegroundColor Green
}

# Activate virtual environment
Write-Host "`n[2/5] Activating virtual environment..." -ForegroundColor Green
.\.venv\Scripts\Activate.ps1

# Update pip and install dependencies
Write-Host "`n[3/5] Updating pip and installing dependencies..." -ForegroundColor Green
pip install --upgrade pip
pip install -r requirements.txt
pip install huggingface-hub pandas transformers gitpython

# Ensure models directory exists
Write-Host "`n[4/5] Creating models directory if needed..." -ForegroundColor Green
if (-not (Test-Path ".\models\downloaded")) {
    New-Item -Path ".\models\downloaded" -ItemType Directory -Force | Out-Null
}

# Prepare download arguments
$downloadArgs = @(".\src\download_models.py")

if ($modelType) {
    $downloadArgs += "--model-types"
    $downloadArgs += $modelType
}

if ($forceDownload) {
    $downloadArgs += "--force"
}

if ($dryRun) {
    $downloadArgs += "--dry-run"
}

# Run the download script
Write-Host "`n[5/5] Downloading models..." -ForegroundColor Green
python $downloadArgs

Write-Host "`n✅ Download process completed!" -ForegroundColor Green
Write-Host "You can now use the models for video data processing." -ForegroundColor Green