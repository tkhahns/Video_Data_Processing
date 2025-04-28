# Video Data Processing Model Downloader
Write-Host "=== Video Data Processing Model Downloader ===" -ForegroundColor Cyan
Write-Host "This script will set up the environment and download required models." -ForegroundColor Cyan

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
pip install huggingface-hub pandas transformers

# Ensure models directory exists
Write-Host "`n[4/5] Creating models directory if needed..." -ForegroundColor Green
if (-not (Test-Path ".\models\downloaded")) {
    New-Item -Path ".\models\downloaded" -ItemType Directory -Force | Out-Null
}

# Run the download script
Write-Host "`n[5/5] Downloading models..." -ForegroundColor Green
python download_models.py

Write-Host "`n✅ Download process completed successfully!" -ForegroundColor Green
Write-Host "You can now use the models for video data processing." -ForegroundColor Green