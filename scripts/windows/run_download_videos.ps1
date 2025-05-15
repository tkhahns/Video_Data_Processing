# Video Data Processing SharePoint Downloader
Write-Host "=== Video Data Processing SharePoint Downloader ===" -ForegroundColor Cyan
Write-Host "This script simplifies downloading videos from SharePoint." -ForegroundColor Cyan

# Get the script's directory and project root
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = (Get-Item $SCRIPT_DIR).Parent.Parent.FullName

# Change to project root
Set-Location -Path $PROJECT_ROOT

# Check if Poetry is installed
if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
    Write-Host "`nPoetry is not installed. Installing poetry is required for dependency management." -ForegroundColor Red
    Write-Host "Please install Poetry with: (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -" -ForegroundColor Red
    Write-Host "All dependencies are defined in pyproject.toml" -ForegroundColor Red
    exit 1
} else {
    # Install dependencies using Poetry
    Write-Host "`n[1/2] Installing dependencies with Poetry..." -ForegroundColor Green
    poetry install --with download --with common
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Poetry installation had issues. Retrying with common dependencies only..." -ForegroundColor Yellow
        poetry install --with common
    }
}

# Display help if help flag is provided
if ($args.Count -gt 0 -and ($args[0] -eq "--help" -or $args[0] -eq "-h")) {
    Write-Host "`nUsage: .\run_download_videos.ps1 [options]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  --url URL             SharePoint folder URL containing videos (optional)"
    Write-Host "  --output-dir PATH     Directory to save downloaded files (default: .\data\videos)"
    Write-Host "  --list-only           Just list files without downloading"
    Write-Host "  --debug               Enable debug mode with detailed logging"
    Write-Host "  --batch               Process all files without manual selection" -ForegroundColor Yellow
    Write-Host ""
    exit 0
}

# Check if URL is provided in arguments
$urlProvided = $false
$url = $null
$batchMode = $false
$otherArgs = @()

for ($i = 0; $i -lt $args.Count; $i++) {
    if ($args[$i] -eq "--url" -and $i+1 -lt $args.Count) {
        $url = $args[$i+1]
        $urlProvided = $true
        $i++
    } elseif ($args[$i] -eq "--batch") {
        $batchMode = $true
    } else {
        $otherArgs += $args[$i]
    }
}

# If URL wasn't provided in command line, prompt for it
if (-not $urlProvided) {
    Write-Host "`nPlease enter the SharePoint URL containing the videos you want to download:" -ForegroundColor Yellow
    $url = Read-Host
    
    # Validate that a URL was entered
    while ([string]::IsNullOrEmpty($url)) {
        Write-Host "URL cannot be empty. Please enter a valid SharePoint URL:" -ForegroundColor Red
        $url = Read-Host
    }
}

# Run the download script
Write-Host "`n[2/2] Running SharePoint downloader..." -ForegroundColor Green

# Build command arguments
$cmdArgs = @("--url", $url)

# Add batch mode flag if specified
if ($batchMode) {
    Write-Host "Running in batch mode - downloading all files without manual selection" -ForegroundColor Cyan
    $cmdArgs += "--batch"
}

foreach ($arg in $otherArgs) {
    $cmdArgs += $arg
}

# Use Poetry to run the script
poetry run python -m src.download_videos $cmdArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nDownload process completed successfully." -ForegroundColor Green
} else {
    Write-Host "`nAn error occurred during the download process." -ForegroundColor Red
    exit $LASTEXITCODE
}
