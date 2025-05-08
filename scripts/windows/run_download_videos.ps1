# Video Data Processing SharePoint Downloader
Write-Host "=== Video Data Processing SharePoint Downloader ===" -ForegroundColor Cyan
Write-Host "This script simplifies downloading videos from SharePoint." -ForegroundColor Cyan

# Check if virtual environment exists
if (-not (Test-Path ".venv")) {
    Write-Host "`n[1/2] Creating virtual environment..." -ForegroundColor Green
    python -m venv .venv
} else {
    Write-Host "`n[1/2] Using existing virtual environment." -ForegroundColor Green
}

# Activate virtual environment
Write-Host "`n[2/2] Activating virtual environment..." -ForegroundColor Green
.\.venv\Scripts\Activate.ps1

# Help message if --help flag is provided
if ($args.Count -gt 0 -and ($args[0] -eq "--help" -or $args[0] -eq "-h")) {
    Write-Host "`nUsage: .\run_download_videos.ps1 [options]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  --url URL             SharePoint folder URL containing videos (optional)"
    Write-Host "  --output-dir PATH     Directory to save downloaded files (default: ./data/videos)"
    Write-Host "  --list-only           Just list files without downloading"
    Write-Host "  --debug               Enable debug mode with detailed logging"
    Write-Host ""
    exit 0
}

# Collect all arguments that are not for the URL
$urlProvided = $false
$scriptArgs = @()
$i = 0
while ($i -lt $args.Count) {
    if ($args[$i] -eq "--url" -and $i+1 -lt $args.Count) {
        $url = $args[$i+1]
        $urlProvided = $true
        $i += 2
    } else {
        $scriptArgs += $args[$i]
        $i++
    }
}

# If URL wasn't provided in command line, prompt for it
if (-not $urlProvided) {
    Write-Host "`nPlease enter the SharePoint URL containing the videos you want to download:" -ForegroundColor Yellow
    $url = Read-Host
    
    # Validate that a URL was entered
    while ([string]::IsNullOrWhiteSpace($url)) {
        Write-Host "URL cannot be empty. Please enter a valid SharePoint URL:" -ForegroundColor Red
        $url = Read-Host
    }
}

# Run the download script with all arguments
Write-Host "`nRunning SharePoint downloader..." -ForegroundColor Green
try {
    python .\src\download_videos\main.py --url "$url" $scriptArgs
    Write-Host "`nDownload process completed." -ForegroundColor Green
} catch {
    Write-Host "`nAn error occurred during the download process." -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
