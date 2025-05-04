# Video Data Processing - Speech Separation
Write-Host "=== Video Data Processing Speech Separation ===" -ForegroundColor Cyan
Write-Host "This script extracts and isolates speech from video files." -ForegroundColor Cyan

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

# Install required packages if needed
try {
    python -c "import speechbrain, tqdm, pydub, colorama" 2>$null
    $packagesInstalled = $?
} catch {
    $packagesInstalled = $false
}

if (-not $packagesInstalled) {
    Write-Host "`nInstalling required packages..." -ForegroundColor Yellow
    pip install speechbrain moviepy torchaudio tqdm pydub ffmpeg-python colorama
}

# Help message if --help flag is provided
if ($args.Count -gt 0 -and ($args[0] -eq "--help" -or $args[0] -eq "-h")) {
    Write-Host "`nUsage: .\run_separate_speech.ps1 [options] <video_file(s)>" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  --output-dir DIR     Directory to save separated speech files (default: ./output/separated_speech)"
    Write-Host "  --model MODEL        Speech separation model to use (sepformer, conv-tasnet)"
    Write-Host "  --file-type TYPE     Output file format: wav (1), mp3 (2), or both (3) (default: mp3)"
    Write-Host "  --recursive          Process video files in subdirectories recursively"
    Write-Host "  --debug              Enable debug logging"
    Write-Host "  --interactive        Force interactive video selection mode"
    Write-Host "  --detect-dialogues   Enable dialogue detection (identifies different speakers)"
    Write-Host "  --help               Show this help message"
    Write-Host ""
    Write-Host "If run without arguments, the script will show an interactive video selection menu."
    exit
}

# Run the speech separation script
Write-Host "`nRunning speech separation..." -ForegroundColor Green

try {
    # If no arguments are provided, use interactive mode
    if ($args.Count -eq 0) {
        Write-Host "Entering interactive mode..." -ForegroundColor Yellow
        python -m src.separate_speech --interactive --detect-dialogues
    } else {
        # Otherwise, pass all arguments to the script with detect-dialogues flag
        python -m src.separate_speech --detect-dialogues $args
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nSpeech separation process completed successfully." -ForegroundColor Green
    } else {
        Write-Host "`nSpeech separation process completed with errors." -ForegroundColor Yellow
        exit $LASTEXITCODE
    }
} catch {
    Write-Host "`nAn error occurred during the speech separation process." -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
