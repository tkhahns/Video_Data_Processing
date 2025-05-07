# Video Data Processing - Emotion Recognition
Write-Host "=== Video Data Processing Emotion Recognition ===" -ForegroundColor Cyan
Write-Host "This script detects and labels emotions in video files." -ForegroundColor Cyan

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
    python -c "import feat, cv2, numpy, tqdm" 2>$null
    $packagesInstalled = $?
} catch {
    $packagesInstalled = $false
}

if (-not $packagesInstalled) {
    Write-Host "`nInstalling required packages..." -ForegroundColor Yellow
    pip install py-feat==0.5.1 opencv-python numpy tqdm
}

# Help message if --help flag is provided
if ($args.Count -gt 0 -and ($args[0] -eq "--help" -or $args[0] -eq "-h")) {
    Write-Host "`nUsage: .\run_emotion_recognition.ps1 [options] <video_file(s)>" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  --output-dir DIR     Directory to save emotion recognition results (default: ./output/emotions)"
    Write-Host "  --input DIR/FILE     Input directory or video file"
    Write-Host "  --process-all        Process all video files in the input directory"
    Write-Host "  --interval NUM       Frame interval in seconds (default: 1.0)"
    Write-Host "  --threshold NUM      Confidence threshold for emotions (default: 0.5)"
    Write-Host "  --device TYPE        Device to run on: cpu or cuda (default: cpu)"
    Write-Host "  --recursive          Process video files in subdirectories recursively"
    Write-Host "  --debug              Enable debug logging"
    Write-Host "  --interactive        Force interactive video selection mode"
    Write-Host "  --help               Show this help message"
    Write-Host ""
    Write-Host "If run without arguments, the script will show an interactive video selection menu."
    exit
}

# Run the emotion recognition script
Write-Host "`nRunning emotion recognition..." -ForegroundColor Green

try {
    # If no arguments are provided, use interactive mode
    if ($args.Count -eq 0) {
        Write-Host "Entering interactive mode..." -ForegroundColor Yellow
        python -m src.emotion_recognition_main --interactive
    } else {
        # Otherwise, pass all arguments to the script
        python -m src.emotion_recognition_main $args
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nEmotion recognition process completed successfully." -ForegroundColor Green
    } else {
        Write-Host "`nEmotion recognition process completed with errors." -ForegroundColor Yellow
        exit $LASTEXITCODE
    }
} catch {
    Write-Host "`nAn error occurred during the emotion recognition process." -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
