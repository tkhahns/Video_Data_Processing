# Video Data Processing - Speech-to-Text
Write-Host "=== Video Data Processing Speech-to-Text ===" -ForegroundColor Cyan
Write-Host "This script transcribes speech audio files to text." -ForegroundColor Cyan

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
    python -c "import torch, transformers, colorama" 2>$null
    $packagesInstalled = $?
} catch {
    $packagesInstalled = $false
}

if (-not $packagesInstalled) {
    Write-Host "`nInstalling required packages..." -ForegroundColor Yellow
    pip install torch transformers tqdm colorama
    pip install git+https://github.com/m-bain/whisperX.git
}

# Help message if --help flag is provided
if ($args.Count -gt 0 -and ($args[0] -eq "--help" -or $args[0] -eq "-h")) {
    Write-Host "`nUsage: .\run_speech_to_text.ps1 [options] <audio_file(s)>" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  --output-dir DIR     Directory to save transcription files (default: ./output/transcripts)"
    Write-Host "  --model MODEL        Speech-to-text model to use (whisperx, xlsr)"
    Write-Host "  --language LANG      Language code for transcription (default: en)"
    Write-Host "  --output-format FMT  Output format: srt, txt, or both (default: srt)"
    Write-Host "  --recursive          Process audio files in subdirectories recursively"
    Write-Host "  --select             Force file selection prompt even when files are provided"
    Write-Host "  --debug              Enable debug logging"
    Write-Host "  --interactive        Force interactive audio selection mode"
    Write-Host "  --help               Show this help message"
    Write-Host ""
    Write-Host "If run without arguments, the script will show an interactive audio selection menu."
    exit
}

# Run the speech-to-text script
Write-Host "`nRunning speech-to-text transcription..." -ForegroundColor Green

try {
    # If no arguments are provided, use interactive mode
    if ($args.Count -eq 0) {
        Write-Host "Entering interactive mode..." -ForegroundColor Yellow
        python -m src.speech_to_text --interactive
    } else {
        # Otherwise, pass all arguments to the script
        python -m src.speech_to_text $args
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nSpeech-to-text transcription process completed successfully." -ForegroundColor Green
    } else {
        Write-Host "`nSpeech-to-text transcription process completed with errors." -ForegroundColor Yellow
        exit $LASTEXITCODE
    }
} catch {
    Write-Host "`nAn error occurred during the speech-to-text transcription process." -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
