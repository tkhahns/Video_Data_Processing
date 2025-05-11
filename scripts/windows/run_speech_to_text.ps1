# Video Data Processing Speech-to-Text
Write-Host "=== Video Data Processing Speech-to-Text ===" -ForegroundColor Cyan
Write-Host "This script transcribes speech audio files to text." -ForegroundColor Cyan

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
    poetry install --with speech --with common
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Poetry installation had issues. Retrying with common dependencies only..." -ForegroundColor Yellow
        poetry install --with common
    }
}

# Help message if --help flag is provided
if ($args.Count -gt 0 -and ($args[0] -eq "--help" -or $args[0] -eq "-h")) {
    Write-Host "`nUsage: .\run_speech_to_text.ps1 [options] <audio_file(s)>" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  --input-dir DIR      Directory containing input audio files"
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
    exit 0
}

# Parse input arguments
$inputDir = $null
$outputDir = $null
$otherArgs = @()

for ($i = 0; $i -lt $args.Count; $i++) {
    if ($args[$i] -eq "--input-dir" -and $i+1 -lt $args.Count) {
        $inputDir = $args[$i+1]
        $i++
    }
    elseif ($args[$i] -eq "--output-dir" -and $i+1 -lt $args.Count) {
        $outputDir = $args[$i+1]
        $i++
    }
    else {
        $otherArgs += $args[$i]
    }
}

# Run the speech-to-text script
Write-Host "`n[2/2] Running speech-to-text transcription..." -ForegroundColor Green

# Build command based on input parameters
$cmdArgs = @()

# Add input directory if specified
if ($inputDir) {
    Write-Host "Using input directory: $inputDir" -ForegroundColor Cyan
    $cmdArgs += "--input-dir", $inputDir
}

# Add output directory if specified
if ($outputDir) {
    $cmdArgs += "--output-dir", $outputDir
}

# Add other arguments
foreach ($arg in $otherArgs) {
    $cmdArgs += $arg
}

# Use Poetry to run the script
try {
    if ($cmdArgs.Count -eq 0 -and $otherArgs.Count -eq 0) {
        Write-Host "Entering interactive mode..." -ForegroundColor Cyan
        poetry run python -m src.speech_to_text --interactive
    } else {
        # Otherwise, pass all arguments to the script
        poetry run python -m src.speech_to_text $cmdArgs
    }

    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nSpeech-to-text transcription process completed successfully." -ForegroundColor Green
    } else {
        Write-Host "`nAn error occurred during the speech-to-text transcription process." -ForegroundColor Red
        exit $LASTEXITCODE
    }
} catch {
    Write-Host "`nAn exception occurred during speech-to-text: $_" -ForegroundColor Red
    exit 1
}
