# Video Data Processing Speech Separation
Write-Host "=== Video Data Processing Speech Separation ===" -ForegroundColor Cyan
Write-Host "This script extracts and isolates speech from video files." -ForegroundColor Cyan

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
    Write-Host "`nUsage: .\run_separate_speech.ps1 [options] <video_file(s)>" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  --input-dir DIR      Directory containing input video files (default: ./data/videos)"
    Write-Host "  --output-dir DIR     Directory to save separated speech files (default: ./output/separated_speech)"
    Write-Host "  --model MODEL        Speech separation model to use (sepformer, conv-tasnet)" 
    Write-Host "  --file-type TYPE     Output file format: wav (1), mp3 (2), or both (3) (default: mp3)"
    Write-Host "  --recursive          Process video files in subdirectories recursively"
    Write-Host "  --debug              Enable debug logging"
    Write-Host "  --interactive        Force interactive video selection mode"
    Write-Host "  --help               Show this help message"
    Write-Host ""
    Write-Host "If run without arguments, the script will show an interactive video selection menu."
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

# Run the speech separation script
Write-Host "`n[2/2] Running speech separation..." -ForegroundColor Green

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
        poetry run python -m src.separate_speech --interactive
    } else {
        # Otherwise, pass all arguments to the script
        poetry run python -m src.separate_speech $cmdArgs
    }

    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nSpeech separation process completed successfully." -ForegroundColor Green
    } else {
        Write-Host "`nAn error occurred during the speech separation process." -ForegroundColor Red
        exit $LASTEXITCODE
    }
} catch {
    Write-Host "`nAn exception occurred during speech separation: $_" -ForegroundColor Red
    exit 1
}
