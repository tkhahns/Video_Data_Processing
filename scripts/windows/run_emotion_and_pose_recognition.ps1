# Video Data Processing Emotion and Pose Recognition
Write-Host "=== Video Data Processing Emotion and Pose Recognition ===" -ForegroundColor Cyan
Write-Host "This script analyzes emotions and body poses in video files." -ForegroundColor Cyan

# Get the script's directory and project root
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = (Get-Item $SCRIPT_DIR).Parent.Parent.FullName

# Change to project root
Set-Location -Path $PROJECT_ROOT

# Check for Hugging Face token in environment
if ([string]::IsNullOrEmpty($env:HUGGINGFACE_TOKEN)) {
    Write-Host "`n=== Hugging Face Authentication ===" -ForegroundColor Cyan
    Write-Host "This tool requires a Hugging Face token for accessing models." -ForegroundColor Yellow
    Write-Host "You can get your token from: https://huggingface.co/settings/tokens" -ForegroundColor Yellow
    Write-Host "Note: Your token will only be used for this session and will not be saved." -ForegroundColor Yellow
    
    # Prompt for token
    $HF_TOKEN = Read-Host -Prompt "Enter your Hugging Face token"
    
    if ([string]::IsNullOrEmpty($HF_TOKEN)) {
        Write-Host "No token provided. Some features may not work correctly." -ForegroundColor Red
    } else {
        Write-Host "Token received for this session" -ForegroundColor Green
    }
    
    # Set environment variable
    $env:HUGGINGFACE_TOKEN = $HF_TOKEN
}

# Check if Poetry is installed
if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
    Write-Host "`nPoetry is not installed. Installing poetry is required for dependency management." -ForegroundColor Red
    Write-Host "Please install Poetry with: (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -" -ForegroundColor Red
    Write-Host "All dependencies are defined in pyproject.toml" -ForegroundColor Red
    exit 1
} else {
    # Install dependencies using Poetry
    Write-Host "`n[1/2] Installing dependencies with Poetry..." -ForegroundColor Green
    poetry install --with emotion --with common
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Poetry installation had issues. Retrying with common dependencies only..." -ForegroundColor Yellow
        poetry install --with common
    }
}

# Help message if --help flag is provided
if ($args.Count -gt 0 -and ($args[0] -eq "--help" -or $args[0] -eq "-h")) {
    Write-Host "`nUsage: .\run_emotion_recognition.ps1 [options] <video_file(s)>" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  --input-dir DIR      Directory containing input video files"
    Write-Host "  --output-dir DIR     Directory to save emotion analysis results (default: ./output/emotions)"
    Write-Host "  --batch              Process all videos in input directory"
    Write-Host "  --interactive        Force interactive video selection mode"
    Write-Host "  --debug              Enable debug logging"
    Write-Host "  --help               Show this help message"
    Write-Host ""
    Write-Host "If run without arguments, the script will show an interactive video selection menu."
    exit 0
}

# Parse input arguments
$inputDir = $null
$outputDir = $null
$batchMode = $false
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
    elseif ($args[$i] -eq "--batch") {
        $batchMode = $true
    }
    else {
        $otherArgs += $args[$i]
    }
}

# Run the emotion recognition script
Write-Host "`n[2/2] Running emotion and pose recognition analysis..." -ForegroundColor Green

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

# Always add pose estimation by default (unless --no-pose is explicitly included)
$NO_POSE_PRESENT = $false
foreach ($arg in $otherArgs) {
    if ($arg -eq "--no-pose") {
        $NO_POSE_PRESENT = $true
        break
    }
}

if (-not $NO_POSE_PRESENT) {
    $cmdArgs += "--with-pose"
}

# Always use multi-speaker by default (unless --single-speaker is explicitly included)
$SINGLE_SPEAKER_PRESENT = $false
foreach ($arg in $otherArgs) {
    if ($arg -eq "--single-speaker") {
        $SINGLE_SPEAKER_PRESENT = $true
        break
    }
}

if (-not $SINGLE_SPEAKER_PRESENT) {
    $cmdArgs += "--multi-speaker"
}

# Add batch mode flag if specified
if ($batchMode) {
    Write-Host "Running in batch mode - processing all files without manual selection" -ForegroundColor Cyan
    $cmdArgs += "--batch"
}

# Add other arguments
foreach ($arg in $otherArgs) {
    $cmdArgs += $arg
}

# Use Poetry to run the script
try {
    if ($cmdArgs.Count -eq 0 -and $otherArgs.Count -eq 0) {
        Write-Host "Entering interactive mode with pose estimation and multi-speaker tracking..." -ForegroundColor Cyan
        poetry run python -m src.emotion_and_pose_recognition.cli --with-pose --multi-speaker --interactive
    } else {
        # Otherwise, pass all arguments to the script
        poetry run python -m src.emotion_and_pose_recognition.cli $cmdArgs
    }

    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nEmotion and pose recognition analysis completed successfully." -ForegroundColor Green
    } else {
        Write-Host "`nAn error occurred during emotion and pose recognition analysis." -ForegroundColor Red
        exit $LASTEXITCODE
    }
} catch {
    Write-Host "`nAn exception occurred during emotion and pose recognition: $_" -ForegroundColor Red
    exit 1
}

# Clear the token from the environment if we set it locally
if ([string]::IsNullOrEmpty($HF_TOKEN) -eq $false) {
    $env:HUGGINGFACE_TOKEN = ""
}