# 
# Emotion Recognition Video Processing Script
#
# This script runs the emotion recognition module to detect and analyze 
# facial emotions in video files.
#
# Usage:
#   .\run_emotion_recognition.ps1                     # Interactive mode (default)
#   .\run_emotion_recognition.ps1 process C:\path\to\video.mp4 --output C:\path\to\output.mp4
#   .\run_emotion_recognition.ps1 batch C:\input\dir C:\output\dir
#   .\run_emotion_recognition.ps1 interactive --input_dir C:\path\to\videos
#   .\run_emotion_recognition.ps1 check

# Get the directory where this script is located
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Get the project root directory (parent of scripts directory)
$ProjectRoot = Split-Path -Parent $ScriptDir

# Activate virtual environment if it exists
if (Test-Path "$ProjectRoot\venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..."
    & "$ProjectRoot\venv\Scripts\Activate.ps1"
}

# Check for Python
try {
    $PythonVersion = python --version
}
catch {
    Write-Error "Error: Python is not installed or not in PATH"
    exit 1
}

# Install main dependencies individually to avoid conflicts
Write-Host "Installing dependencies..."
pip install "numpy>=1.25.0" "pandas>=2.1.0"
pip install "matplotlib>=3.7.0"  # Install compatible matplotlib version first
pip install "opencv-python>=4.8.0.76"
pip install "tensorflow>=2.13.0" tf-keras  # Core dependencies for emotion recognition

# Check for DeepFace
Write-Host "Checking for DeepFace dependency..."
try {
    python -c "import deepface"
    Write-Host "DeepFace is already installed."
}
catch {
    Write-Host "DeepFace not found. Attempting to install..."
    pip install deepface
    
    # Verify installation
    try {
        python -c "import deepface"
        Write-Host "DeepFace installed successfully."
    }
    catch {
        Write-Error "Error: Failed to install DeepFace. Please install manually with: pip install deepface"
        exit 1
    }
}

# Install MediaPipe for body pose estimation
Write-Host "Checking for MediaPipe dependency..."
try {
    python -c "import mediapipe"
    Write-Host "MediaPipe is already installed."
}
catch {
    Write-Host "MediaPipe not found. Attempting to install..."
    pip install mediapipe
    
    # Verify installation
    try {
        python -c "import mediapipe"
        Write-Host "MediaPipe installed successfully."
    }
    catch {
        Write-Host "Warning: Failed to install MediaPipe. Body pose estimation may not work properly." -ForegroundColor Yellow
    }
}

# Make sure the default directories exist
$dataDir = Join-Path $ProjectRoot "data\videos"
$outputDir = Join-Path $ProjectRoot "output\emotions"

if (-not (Test-Path $dataDir)) {
    Write-Host "Creating data directory: $dataDir" -ForegroundColor Green
    New-Item -Path $dataDir -ItemType Directory -Force | Out-Null
}

if (-not (Test-Path $outputDir)) {
    Write-Host "Creating output directory: $outputDir" -ForegroundColor Green
    New-Item -Path $outputDir -ItemType Directory -Force | Out-Null
}

# Add the project root to PYTHONPATH to allow imports
$env:PYTHONPATH = "$ProjectRoot;$env:PYTHONPATH"

Write-Host "Running Emotion Recognition module with body pose estimation enabled by default..." -ForegroundColor Green

# Check if --no-pose is already in the arguments
$noPosePresent = $false
foreach ($arg in $args) {
    if ($arg -eq "--no-pose") {
        $noPosePresent = $true
        break
    }
}

# Run the emotion recognition module with all arguments passed to this script
if ($noPosePresent) {
    # If --no-pose is already in the arguments, don't add --with-pose
    python -m src.emotion_recognition.cli $args
} else {
    # Add --with-pose to arguments (it's now the default)
    python -m src.emotion_recognition.cli --with-pose $args
}

# Exit with the same code as the python command
exit $LASTEXITCODE
