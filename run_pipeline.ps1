# 
# Complete Video Processing Pipeline Script
#
# This script runs the entire video processing pipeline in the following order:
# 1. Download videos from SharePoint
# 2. Process videos in parallel:
#    - Speech separation followed by speech-to-text transcription
#    - Emotion and body pose recognition

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "Video Data Processing Pipeline" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host "This script will run all processing steps in sequence."

# Get the directory where this script is located
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
# Set ProjectRoot to script directory (since script is in project root)
$ProjectRoot = $ScriptDir

# Check if Python is installed
try {
    $pythonVersion = python --version
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
}
catch {
    Write-Error "Python not found. Please install Python 3.12 or later."
    exit 1
}

# Check if Poetry is installed
$poetryInstalled = $null
try {
    $poetryInstalled = (Get-Command poetry -ErrorAction Stop) -ne $null
} catch {
    $poetryInstalled = $false
}

if (-not $poetryInstalled) {
    Write-Host "Poetry not found. Installing Poetry..." -ForegroundColor Yellow
    
    # Install Poetry using the PowerShell installer script
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
    
    # Add Poetry to the PATH for this session
    $env:PATH = "$env:USERPROFILE\.poetry\bin;$env:PATH"
    
    # Verify installation was successful
    try {
        $poetryInstalled = (Get-Command poetry -ErrorAction Stop) -ne $null
        if ($poetryInstalled) {
            Write-Host "Poetry installed successfully." -ForegroundColor Green
        } else {
            Write-Host "Failed to install Poetry. Please install manually from https://python-poetry.org/docs/#installation" -ForegroundColor Red
            exit 1
        }
    } catch {
        Write-Host "Failed to install Poetry. Please install manually from https://python-poetry.org/docs/#installation" -ForegroundColor Red
        exit 1
    }
}

# Configure Poetry to create the virtualenv in the project directory
Write-Host "Configuring Poetry..." -ForegroundColor Green
poetry config virtualenvs.in-project true

# Set environment variables to suppress model download output
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$env:TRANSFORMERS_VERBOSITY = "error"
$env:TOKENIZERS_PARALLELISM = "false"
$env:PYTHONWARNINGS = "ignore"

# Install base dependencies
Write-Host "Installing common dependencies..." -ForegroundColor Green
Push-Location $ProjectRoot
poetry install --no-root
Pop-Location

# Create and ensure essential directories exist
$DataDir = Join-Path $ProjectRoot "data"
$VideosDir = Join-Path $DataDir "videos"
$OutputDir = Join-Path $ProjectRoot "output"
New-Item -Path $DataDir -ItemType Directory -Force | Out-Null
New-Item -Path $VideosDir -ItemType Directory -Force | Out-Null
New-Item -Path $OutputDir -ItemType Directory -Force | Out-Null
Write-Host "Ensuring data directories exist: $VideosDir" -ForegroundColor Green

# Create a timestamp for the pipeline run
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$pipelineDir = Join-Path $OutputDir "pipeline_results_${timestamp}"
$timestampedVideosDir = Join-Path $DataDir "videos_${timestamp}"

# Create pipeline directory
New-Item -Path $pipelineDir -ItemType Directory -Force | Out-Null
Write-Host "Results will be stored in: $pipelineDir" -ForegroundColor Yellow

# Ask user about video source
Write-Host "`n===== Video Source Selection =====" -ForegroundColor Cyan
Write-Host "Please select how you want to provide input videos:" -ForegroundColor Yellow
Write-Host "1. Use existing videos from $VideosDir" -ForegroundColor White
Write-Host "2. Download new videos from SharePoint" -ForegroundColor White

$videoChoice = Read-Host "Enter your choice (1 or 2)"

switch ($videoChoice) {
    "1" {
        Write-Host "Using existing videos from $VideosDir..." -ForegroundColor Green
        $videoDir = $VideosDir
        
        # Check if directory has video files
        $videoFiles = Get-ChildItem -Path $videoDir -Filter "*.mp4" -Recurse
        $videoFiles += Get-ChildItem -Path $videoDir -Filter "*.avi" -Recurse
        $videoFiles += Get-ChildItem -Path $videoDir -Filter "*.mov" -Recurse
        $videoFiles += Get-ChildItem -Path $videoDir -Filter "*.mkv" -Recurse
        
        if ($videoFiles.Count -eq 0) {
            Write-Host "No video files found in $videoDir!" -ForegroundColor Red
            $downloadChoice = Read-Host "Would you like to download videos instead? (y/n)"
            if ($downloadChoice -match "^[Yy]") {
                Write-Host "Proceeding to video download..." -ForegroundColor Green
                $videoChoice = "2"
                New-Item -Path $timestampedVideosDir -ItemType Directory -Force | Out-Null
                $videoDir = $timestampedVideosDir
            } else {
                Write-Host "Please add video files to $videoDir and run the script again." -ForegroundColor Red
                exit 1
            }
        } else {
            Write-Host "Found $($videoFiles.Count) video file(s) in $videoDir" -ForegroundColor Green
        }
    }
    "2" {
        Write-Host "Downloading videos to timestamped directory..." -ForegroundColor Green
        New-Item -Path $timestampedVideosDir -ItemType Directory -Force | Out-Null
        $videoDir = $timestampedVideosDir
    }
    default {
        Write-Host "Invalid choice. Using existing videos by default." -ForegroundColor Yellow
        $videoDir = $VideosDir
    }
}

# Step 1: Download videos (only if selected)
$downloadExit = 0
if ($videoChoice -eq "2") {
    Write-Host "`n===== Step 1: Download Videos =====" -ForegroundColor Green
    
    # Install download dependencies
    Write-Host "Installing download dependencies..." -ForegroundColor Green
    Push-Location $ProjectRoot
    poetry install --only download
    Pop-Location
    
    # Run the download script with poetry
    Push-Location $ProjectRoot
    poetry run python -m src.download_videos --output-dir $videoDir @args
    $downloadExit = $LASTEXITCODE
    Pop-Location

    if ($downloadExit -ne 0) {
        Write-Host "Video download failed or was canceled (exit code $downloadExit)." -ForegroundColor Red
        Write-Host "Pipeline will use any existing videos in $VideosDir directory." -ForegroundColor Yellow
        $videoDir = $VideosDir
    } else {
        Write-Host "Video download completed successfully to $videoDir." -ForegroundColor Green
    }
} else {
    Write-Host "`n===== Step 1: Download Videos (Skipped) =====" -ForegroundColor Yellow
}

# Step 2a: Speech Separation (in background)
Write-Host "`n===== Step 2a: Speech Separation =====" -ForegroundColor Green
$speechJob = Start-Job -ScriptBlock {
    param($projectRoot, $pipelineDir, $videoDir)
    
    # Change to project root to access Poetry config
    Set-Location $projectRoot
    
    # Set environment variables in the job
    $env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
    $env:TRANSFORMERS_VERBOSITY = "error"
    $env:TOKENIZERS_PARALLELISM = "false"
    $env:PYTHONWARNINGS = "ignore"
    
    # Install speech dependencies in an isolated environment
    Write-Host "Installing speech dependencies..." -ForegroundColor Yellow
    poetry install --with common --with speech --no-root
    
    # Run speech separation (remove quiet flag)
    $speechExit = 0
    try {
        poetry run python -m src.separate_speech --output-dir "${pipelineDir}\speech" "${videoDir}"
        $speechExit = $LASTEXITCODE
    } catch {
        $speechExit = 1
    }
    
    # If speech separation succeeded, run speech-to-text (remove quiet flag)
    $sttExit = 1
    if ($speechExit -eq 0) {
        try {
            poetry run python -m src.speech_to_text --input-dir "${pipelineDir}\speech" --output-dir "${pipelineDir}\transcripts"
            $sttExit = $LASTEXITCODE
        } catch {
            $sttExit = 1
        }
    }
    
    # Return exit codes
    return @{
        SpeechExit = $speechExit
        SttExit = $sttExit
    }
} -ArgumentList $ProjectRoot, $pipelineDir, $videoDir

# Step 2b: Emotion Recognition (in background)
Write-Host "`n===== Step 2b: Emotion Recognition =====" -ForegroundColor Green
$emotionJob = Start-Job -ScriptBlock {
    param($projectRoot, $pipelineDir, $videoDir)
    
    # Change to project root to access Poetry config
    Set-Location $projectRoot
    
    # Set environment variables in the job
    $env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
    $env:TRANSFORMERS_VERBOSITY = "error"
    $env:TOKENIZERS_PARALLELISM = "false"
    $env:PYTHONWARNINGS = "ignore"
    
    # Install emotion dependencies in a separate isolated environment
    Write-Host "Installing emotion dependencies..." -ForegroundColor Yellow
    poetry install --with common --with emotion --no-root
    
    # Run emotion recognition (remove quiet flag)
    $emotionExit = 0
    try {
        poetry run python -m src.emotion_recognition.cli --output-dir "${pipelineDir}\emotions" "${videoDir}"
        $emotionExit = $LASTEXITCODE
    } catch {
        $emotionExit = 1
    }
    
    # Return exit code
    return $emotionExit
} -ArgumentList $ProjectRoot, $pipelineDir, $videoDir

# Wait for speech separation to complete
Write-Host "`nWaiting for speech separation to complete..." -ForegroundColor Yellow
$speechJob | Wait-Job | Out-Null
$speechResult = Receive-Job -Job $speechJob
$speechExit = $speechResult.SpeechExit
$sttExit = $speechResult.SttExit
Remove-Job -Job $speechJob

Write-Host "`n===== Speech separation completed with status: $speechExit =====" -ForegroundColor $(if ($speechExit -eq 0) {"Green"} else {"Red"})

# Step 3: Speech-to-text (only if speech separation succeeded)
if ($speechExit -eq 0) {
    Write-Host "`n===== Speech to Text completed with status: $sttExit =====" -ForegroundColor $(if ($sttExit -eq 0) {"Green"} else {"Red"})
} else {
    Write-Host "Speech separation failed. Skipping speech to text step." -ForegroundColor Yellow
}

# Wait for emotion recognition to complete
Write-Host "`nWaiting for emotion recognition to complete..." -ForegroundColor Yellow
$emotionJob | Wait-Job | Out-Null
$emotionResult = Receive-Job -Job $emotionJob
$emotionExit = $emotionResult
Remove-Job -Job $emotionJob

Write-Host "`n===== Emotion recognition completed with status: $emotionExit =====" -ForegroundColor $(if ($emotionExit -eq 0) {"Green"} else {"Red"})

# Summarize results
Write-Host "`n==================================" -ForegroundColor Cyan
Write-Host "Pipeline Execution Summary" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host "1. Download Videos: $(if ($downloadExit -eq 0) {"SUCCESS"} else {"FAILED"})" -ForegroundColor $(if ($downloadExit -eq 0) {"Green"} else {"Red"})
Write-Host "2. Speech Separation: $(if ($speechExit -eq 0) {"SUCCESS"} else {"FAILED"})" -ForegroundColor $(if ($speechExit -eq 0) {"Green"} else {"Red"})
Write-Host "3. Speech to Text: $(if ($sttExit -eq 0) {"SUCCESS"} else {"FAILED"})" -ForegroundColor $(if ($sttExit -eq 0) {"Green"} else {"Red"})
Write-Host "4. Emotion Recognition: $(if ($emotionExit -eq 0) {"SUCCESS"} else {"FAILED"})" -ForegroundColor $(if ($emotionExit -eq 0) {"Green"} else {"Red"})
Write-Host "`nResults saved to: $pipelineDir" -ForegroundColor Yellow

# Overall success
if (($speechExit -eq 0) -and ($sttExit -eq 0) -and ($emotionExit -eq 0)) {
    Write-Host "`nPipeline completed successfully!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "`nPipeline completed with some errors. Check the logs for details." -ForegroundColor Yellow
    exit 1
}
