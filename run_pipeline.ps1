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

# Check for and create virtual environment if needed - always use absolute paths
$VenvPath = Join-Path $ProjectRoot ".venv"
$AltVenvPath = Join-Path $ProjectRoot "venv"

# Always use absolute paths to avoid creating venvs in wrong locations
if (Test-Path (Join-Path $VenvPath "Scripts\Activate.ps1")) {
    Write-Host "Found existing virtual environment at $VenvPath" -ForegroundColor Green
} elseif (Test-Path (Join-Path $AltVenvPath "Scripts\Activate.ps1")) {
    Write-Host "Found existing virtual environment at $AltVenvPath" -ForegroundColor Green
} else {
    Write-Host "No virtual environment found. Creating one at $VenvPath..." -ForegroundColor Yellow
    # Change to project root directory before creating the venv
    Push-Location $ProjectRoot
    python -m venv .venv
    Pop-Location
    
    if (-not $?) {
        Write-Error "Failed to create virtual environment. Please check your Python installation."
        exit 1
    }
    Write-Host "Virtual environment created successfully at $VenvPath." -ForegroundColor Green
}

# Activate virtual environment using absolute paths
if (Test-Path (Join-Path $VenvPath "Scripts\Activate.ps1")) {
    Write-Host "Activating virtual environment from $VenvPath..." -ForegroundColor Green
    & (Join-Path $VenvPath "Scripts\Activate.ps1")
} elseif (Test-Path (Join-Path $AltVenvPath "Scripts\Activate.ps1")) {
    Write-Host "Activating virtual environment from $AltVenvPath..." -ForegroundColor Green
    & (Join-Path $AltVenvPath "Scripts\Activate.ps1")
}

# Install dependencies from requirements.txt using absolute path
Write-Host "Installing required dependencies..." -ForegroundColor Green
pip install -r (Join-Path $ProjectRoot "requirements.txt")

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
    & "$ProjectRoot\scripts\windows\run_download_videos.ps1" --output-dir $videoDir @args
    $downloadExit = $LASTEXITCODE

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
    param($scriptPath, $outputDir, $videoDir)
    & "$scriptPath" --output-dir "$outputDir" "$videoDir"
    return $LASTEXITCODE
} -ArgumentList "$ProjectRoot\scripts\windows\run_separate_speech.ps1", "${pipelineDir}\speech", $videoDir

# Step 2b: Emotion Recognition (in background)
Write-Host "`n===== Step 2b: Emotion Recognition =====" -ForegroundColor Green
$emotionJob = Start-Job -ScriptBlock {
    param($scriptPath, $outputDir, $videoDir)
    & "$scriptPath" --output-dir "$outputDir" "$videoDir"
    return $LASTEXITCODE
} -ArgumentList "$ProjectRoot\scripts\windows\run_emotion_recognition.ps1", "${pipelineDir}\emotions", $videoDir

# Wait for speech separation to complete
Write-Host "`nWaiting for speech separation to complete..." -ForegroundColor Yellow
$speechJob | Wait-Job | Out-Null
$speechResult = Receive-Job -Job $speechJob
$speechExit = $speechJob.ChildJobs[0].Output[-1]
Remove-Job -Job $speechJob

Write-Host "`n===== Speech separation completed with status: $speechExit =====" -ForegroundColor $(if ($speechExit -eq 0) {"Green"} else {"Red"})

# Step 3: Speech-to-text (only if speech separation succeeded)
$sttExit = 1
if ($speechExit -eq 0) {
    Write-Host "`n===== Step 3: Speech to Text =====" -ForegroundColor Green
    & "$ProjectRoot\scripts\windows\run_speech_to_text.ps1" --input-dir "${pipelineDir}\speech" --output-dir "${pipelineDir}\transcripts"
    $sttExit = $LASTEXITCODE
    Write-Host "`n===== Speech to text completed with status: $sttExit =====" -ForegroundColor $(if ($sttExit -eq 0) {"Green"} else {"Red"})
} else {
    Write-Host "Speech separation failed. Skipping speech to text step." -ForegroundColor Yellow
}

# Wait for emotion recognition to complete
Write-Host "`nWaiting for emotion recognition to complete..." -ForegroundColor Yellow
$emotionJob | Wait-Job | Out-Null
$emotionResult = Receive-Job -Job $emotionJob
$emotionExit = $emotionJob.ChildJobs[0].Output[-1]
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
