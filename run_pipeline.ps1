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

# Check if Python is installed
try {
    $pythonVersion = python --version
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
}
catch {
    Write-Error "Python not found. Please install Python 3.12 or later."
    exit 1
}

# Check for and create virtual environment if needed
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    Write-Host "Found existing virtual environment (.venv)" -ForegroundColor Green
} elseif (Test-Path ".\venv\Scripts\Activate.ps1") {
    Write-Host "Found existing virtual environment (venv)" -ForegroundColor Green
} else {
    Write-Host "No virtual environment found. Creating one (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
    if (-not $?) {
        Write-Error "Failed to create virtual environment. Please check your Python installation."
        exit 1
    }
    Write-Host "Virtual environment created successfully." -ForegroundColor Green
}

# Activate virtual environment
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Green
    & ".\.venv\Scripts\Activate.ps1"
} elseif (Test-Path ".\venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Green
    & ".\venv\Scripts\Activate.ps1"
}

# Install dependencies from requirements.txt
Write-Host "Installing required dependencies..." -ForegroundColor Green
pip install -r requirements.txt

# Create a timestamp for the pipeline run
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$pipelineDir = "pipeline_results_${timestamp}"

# Create pipeline directory
New-Item -Path $pipelineDir -ItemType Directory -Force | Out-Null
Write-Host "Results will be stored in: $pipelineDir" -ForegroundColor Yellow

# Step 1: Download videos
Write-Host "`n===== Step 1: Download Videos =====" -ForegroundColor Green
& ".\scripts\windows\run_download_videos.ps1" --output-dir "${pipelineDir}\videos" @args
$downloadExit = $LASTEXITCODE

if ($downloadExit -ne 0) {
    Write-Host "Video download failed or was canceled (exit code $downloadExit)." -ForegroundColor Red
    Write-Host "Pipeline will use any existing videos in data\videos directory." -ForegroundColor Yellow
    $videoDir = "data\videos"
} else {
    Write-Host "Video download completed successfully." -ForegroundColor Green
    $videoDir = "${pipelineDir}\videos"
}

# Step 2a: Speech Separation (in background)
Write-Host "`n===== Step 2a: Speech Separation =====" -ForegroundColor Green
$speechJob = Start-Job -ScriptBlock {
    param($scriptPath, $outputDir, $videoDir)
    & "$scriptPath" --output-dir "$outputDir" "$videoDir"
    return $LASTEXITCODE
} -ArgumentList "$(Get-Location)\scripts\windows\run_separate_speech.ps1", "${pipelineDir}\speech", $videoDir

# Step 2b: Emotion Recognition (in background)
Write-Host "`n===== Step 2b: Emotion Recognition =====" -ForegroundColor Green
$emotionJob = Start-Job -ScriptBlock {
    param($scriptPath, $outputDir, $videoDir)
    & "$scriptPath" --output-dir "$outputDir" "$videoDir"
    return $LASTEXITCODE
} -ArgumentList "$(Get-Location)\scripts\windows\run_emotion_recognition.ps1", "${pipelineDir}\emotions", $videoDir

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
    & ".\scripts\windows\run_speech_to_text.ps1" --input-dir "${pipelineDir}\speech" --output-dir "${pipelineDir}\transcripts"
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
