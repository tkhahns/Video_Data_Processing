# Video Data Processing - Complete Pipeline
Write-Host "=== Video Data Processing - Complete Pipeline ===" -ForegroundColor Cyan
Write-Host "This script downloads videos to a timestamped directory and processes them." -ForegroundColor Cyan

# Get the script's directory (project root)
$PROJECT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path

# Create timestamped directory
$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$DOWNLOADS_DIR = Join-Path -Path $PROJECT_ROOT -ChildPath "data\downloads_$TIMESTAMP"
$RESULTS_DIR = Join-Path -Path $PROJECT_ROOT -ChildPath "output\pipeline_results_$TIMESTAMP"
$SPEECH_OUTPUT_DIR = Join-Path -Path $RESULTS_DIR -ChildPath "speech"
$TRANSCRIPT_OUTPUT_DIR = Join-Path -Path $RESULTS_DIR -ChildPath "transcripts"
$EMOTIONS_AND_POSE_DIR = Join-Path -Path $RESULTS_DIR -ChildPath "emotions_and_pose"

Write-Host "`nCreating timestamped directories:" -ForegroundColor Green
Write-Host "- Downloaded Videos: $DOWNLOADS_DIR" -ForegroundColor Yellow
Write-Host "- Pipeline results: $RESULTS_DIR" -ForegroundColor Yellow
Write-Host "  |- Speech: speech/" -ForegroundColor Yellow
Write-Host "  |- Transcripts: transcripts/" -ForegroundColor Yellow
Write-Host "  |- Emotions and Pose: emotions_and_pose/" -ForegroundColor Yellow

# Create directories
New-Item -Path $DOWNLOADS_DIR -ItemType Directory -Force | Out-Null
New-Item -Path $SPEECH_OUTPUT_DIR -ItemType Directory -Force | Out-Null
New-Item -Path $TRANSCRIPT_OUTPUT_DIR -ItemType Directory -Force | Out-Null
New-Item -Path $EMOTIONS_AND_POSE_DIR -ItemType Directory -Force | Out-Null

# Change to project root
Set-Location -Path $PROJECT_ROOT

# Check if Poetry is installed
if (Get-Command poetry -ErrorAction SilentlyContinue) {
    Write-Host "`n[1/6] Installing dependencies using Poetry..." -ForegroundColor Green
    
    # Install all dependencies from pyproject.toml
    Write-Host "Installing base dependencies..." -ForegroundColor Cyan
    poetry install --no-interaction
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install base dependencies" -ForegroundColor Red
        exit 1
    }
    
    # Install specific groups
    Write-Host "Installing common dependencies..." -ForegroundColor Cyan
    poetry install --with common --no-interaction
    if ($LASTEXITCODE -ne 0) { 
        Write-Host "Warning: Some common dependencies failed to install" -ForegroundColor Yellow
    }
    
    Write-Host "Installing emotion recognition dependencies..." -ForegroundColor Cyan
    poetry install --with emotion --no-interaction
    if ($LASTEXITCODE -ne 0) { 
        Write-Host "Warning: Some emotion recognition dependencies failed to install" -ForegroundColor Yellow
    }
    
    Write-Host "Installing speech recognition dependencies..." -ForegroundColor Cyan
    poetry install --with speech --no-interaction
    if ($LASTEXITCODE -ne 0) { 
        Write-Host "Warning: Some speech recognition dependencies failed to install" -ForegroundColor Yellow
    }
    
    Write-Host "Installing download dependencies..." -ForegroundColor Cyan
    poetry install --with download --no-interaction
    if ($LASTEXITCODE -ne 0) { 
        Write-Host "Warning: Some download dependencies failed to install" -ForegroundColor Yellow
    }
    
    Write-Host "Dependencies installation completed." -ForegroundColor Green
    
    # STEP 1: Run video downloader
    Write-Host "`n[3/6] Running video downloader with Poetry..." -ForegroundColor Green
    
    # Build download command
    $downloadArgs = @()
    # Add original arguments except --url
    $urlProvided = $false
    $i = 0
    while ($i -lt $args.Count) {
        if ($args[$i] -eq "--url" -and $i+1 -lt $args.Count) {
            $url = $args[$i+1]
            $urlProvided = $true
            $i += 2
        } else {
            $downloadArgs += $args[$i]
            $i++
        }
    }
    # Add output directory
    $downloadArgs += "--output-dir", $DOWNLOADS_DIR
    
    # Call downloader script
    if ($urlProvided) {
        poetry run python -m src.download_videos --url $url $downloadArgs
    } else {
        # Prompt for URL
        Write-Host "Please enter SharePoint URL containing videos:" -ForegroundColor Yellow
        $url = Read-Host
        poetry run python -m src.download_videos --url $url $downloadArgs
    }
    $DOWNLOAD_EXIT = $LASTEXITCODE
    
    # STEP 2: Run speech separation if download was successful
    if ($DOWNLOAD_EXIT -eq 0) {
        Write-Host "`n[4/6] Running speech separation on downloaded videos..." -ForegroundColor Green
        poetry run python -m src.separate_speech --input-dir $DOWNLOADS_DIR --output-dir $SPEECH_OUTPUT_DIR
        $SPEECH_EXIT = $LASTEXITCODE
        
        # STEP 3: Run speech-to-text if speech separation was successful
        if ($SPEECH_EXIT -eq 0) {
            Write-Host "`n[5/6] Running speech-to-text on separated audio..." -ForegroundColor Green
            poetry run python -m src.speech_to_text --input-dir $SPEECH_OUTPUT_DIR --output-dir $TRANSCRIPT_OUTPUT_DIR
            $TRANSCRIPT_EXIT = $LASTEXITCODE
            
            # STEP 4: Run emotion recognition on the original videos
            Write-Host "`n[6/6] Running emotion recognition on downloaded videos..." -ForegroundColor Green
            poetry run python -m src.emotion_recognition.cli --input-dir $DOWNLOADS_DIR --output-dir $EMOTIONS_AND_POSE_DIR --with-pose
            $EMOTION_EXIT = $LASTEXITCODE
            
            # Report the final status of all pipeline steps
            Write-Host "`n===== Pipeline Execution Summary =====" -ForegroundColor Cyan
            Write-Host "- Video Download: $(if ($DOWNLOAD_EXIT -eq 0) { "✅ Success" } else { "❌ Failed" })" -ForegroundColor $(if ($DOWNLOAD_EXIT -eq 0) { "Green" } else { "Red" })
            Write-Host "- Speech Separation: $(if ($SPEECH_EXIT -eq 0) { "✅ Success" } else { "❌ Failed" })" -ForegroundColor $(if ($SPEECH_EXIT -eq 0) { "Green" } else { "Red" })
            Write-Host "- Speech-to-Text: $(if ($TRANSCRIPT_EXIT -eq 0) { "✅ Success" } else { "❌ Failed" })" -ForegroundColor $(if ($TRANSCRIPT_EXIT -eq 0) { "Green" } else { "Red" })
            Write-Host "- Emotion Recognition with Pose: $(if ($EMOTION_EXIT -eq 0) { "✅ Success" } else { "❌ Failed" })" -ForegroundColor $(if ($EMOTION_EXIT -eq 0) { "Green" } else { "Red" })
            
            Write-Host "`nResults and outputs:"
            Write-Host "- Downloaded videos: $DOWNLOADS_DIR"
            Write-Host "- Separated speech: $SPEECH_OUTPUT_DIR"
            Write-Host "- Transcripts: $TRANSCRIPT_OUTPUT_DIR"
            Write-Host "- Emotion and pose analysis: $EMOTIONS_AND_POSE_DIR"
        } else {
            Write-Host "`nSpeech separation failed with exit code $SPEECH_EXIT" -ForegroundColor Red
            Write-Host "Downloaded videos are still available at: $DOWNLOADS_DIR" -ForegroundColor Yellow
        }
    } else {
        Write-Host "`nVideo download failed or was canceled (exit code $DOWNLOAD_EXIT)." -ForegroundColor Red
        Write-Host "Pipeline halted." -ForegroundColor Red
    }
} else {
    Write-Host "`nPoetry not found. Cannot run the complete pipeline without Poetry." -ForegroundColor Red
    Write-Host "Please install Poetry: (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -" -ForegroundColor Yellow
    exit 1
}
