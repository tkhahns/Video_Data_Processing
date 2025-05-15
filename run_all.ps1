# Video Data Processing - Complete Pipeline

# Record start time
$START_TIME = Get-Date

Write-Host "=== Video Data Processing - Complete Pipeline ===" -ForegroundColor Cyan
Write-Host "This script downloads videos to a timestamped directory and processes them." -ForegroundColor Cyan

# Get the script's directory (project root)
$PROJECT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path

# Hugging Face Token Handling
Write-Host "`n=== Hugging Face Authentication ===" -ForegroundColor Cyan
Write-Host "This tool requires a Hugging Face token for accessing models." -ForegroundColor Yellow
Write-Host "You can get your token from: https://huggingface.co/settings/tokens" -ForegroundColor Yellow
Write-Host "Note: Your token will only be used for this session and will not be saved." -ForegroundColor Yellow

# Prompt for token
$HUGGINGFACE_TOKEN = Read-Host -Prompt "Enter your Hugging Face token" -AsSecureString

# Convert SecureString to plain text for use in environment variable
$BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($HUGGINGFACE_TOKEN)
$TOKEN_VALUE = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)
[System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($BSTR)

if ([string]::IsNullOrEmpty($TOKEN_VALUE)) {
    Write-Host "No token provided. Some features may not work correctly." -ForegroundColor Red
} else {
    Write-Host "Token received for this session" -ForegroundColor Green
}

# Set environment variable
$env:HUGGINGFACE_TOKEN = $TOKEN_VALUE

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
    
    # STEP 2: Proceed if download was successful
    if ($DOWNLOAD_EXIT -eq 0) {
        # Check if any videos were downloaded
        $VIDEO_COUNT = (Get-ChildItem -Path $DOWNLOADS_DIR -Recurse -File | Where-Object { 
            $_.Extension -match "\.(mp4|avi|mov|mkv)$" -or $_.Extension -match "\.(MP4|AVI|MOV|MKV)$" 
        }).Count
        
        if ($VIDEO_COUNT -eq 0) {
            Write-Host "`nNo videos were found in the directory: $DOWNLOADS_DIR" -ForegroundColor Red
            Write-Host "Available files:" -ForegroundColor Yellow
            Get-ChildItem -Path $DOWNLOADS_DIR | Format-Table Name, Length -AutoSize
            Write-Host "Pipeline halted - no videos to process." -ForegroundColor Red
            exit 1
        }
        
        Write-Host "`n$VIDEO_COUNT videos found in the downloads directory:" -ForegroundColor Green
        Get-ChildItem -Path $DOWNLOADS_DIR -Recurse -File | Where-Object { 
            $_.Extension -match "\.(mp4|avi|mov|mkv)$" -or $_.Extension -match "\.(MP4|AVI|MOV|MKV)$" 
        } | ForEach-Object { Write-Host $_.Name }
        
        # Ask user if they want to process all videos or select manually
        Write-Host "`nHow would you like to process the downloaded videos?" -ForegroundColor Yellow
        Write-Host "1. Process all downloaded videos automatically" -ForegroundColor White
        Write-Host "2. Select specific videos to process at each step" -ForegroundColor White
        
        $choice = Read-Host "Enter your choice (1 or 2)"
        $PROCESS_ALL = $false
        $BATCH_FLAG = ""
        
        if ($choice -eq "1") {
            $PROCESS_ALL = $true
            $BATCH_FLAG = "--batch"
            Write-Host "Processing all videos automatically." -ForegroundColor Green
        } else {
            Write-Host "You will be prompted to select videos at each step." -ForegroundColor Yellow
        }
        
        # Run all steps sequentially
        
        # STEP 3: Run speech separation
        Write-Host "`n[4/6] Running speech separation..." -ForegroundColor Green
        & poetry run python -m src.separate_speech --input-dir "$DOWNLOADS_DIR" --output-dir "$SPEECH_OUTPUT_DIR" $BATCH_FLAG
        $SPEECH_EXIT = $LASTEXITCODE
        
        # STEP 4: Run speech-to-text if speech separation was successful
        $TRANSCRIPT_EXIT = 1  # Default to failure
        if ($SPEECH_EXIT -eq 0) {
            Write-Host "`n[5/6] Running speech-to-text on separated audio..." -ForegroundColor Green
            & poetry run python -m src.speech_to_text --input-dir "$SPEECH_OUTPUT_DIR" --output-dir "$TRANSCRIPT_OUTPUT_DIR" --diarize $BATCH_FLAG
            $TRANSCRIPT_EXIT = $LASTEXITCODE
        } else {
            Write-Host "`nSpeech separation failed with exit code $SPEECH_EXIT" -ForegroundColor Red
            Write-Host "Skipping speech-to-text step." -ForegroundColor Yellow
        }
        
        # STEP 5: Run emotion and pose recognition
        Write-Host "`n[6/6] Running emotion and pose recognition..." -ForegroundColor Green
        & poetry run python -m src.emotion_and_pose_recognition.cli --input-dir "$DOWNLOADS_DIR" --output-dir "$EMOTIONS_AND_POSE_DIR" $BATCH_FLAG
        $EMOTION_EXIT = $LASTEXITCODE
        
        # Report the final status of all pipeline steps
        Write-Host "`n===== Pipeline Execution Summary =====" -ForegroundColor Cyan
        Write-Host "- Video Download: $(if ($DOWNLOAD_EXIT -eq 0) { "✅ Success" } else { "❌ Failed" })" -ForegroundColor $(if ($DOWNLOAD_EXIT -eq 0) { "Green" } else { "Red" })
        Write-Host "- Speech Separation: $(if ($SPEECH_EXIT -eq 0) { "✅ Success" } else { "❌ Failed" })" -ForegroundColor $(if ($SPEECH_EXIT -eq 0) { "Green" } else { "Red" })
        Write-Host "- Speech-to-Text: $(if ($TRANSCRIPT_EXIT -eq 0) { "✅ Success" } else { "❌ Failed" })" -ForegroundColor $(if ($TRANSCRIPT_EXIT -eq 0) { "Green" } else { "Red" })
        Write-Host "- Emotion and Pose Recognition: $(if ($EMOTION_EXIT -eq 0) { "✅ Success" } else { "❌ Failed" })" -ForegroundColor $(if ($EMOTION_EXIT -eq 0) { "Green" } else { "Red" })
        
        # Calculate total process time
        $END_TIME = Get-Date
        $ELAPSED_TIME = $END_TIME - $START_TIME
        $TIME_FORMAT = "{0:hh\:mm\:ss}" -f $ELAPSED_TIME
        
        Write-Host "`nTotal process time: $TIME_FORMAT (HH:MM:SS)" -ForegroundColor Cyan
        
        Write-Host "`nResults and outputs:" -ForegroundColor Green
        Write-Host "- Downloaded videos: $DOWNLOADS_DIR"
        Write-Host "- Separated speech: $SPEECH_OUTPUT_DIR"
        Write-Host "- Transcripts: $TRANSCRIPT_OUTPUT_DIR"
        Write-Host "- Emotion and pose analysis: $EMOTIONS_AND_POSE_DIR"
    } else {
        Write-Host "`nVideo download failed or was canceled (exit code $DOWNLOAD_EXIT)." -ForegroundColor Red
        Write-Host "Pipeline halted." -ForegroundColor Red
        
        # Show execution time even if the pipeline was halted
        $END_TIME = Get-Date
        $ELAPSED_TIME = $END_TIME - $START_TIME
        $MINUTES = [int]$ELAPSED_TIME.TotalMinutes
        $SECONDS = [int]$ELAPSED_TIME.Seconds
        Write-Host "Execution time: ${MINUTES}m ${SECONDS}s" -ForegroundColor Yellow
    }
} else {
    Write-Host "`nPoetry not found. Cannot run the complete pipeline without Poetry." -ForegroundColor Red
    Write-Host "Please install Poetry: (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -" -ForegroundColor Yellow
    exit 1
}

# Clear the token from the environment
$env:HUGGINGFACE_TOKEN = ""
