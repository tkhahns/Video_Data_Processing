## Video Data Processing Setup Instructions

These instructions will help you set up the environment for the Video Data Processing application on macOS using Python 3.12.10 managed by pyenv.

---

## Prerequisites

Ensure you have **pyenv** installed and Python 3.12.10 set as your global interpreter:

```bash
# Install pyenv (if not already installed)
brew update
brew install pyenv

# Initialize pyenv in your shell (if not already configured)
# Add to your ~/.zprofile:
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"

# Add to your ~/.zshrc:
eval "$(pyenv init -)"

# Install and set Python 3.12.10 globally
pyenv install --skip-existing 3.12.10
pyenv global 3.12.10

# Verify the Python version
python --version    # → Python 3.12.10
python3 --version   # → Python 3.12.10
``` 

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/tkhahns/Video_Data_Processing.git
cd Video_Data_Processing
```

### 2. Create and activate a virtual environment

```bash
# Create the virtual environment using the specified Python
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```  

### 3. Update pip and install dependencies

```bash
# Update pip to the latest version
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# Install huggingface-hub
pip install huggingface-hub
```

---

## Fetch all models

You can download the required models using either of these methods:

### Option 1: Using the automated script (recommended)

```bash
# Make the script executable
chmod +x run_download.sh

# Run the script
./run_download.sh
```

This script automatically:
- Creates and activates the virtual environment
- Updates pip and installs dependencies
- Creates the models directory
- Downloads all required models

### Option 2: Manual download

```bash
python download_models.py
```

---

## Troubleshooting

- If you encounter `ModuleNotFoundError` despite the package being installed, try:
  ```bash
  pip uninstall [package-name]
  pip install [package-name]
  ```

- For sentencepiece build errors:
  ```bash
  # Install without building isolation
  pip install --prefer-binary sentencepiece
  ```

- If you still have import errors, verify you're using the correct Python interpreter:
  ```bash
  which python
  # Should point to your virtual environment's .venv/bin/python
  ```

---

## Download Videos from SharePoint

You can download videos from SharePoint using the included browser automation tool:

```bash
# Basic usage
python -m download_videos --url "https://your-sharepoint-site.com/folder-with-videos"

# Save to a specific directory
python -m download_videos --url "https://your-sharepoint-site.com/folder-with-videos" --output-dir "./my-videos"

# Just list files without downloading
python -m download_videos --url "https://your-sharepoint-site.com/folder-with-videos" --list-only

# Enable debug mode for troubleshooting
python -m download_videos --url "https://your-sharepoint-site.com/folder-with-videos" --debug
```

This will:
1. Open a browser window for SharePoint authentication
2. Find all available files in the specified folder
3. Present you with a list of files to choose from
4. Download your selected files

**Note:** The tool requires authentication to SharePoint. You'll need to sign in through the browser window that opens.