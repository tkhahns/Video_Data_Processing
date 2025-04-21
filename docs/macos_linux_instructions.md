# Video Data Processing Setup Instructions

These instructions will help you set up the environment for the Video Data Processing application on macOS or Linux systems using Python 3.13.3.

## Prerequisites

Ensure Python 3.13.3 is installed on your system:

```bash
# Check Python version
python3 --version

# If needed, install Python 3.13.3
# For macOS (using Homebrew):
brew install python@3.13

# For Linux (Ubuntu/Debian):
# Use pyenv or your preferred method to install Python 3.13.3
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/tkhahns/Video_Data_Processing
```

### 2. Create and activate a virtual environment

```bash
# Create virtual environment
python3.13 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Update pip and install dependencies

```bash
# Update pip to latest version
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Explicitly reinstall certain packages to avoid import errors
pip uninstall -y huggingface-hub
pip install huggingface-hub
```

### 4. Install system dependencies

For macOS:
```bash
brew install cmake pkg-config protobuf
```

For Linux (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install cmake pkg-config protobuf-compiler
```

### 5. Install additional Python packages

```bash
pip install sentencepiece --no-build-isolation
```

## Verification

Verify the installation by running:

```bash
# Check if key packages are properly installed
python -c "import pandas, transformers, huggingface_hub; print('Installation successful!')"

# Test running a basic script
python models/download_models.py --help
```

## Troubleshooting

- If you encounter `ModuleNotFoundError` despite the package being installed, try:
  ```bash
  pip uninstall [package-name]
  pip install [package-name]
  ```

- For sentencepiece build errors:
  ```bash
  # Try installing without building
  pip install --prefer-binary sentencepiece
  ```

- If you still have import errors, verify you're using the correct Python interpreter:
  ```bash
  which python
  # Should point to your virtual environment
  ```