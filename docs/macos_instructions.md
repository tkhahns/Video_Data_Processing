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

# Reinstall problematic packages if needed
pip uninstall -y huggingface-hub
pip install huggingface-hub
```

### 4. Install system dependencies (macOS)

```bash
brew install cmake pkg-config protobuf
```

### 5. Install additional Python packages

```bash
pip install sentencepiece --no-build-isolation
```

---

## Verification

Verify the installation by running:

```bash
# Check if key packages are properly installed
python -c "import pandas, transformers, huggingface_hub; print('Installation successful!')"

# Test running a basic script
python models/download_models.py --help
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