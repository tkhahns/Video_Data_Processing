## Video Data Processing Setup Instructions

These instructions will help you set up the environment for the Video Data Processing application on Windows.

---

## Prerequisites

Ensure you have **Python 3.12.10** installed on your system:

```bash
# Download and install Python 3.12.10 from the official Python website
# https://www.python.org/downloads/release/python-31210/

# Verify the Python version (in Command Prompt or PowerShell)
python --version    # → Python 3.12.10
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
# Create the virtual environment
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\activate
```

### 3. Update pip and install dependencies

```bash
# Update pip to the latest version
python -m pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# Install huggingface-hub
pip install huggingface-hub
```

---

## Fetch all models

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

- If you're having permission issues, try running Command Prompt or PowerShell as Administrator

- If you still have import errors, verify you're using the correct Python interpreter:
  ```bash
  where python
  # Should point to your virtual environment's .venv\Scripts\python.exe
  ```