# Data Analysis Project

## Setup Instructions

### Option 1: With Poetry

```bash
# Install dependencies
poetry install
```

### Option 2: Without Poetry

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
.
├── notebooks/       # Jupyter notebooks
├── utils/           # User defined utility functions
├── report/          # Report documentation
├── poetry.lock      # Poetry lock file
├── pyproject.toml   # Poetry dependencies (for Poetry users)
├── README.md
└── requirements.txt # Pip dependencies (for non-Poetry users)
```

## Adding New Dependencies

**If you're using Poetry:**

```bash
poetry add <package-name>
poetry run pip freeze > requirements.txt 
git add pyproject.toml poetry.lock requirements.txt
git commit -m "Add <package-name>"
```

**If you're NOT using Poetry:**

```bash
# Ask the project maintainer to add it to Poetry
pip install <package-name>
```
