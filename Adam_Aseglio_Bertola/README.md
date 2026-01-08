# MyAnimeList Data Analysis Pipeline

A comprehensive data analysis project exploring the MyAnimeList (MAL) ecosystem through Python and Jupyter Notebooks.

## Project Overview

This project implements a complete data analysis pipeline that:  
- **Loads and profiles** 13 MAL datasets   
- **Cleans and preprocesses** data while maintaining relational integrity  
- **Visualizes patterns** across anime content, characters, creators, and user behavior  
- **Answers the question:** *What defines success in the anime industry?*  

### Key Features

**Macro-to-Micro Exploration**: From global anime geography to individual creator networks  
**Relational Data Handling**: Preserves cross-dataset integrity for complex queries  
**Interactive Visualizations**: Genre trends, user engagement, format preferences, and more  
**Memory Optimized**: Downcasting utilities for handling large datasets efficiently  

## Project Structure

```

.
└── Adam_Aseglio_Bertola
     ├── README.md
     ├── report
     │    └── report.pdf        # Documentation
     └── solution
          ├── notebooks         # Jupyter notebooks for each task
          ├── poetry.lock       # Poetry dependency management
          ├── pyproject.toml    # Poetry configuration
          ├── requirements.txt  # Python dependencies
          └── utils             # Custom utility functions and modules

```

## Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab (any integrated IDE)
- MAL datasets (update the `INPUT_DIR_PATH` variable in notebooks and `graphs_utils.py`, default is `~/datasets/`)  

### Dependencies Installation

**Option 1: Poetry Users**  

```bash
poetry install
poetry shell
```

**Option 2: Non-Poetry Users**

```bash
# Create virtual environment
python3 -m venv venv     # Mac/Linux
python -m venv venv      # Windows

# Activate
source venv/bin/activate # Mac/Linux
venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

**Option 3: PyCharm Users**  
1. Open project in PyCharm  
2. Go to **File → Settings → Project → Python Interpreter**  
3. Click **Add Interpreter → Add Local Interpreter**  
4. Choose **Poetry Environment** or **Virtualenv Environment**  
5. PyCharm will automatically install dependencies  

## Analysis Pipeline

### Step 1: Data Peek 
Comprehensive overview of 13 MAL datasets including:  
- Structural inspection (shape, data types, null counts)  
- Statistical profiling of numerical features  
- Representative sampling to identify data quality issues

### Step 2: Data Cleaning
Intelligent cleaning pipeline that:  
- Preserves data volume while ensuring relational integrity  
- Removes duplicates and handles missing values surgically  
- Exports cleaned datasets (default is `~/data/cleaned/`)

### Step 3: Data Visualization
Multi-dimensional exploration answering:  
- Geographic reach: Global anime distribution and linguistic diversity  
- User behavior: Engagement patterns from "Plan to Watch" to completion  
- Creator networks: The human engine behind top-performing titles

Tries to answer the question *"What defines success?"* via genre patterns, format preferences, score distributions and much more.

## References

- Course materials:  HCI Data Analysis module (University of Turin 2025/2026)
- External resources: Rob Mulla's [Data Analysis Techniques](https://github.com/robmulla)

## License

Academic project developed for educational purposes at the University of Turin (January 2026).
