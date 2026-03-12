# D200 Final Project

# Setup

## Environment

Install the required packages by running:

```bash
conda env create -f environment.yml
```

Activate the environment by running:

```bash
conda activate final-project-d200
```

## Pre-commit

This repository uses pre-commit to enforce consistent code. To install
pre-commit, run:

```bash
pre-commit install
```

To run the checks manually, run:

```bash
pre-commit run --all-files
```
## Installing the package
Before downloading the data and running the analyses, you must first install the project's package:

```bash
pip install -e .
```

# Getting the data
TODO: FINISH

# Reproducing the analysis
After pulling the data as above, run ```eda.ipynb``` to see the exploratory data analysis and save the cleaned
data. 

TODO: FINISH
