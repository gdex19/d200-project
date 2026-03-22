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

## Installing the package
Before downloading the data and running the analyses, you must first install the project's package:

```bash
pip install -e .
```

# Getting the data
Download data from https://drive.google.com/file/d/1TOq3kuFB2RkvrQRKvcrK2h1id9KVBjIJ/view?usp=drive_link and place in 
data directory (not in the package). 

# Reproducing the analysis
After downloading the data as above, run ```mdn.ipynb``` to reproduce the analysis. Note that the end of the notebook is
in progress, and that the grid search does not contain all combinations of parameters that were tested. Feel free to play around!

TODO: FINISH
