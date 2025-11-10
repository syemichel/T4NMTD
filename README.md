# README

## Installation

### Step 1: Create Conda Environment

Find the **environment.yaml** file in  **conda** directory and create a conda environment using the following command:


```bash
conda env create -f environment.yaml
conda activate [environment_name]
```
Note: Replace [environment_name] with the environment name specified in the environment.yaml file.

### Step 2: Install MuJoCo

Install MuJoCo platform version 2.1.2
please follow the official documentation for installation and configuration.
###Step 3: Replace Conda Packages

Find the compressed files in the **conda/package** folder, extract them and replace the corresponding packages in the conda environment:

## Usage

The code implements 7 algorithms:
- **T4NMTD**: Our proposed framework
- **QRM, HRM, MOD, LSTS, DIRL, HDQN**: Related works for comparison

### Experiment 1: Training on 8 Tasks

Each algorithm has its own folder containing training scripts for all 8 tasks. The training scripts follow the naming convention `[Algorithm]_task[N].py` where N is the task number (1-8).

For example:
- To train T4NMTD on Task 1: Run `python T4NMTD_task1.py` in the T4NMTD folder
- To train QRM on Task 3: Run `python QRM_task3.py` in the QRM folder

### Experiments 2, 3, and 4

The code for experiments 2, 3, and 4 can be found in the `T4NMTD/exp` folder. Navigate to this directory to run these additional experiments.

