# DiffusionAnalysis

DiffusionAnalysis is a comprehensive Python package for analyzing diffusion properties from molecular dynamics (MD) trajectories. It offers a suite of tools to calculate and visualize various quantities that characterize diffusive behavior, including mean squared displacements (MSD), van Hove correlation functions, and autocorrelation functions (ACF).

## Features

- Efficient calculation of mean squared displacements (MSD) for selected atoms or groups of atoms
- Drift correction options for MSD calculations to account for system-wide translations
- Calculation of MSD along specific lattice vectors to study anisotropic diffusion
- Flexible plotting functions for visualizing MSD data with customizable options
- Computation of van Hove correlation functions to analyze dynamical heterogeneity (to be added)
- Calculation of autocorrelation functions (ACF) to study time-dependent correlations (to be added)

## Package Structure
```
DiffusionAnalysis/
│
├── analysis/
│   └── simple_msd_analysis.py      
│
├── examples/
│   ├── dat_directory/
│   ├── example.ipynb
│   └── msd_plot.png
│
├── loaders/    
│   ├── ase_atoms_loader.py
│   ├── base_structure_loader.py
│   ├── dat_directory_structure_loader.py
│   └── xyz_directory_structure_loader.py
│
├── trajectory/
│   └── displacement_trajectory.py
│
├── utils/
│   ├── io.py
│   └── transforms.py
```

### simple_msd_analysis.py
This module contains the SimpleMSDAnalysis class, which is used to perform mean squared displacement (MSD) analysis on a DisplacementTrajectory object. The class provides methods to calculate the MSD, correct for drift, and plot the total, or directional, MSD.

## Installation

You can easily install DiffusionAnalysis :
```bash
git clone https://github.com/chrisdavi3s/DiffusionAnalysis.git
cd DiffusionAnalysis
pip install -r requirements.txt
```
Install the package in development mode:
```bash
pip install -e .
```

## Usage

Here's a simple example demonstrating how to use DiffusionAnalysis to calculate and plot the MSD:

```python
from diffusionanalysis.loaders import DatDirectoryStructureLoader
from diffusionanalysis.trajectory import DisplacementTrajectory 
from diffusionanalysis.analysis import SimpleMSDAnalysis

# Load the MD trajectory
loader = DatDirectoryStructureLoader("path/to/trajectory")
trajectory = DisplacementTrajectory(loader)
trajectory.generate_displacement_trajectory()

# Perform MSD analysis
msd_analysis = SimpleMSDAnalysis(trajectory)
msd_data = msd_analysis.calculate_msd()

# Plot the MSD
fig = msd_analysis.plot_msd(msd_data)
fig.savefig("msd_plot.png")