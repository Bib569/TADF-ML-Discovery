# TADF-ML-Discovery
A physics-guided machine learning framework for accelerated discovery of Thermally Activated Delayed Fluorescence (TADF) emitters through accurate prediction of singlet-triplet energy gaps using quantum chemical descriptors and custom loss functions.

## Overview

This project implements a physics-guided machine learning model to predict the singlet-triplet energy gap (delE) in organic molecules based on quantum chemical descriptors. The framework uses a gradient boosting regressor with a physics-informed custom loss function to improve prediction accuracy.

## Paper

This repository contains code and documentation for the paper:
**Accelerated Discovery of TADF Emitters via Physics-Guided Machine Learning of Excited-State Properties**

### Authors
Sanyam¹, Bibhas Das¹, and Anirban Mondal¹,*

¹Department of Chemistry, Indian Institute of Technology Gandhinagar, Gujarat, 382355, India  
²Department of Materials Engineering, Indian Institute of Technology Gandhinagar, Gujarat, 382355, India

*Email: amondal@iitgn.ac.in

## Table of Contents

- [Introduction](#introduction)
- [Workflow](#workflow)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
- [Model Details](#model-details)
- [Key Descriptors](#key-descriptors)
- [Results](#results)
- [Customization](#customization)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

## Introduction

This repository provides the implementation of an efficient machine learning approach to predicting singlet-triplet energy gaps (delE) in organic molecules. The framework combines gradient boosting with a physics-informed custom loss function to accurately predict excited state properties critical for the discovery of new TADF emitters. By incorporating quantum chemical principles into the machine learning model, we achieve improved predictive performance while maintaining physical consistency in the predictions.

## Workflow

The code is designed to:
1. Process molecular data with quantum chemical descriptors from computational chemistry calculations
2. Implement a physics-informed machine learning model using gradient boosting regression
3. Optimize model hyperparameters for accurate prediction of delE values
4. Validate predictions against experimental and computational reference data
5. Provide a framework for screening new molecules for TADF properties
6. Visualize results and generate confidence intervals for predictions

## System Requirements

### Operating Systems
- Linux, macOS, or Windows

### Python Requirements
- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy

## Installation

```bash
# Clone the repository
git clone https://github.com/Bib569/TADF-ML-Discovery.git
cd TADF-ML-Discovery

# Set up a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Directory Structure

```
delE-prediction/
├── data/                     # Data directory
│   ├── merged_dataset.csv    # Training dataset
│   ├── del_E_SISSO - Sheet1.csv    # Unknown molecules dataset
│   ├── delE_PIML - experimental.csv    # Experimental dataset
│   └── Newly_generated_moleucles_delE.xlsx    # Generated molecules
├── results/                  # Results directory for outputs
├── README.md                 # Project documentation
├── requirements.txt          # Required packages
├── PIML_combined_delE.py     # Main codes   
└── PIML_f.py                  
```

## Usage

1. Prepare your data files in the `data/` directory
2. Run the main scripts:

```bash
python PIML_combined_delE.py
python PIML_combined_f.py
```

3. Find results in the `results/` directory

## Model Details

The model implements a Gradient Boosting Regressor with a custom loss function incorporating domain knowledge of quantum chemical interactions. The custom loss applies penalties based on physical relationships between descriptors:

```python
def custom_loss(y_true, y_pred, predictors):
    # Define coefficients
    a0 = 1.918494371114761e+05
    a1 = 7.829196736383097e-05
    a2 = 4.598237163638055e-01
    a3 = -4.059285596435883e+04
    c0 = -2.395860624894289e-01

    # Calculate loss using the equation
    loss = (
        c0 +
        a0 * ((predictors['t1_cis'] ** 3) * (predictors['homo_lumo_overlap'] * predictors['dsp_scf'])) +
        a1 * ((predictors['homo_lumo_overlap'] ** 6) / (predictors['exchange_integral'] ** 2)) +
        a2 * np.abs(np.cbrt(predictors['t1_cis']) - (predictors['homo_lumo_overlap'] - predictors['t1_cis'])) +
        a3 * ((predictors['exchange_integral'] ** 3) * np.log(predictors['homo_lumo_overlap']))
    )

    # Calculate the mean squared error
    mse = np.mean((y_true - y_pred) ** 2)

    # Return the total loss as MSE + loss
    return mse + loss
```

## Key Descriptors

The model uses the following quantum chemical descriptors:
- `s1_cis`: Singlet excited state energy
- `t1_cis`: Triplet excited state energy
- `exchange_integral`: Exchange integral
- `dsp_scf`: Dynamic spin polarization
- `homo_lumo_overlap`: Overlap between HOMO and LUMO orbitals

## Results

The model evaluates performance using:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R² score
- Pearson correlation coefficient

Visualizations include:
- Scatter plots of experimental vs. predicted values
- Confidence intervals of predictions
- Distribution histograms of predicted properties

## Customization

To use the model for your own dataset:
1. Prepare your data with the required descriptors
2. Update the file paths in the `main()` function
3. Modify descriptor mappings if your column names differ
4. Run the script

## License

[MIT License](LICENSE)

## Citation

If you use this code in your research, please cite:

```
@article{TADF_ML_2023,
  author = {Sanyam and Das, Bibhas and Mondal, Anirban},
  title = {Accelerated Discovery of TADF Emitters via Physics-Guided Machine Learning of Excited-State Properties},
  journal = {},
  year = {2025},
}
```

## Contact

For questions or feedback, please [open an issue](https://github.com/yourusername/delE-prediction/issues) or contact [amondal@iitgn.ac.in]. 
