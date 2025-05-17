# Quasi-Maximum Likelihood Dynamic Factor Model (DFM-QML)

**Author:** Moka Kaleji • Contact: mohammadkaleji1998@gmail.com

**Affiliation:** Master Thesis in Econometrics: 

Advancing High-Dimensional Factor Models: Integrating Time-Varying 
Loadings and Transition Matrix with Dynamic Factors.

University of Bologna

**Based on:** Barigozzi & Luciani (2021)

This repository provides scripts to estimate a large-scale dynamic factor model via EM (QML) and to evaluate forecast accuracy.

### Files

- **qml_dfm_estimation.m**  
  - Standardizes data  
  - Calls `BL_Estimate` (EM + PCA initial)  
  - Outputs `EM` structure with loadings, state matrices, smoothed factors  

- **dfm_forecast_accuracy.m**  
  - Takes the `EM` output  
  - Prompts for forecast horizon `h`  
  - Generates out-of-sample forecast via Kalman companion form  
  - Computes MSFE, RMSE, 95% confidence intervals  
  - Visualizes forecast vs. actual and factor correlations  

### Usage

1. Ensure you have the BL_Estimate function by Barigozzi & Luciani in your MATLAB path.
2. Place data in:
3. Run estimation:
```matlab
qml_dfm_estimation
dfm_forecast_accuracy
```


## See Also
- [Preprocessing-FRED-Quarterly-Dataset](https://github.com/moka-kaleji/Preprocessing-FRED-Quarterly-Dataset)
- [Locally-Stationary-Factor-Model](https://github.com/moka-kaleji/Locally-Stationary-Factor-Model)
- [Bayesian-Optimized-Locally-Stationary-Factor-Model](https://github.com/moka-kaleji/Bayesian-Optimized-Locally-Stationary-Factor-Model)
