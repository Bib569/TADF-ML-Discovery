#!/usr/bin/env python
"""
DelE Prediction for Molecules

A machine learning tool for predicting electronic energy differences in molecules
using gradient boosting with a physics-informed custom loss function.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

class CustomLossGBR(BaseEstimator, RegressorMixin):
    """
    A wrapper class to use custom loss function with GradientBoostingRegressor.
    
    Parameters:
    -----------
    custom_loss_func : callable
        Custom loss function to be used with the GradientBoostingRegressor.
    *args, **kwargs : 
        Arguments to be passed to the GradientBoostingRegressor.
    """
    def __init__(self, custom_loss_func, *args, **kwargs):
        self.custom_loss_func = custom_loss_func
        self.gbr = GradientBoostingRegressor(*args, **kwargs)

    def fit(self, X, y):
        self.gbr.fit(X, y)
        return self

    def predict(self, X):
        return self.gbr.predict(X)

    def score(self, X, y):
        y_pred = self.predict(X)
        return -self.custom_loss_func(y, y_pred)

def custom_loss(y_true, y_pred, predictors):
    """
    Calculate custom loss for the model based on specific coefficients and predictors.
    
    Parameters:
    -----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    predictors : DataFrame
        DataFrame containing predictor variables.
        
    Returns:
    --------
    float
        Total loss value (MSE + custom loss).
    """
    # Define coefficients
    a0 = 1.918494371114761e+05
    a1 = 7.829196736383097e-05
    a2 = 4.598237163638055e-01
    a3 = -4.059285596435883e+04
    c0 = -2.395860624894289e-01

    # Calculate loss using the equation
    physics loss = (
        c0 +
        a0 * ((predictors['t1_cis'] ** 3) * (predictors['homo_lumo_overlap'] * predictors['dsp_scf'])) +
        a1 * ((predictors['homo_lumo_overlap'] ** 6) / (predictors['exchange_integral'] ** 2)) +
        a2 * np.abs(np.cbrt(predictors['t1_cis']) - (predictors['homo_lumo_overlap'] - predictors['t1_cis'])) +
        a3 * ((predictors['exchange_integral'] ** 3) * np.log(predictors['homo_lumo_overlap']))
    )

    # Calculate the mean squared error
    mse = np.mean((y_true - y_pred) ** 2)

    # Return the total loss as MSE + loss
    return mse + physics loss

def load_and_preprocess_data(data_path, selected_descriptors):
    """
    Load and preprocess the dataset.
    
    Parameters:
    -----------
    data_path : str
        Path to the dataset.
    selected_descriptors : list
        List of descriptor column names to use as predictors.
        
    Returns:
    --------
    tuple
        Tuple containing scaled predictors, input scalers, target variable, target scaler.
    """
    # Load dataset
    dataset = pd.read_csv(data_path)
    
    # Define predictors and targets
    predictors = dataset[selected_descriptors].values
    targets = dataset['delE_CISD'].values
    
    # Scale the target variable
    target_scaler = StandardScaler()
    targets_scl = target_scaler.fit_transform(targets.reshape(-1, 1)).squeeze()
    
    # Scale input features
    inp_scl = []
    scl_inp = predictors.copy()
    for i in range(len(selected_descriptors)):
        s = StandardScaler()
        scl_inp[:, i] = s.fit_transform(predictors[:, i].reshape(-1, 1)).squeeze()
        inp_scl.append(s)
    
    return scl_inp, inp_scl, targets_scl, target_scaler

def train_model(x_train, y_train, n_estimators=980, learning_rate=0.013148580559414888, 
                max_depth=10, min_samples_split=10, min_samples_leaf=2, 
                subsample=0.8067786946581829, max_features=0.9560137659140971, 
                random_state=16043):
    """
    Train a Gradient Boosting model with custom loss function.
    
    Parameters:
    -----------
    x_train : array-like
        Training data features.
    y_train : array-like
        Training data target values.
    n_estimators, learning_rate, max_depth, min_samples_split, min_samples_leaf, 
    subsample, max_features, random_state : 
        Parameters for GradientBoostingRegressor.
        
    Returns:
    --------
    CustomLossGBR
        Trained model.
    """
    model = CustomLossGBR(
        custom_loss,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        max_features=max_features,
        random_state=random_state
    )
    model.fit(x_train, y_train)
    return model

def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance using various metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    corr, _ = pearsonr(y_true, y_pred)
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'Pearson Correlation': corr
    }
    
    return metrics

def plot_predictions(exp_values, pred_values, title, output_path):
    """
    Create a scatter plot of experimental vs. predicted values with confidence intervals.
    
    Parameters:
    -----------
    exp_values : array-like
        Experimental values.
    pred_values : array-like
        Predicted values.
    title : str
        Plot title.
    output_path : str
        Path to save the output figure.
    """
    # Create a DataFrame for plotting
    compare_df = pd.DataFrame({
        'delE (Experimental)': exp_values,
        'delE (GradientBoosting)': pred_values
    })
    
    # Fit linear regression for the trend line
    lr = LinearRegression()
    lr.fit(exp_values.reshape(-1, 1), pred_values.reshape(-1, 1))
    lr_pred = lr.predict(exp_values.reshape(-1, 1)).flatten()
    
    compare_df['delE_LR'] = lr_pred
    
    # Dark bottle green color for the scatterplot points
    dark_bottle_green = '#004953'
    
    # Increase figure size for better visibility
    plt.figure(figsize=(10, 8))
    
    # Scatter plot for Experimental vs. GradientBoosting predictions with dark black borders
    sns.scatterplot(x='delE (Experimental)', y='delE (GradientBoosting)', 
                    data=compare_df, label='GradientBoosting', edgecolor='black', 
                    s=100, color=dark_bottle_green, alpha=0.7)
    
    # Regression line for Linear Regression predictions
    sns.regplot(x='delE (Experimental)', y='delE_LR', 
                data=compare_df, color='red', scatter=False, 
                label='Linear Regression', line_kws={'linewidth': 2})
    
    # Sort the DataFrame by x-values
    sorted_data = compare_df.sort_values(by='delE (Experimental)')
    
    # Calculate standard error of the regression
    se = np.std(pred_values) / np.sqrt(len(pred_values))
    
    # Margin of error for the confidence interval
    margin_of_error = se * 1.96  # Assuming 1.96 for a 95% confidence interval
    
    # Plot shaded region for the confidence interval
    plt.fill_between(sorted_data['delE (Experimental)'], 
                     sorted_data['delE_LR'] - margin_of_error, 
                     sorted_data['delE_LR'] + margin_of_error, 
                     color='red', alpha=0.2, label='CI (95%)')
    
    # Add labels and title
    plt.title(title, fontweight='bold', fontsize=14)
    plt.xlabel('Experimental delE', fontsize=14)
    plt.ylabel('Predicted delE', fontsize=14)
    
    # Add legend with a black border
    legend = plt.legend()
    legend.get_frame().set_edgecolor('black')
    
    # Customize gridlines
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add a black border around the plot
    border_rect = patches.Rectangle((-0.01, -0.01), 1.02, 1.02, linewidth=1.8, 
                                   edgecolor='black', facecolor='none', 
                                   transform=plt.gca().transAxes)
    plt.gca().add_patch(border_rect)
    
    # Customize plot aesthetics
    sns.despine()
    
    # Calculate metrics
    metrics = evaluate_model(exp_values, pred_values)
    
    # Add metrics to the plot
    textstr = (f'Correlation (r): {metrics["Pearson Correlation"]:.4f}\n'
               f'RMSE: {metrics["RMSE"]:.4f}\n'
               f'MAE: {metrics["MAE"]:.4f}')
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    
    # Add the text box in the upper right corner
    plt.gca().text(0.8, 0.15, textstr, transform=plt.gca().transAxes, fontsize=12,
                   verticalalignment='top', bbox=props)
    
    # Save the figure in high resolution with a tight border
    plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
    
    # Close the figure to free memory
    plt.close()
    
    return metrics

def predict_unknown_molecules(model, data_path, inp_scalers, target_scaler, output_path, 
                             descriptor_mapping=None):
    """
    Predict properties for unknown molecules.
    
    Parameters:
    -----------
    model : CustomLossGBR
        Trained model for prediction.
    data_path : str
        Path to the dataset of unknown molecules.
    inp_scalers : list
        List of scalers for input features.
    target_scaler : StandardScaler
        Scaler for the target variable.
    output_path : str
        Path to save predictions.
    descriptor_mapping : dict, optional
        Mapping between dataset column names and model's expected column names.
        
    Returns:
    --------
    DataFrame
        DataFrame with original data and predictions.
    """
    # Default mapping if none provided
    if descriptor_mapping is None:
        descriptor_mapping = {
            's1_cis': 's1_cis',
            't1_cis': 't1_cis',
            'exch_int': 'exchange_integral',
            'dsp_scf': 'dsp_scf',
            'H-L_overlap': 'homo_lumo_overlap'
        }
    
    # Load the dataset of unknown molecules
    if data_path.endswith('.csv'):
        unknown_dataset = pd.read_csv(data_path)
    else:
        unknown_dataset = pd.read_excel(data_path)
    
    # Get the list of input descriptors for the model
    selected_descriptors = list(descriptor_mapping.keys())
    
    # Extract predictors from the unknown dataset
    unknown_predictors = unknown_dataset[selected_descriptors].values
    
    # Scale the unknown predictors
    scl_unknown_predictors = unknown_predictors.copy()
    for i, s in enumerate(inp_scalers):
        scl_unknown_predictors[:, i] = s.transform(unknown_predictors[:, i].reshape(-1, 1)).squeeze()
    
    # Use the trained model to make predictions
    unknown_predictions = model.predict(scl_unknown_predictors)
    
    # Inverse transform the predictions to the original scale
    unknown_predictions = target_scaler.inverse_transform(unknown_predictions.reshape(-1, 1)).flatten()
    
    # Add the predictions to the unknown dataset
    unknown_dataset['Predicted_delE'] = unknown_predictions
    
    # Save the dataset with predictions
    unknown_dataset.to_excel(output_path, index=False)
    
    return unknown_dataset

def plot_distribution(data, col_name='Predicted_delE', output_path='delE_histogram.png'):
    """
    Plot distribution histogram of a specific column in the dataset.
    
    Parameters:
    -----------
    data : DataFrame or array-like
        Data containing the column to plot.
    col_name : str
        Column name to plot.
    output_path : str
        Path to save the output figure.
    """
    if isinstance(data, pd.DataFrame):
        target = data[col_name].values
    else:
        target = data
    
    plt.figure(figsize=(10, 8))
    
    # Use Seaborn for better styling
    sns.distplot(target, bins=50, color='#004953', 
                hist_kws={'edgecolor': 'black', 'alpha': 0.8}, 
                kde=True, kde_kws={'color': 'red'})
    
    # Customize labels and title
    plt.xlabel('delE')
    plt.ylabel('Probability')
    
    # Add gridlines for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Remove the right and upper spines
    sns.despine()
    
    # Save the figure in high resolution
    plt.savefig(output_path, dpi=300)
    
    # Close the figure to free memory
    plt.close()

def main():
    """
    Main function to run the complete workflow.
    """
    # Define data paths
    training_data_path = "data/combined_delE_dataset.csv"
    new_molecules_path = "data/Newly_generated_moleucles.xlsx"
    
    # Define descriptors
    selected_descriptors = ['s1_cis', 't1_cis', 'exchange_integral', 'dsp_scf', 'homo_lumo_overlap']
    
    # Descriptor mapping for unknown datasets
    descriptor_mapping = {
        's1_cis': 's1_cis',
        't1_cis': 't1_cis',
        'exch_int': 'exchange_integral',
        'dsp_scf': 'dsp_scf',
        'H-L_overlap': 'homo_lumo_overlap'
    }
    
    # Load and preprocess data
    scl_inp, inp_scl, targets_scl, target_scaler = load_and_preprocess_data(
        training_data_path, selected_descriptors
    )
    
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        scl_inp, targets_scl, test_size=0.2, random_state=16043
    )
    
    # Train the model
    trained_model = train_model(x_train, y_train)
    
    # Make predictions on training set
    train_preds = trained_model.predict(x_train)
    
    # Inverse transform for evaluation
    y_train_orig = target_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    train_preds_orig = target_scaler.inverse_transform(train_preds.reshape(-1, 1)).flatten()
    
    # Plot and evaluate training results
    train_metrics = plot_predictions(
        y_train_orig, 
        train_preds_orig, 
        'Experimental vs. Predicted delE for Training Data',
        'results/exp_vs_pred_train_delE.png'
    )
    
    # Make predictions on test set
    test_preds = trained_model.predict(x_test)
    
    # Inverse transform for evaluation
    y_test_orig = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    test_preds_orig = target_scaler.inverse_transform(test_preds.reshape(-1, 1)).flatten()
    
    # Plot and evaluate test results
    test_metrics = plot_predictions(
        y_test_orig, 
        test_preds_orig, 
        'Experimental vs. Predicted delE for Testing Data',
        'results/exp_vs_pred_test_delE.png'
    )
    
    # Print evaluation results
    print("Training Set Metrics:")
    for key, value in train_metrics.items():
        print(f"{key}: {value:.4f}")
    
    print("\nTest Set Metrics:")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Predict for newly generated molecules
    new_results = predict_unknown_molecules(
        trained_model,
        new_molecules_path,
        inp_scl,
        target_scaler,
        'results/Predicted_delE_newly_gen_mols.xlsx',
        descriptor_mapping
    )
    
    # Plot distribution of new molecules predictions
    plot_distribution(
        new_results, 
        'Predicted_delE',
        'results/new_molecules_delE_distribution.png'
    )

if __name__ == "__main__":
    main() 
