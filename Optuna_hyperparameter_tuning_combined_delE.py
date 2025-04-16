#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Custom Gradient Boosting Regressor with Physics-Informed Machine Learning

This module implements a custom loss function for predicting quantum chemical 
properties using gradient boosting and Optuna hyperparameter optimization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from scipy.stats import pearsonr
import optuna
from optuna.samplers import TPESampler
import joblib
import os
import logging
from datetime import datetime
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"optuna_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def custom_loss(y_true, y_pred, predictors):
    """
    Physics-informed custom loss function.
    
    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    predictors : DataFrame
        DataFrame containing the predictor variables
        
    Returns
    -------
    float
        Combined loss value (MSE + physics-based loss)
    """
    # Define coefficients
    a0 = 1.918494371114761e+05
    a1 = 7.829196736383097e-05
    a2 = 4.598237163638055e-01
    a3 = -4.059285596435883e+04
    c0 = -2.395860624894289e-01

    # Calculate physics-based loss using the updated equation
    physics_loss = (
        c0 +
        a0 * ((predictors['t1_cis'] ** 3) * (predictors['homo_lumo_overlap'] * predictors['dsp_scf'])) +
        a1 * ((predictors['homo_lumo_overlap'] ** 6) / (predictors['exchange_integral'] ** 2)) +
        a2 * np.abs(np.cbrt(predictors['t1_cis']) - (predictors['homo_lumo_overlap'] - predictors['t1_cis'])) +
        a3 * ((predictors['exchange_integral'] ** 3) * np.log(predictors['homo_lumo_overlap']))
    )

    # Calculate mean squared error
    mse = np.mean((y_true - y_pred) ** 2)

    # Return the total loss as MSE + physics-based loss
    return mse + physics_loss


class CustomLossGBR(BaseEstimator, RegressorMixin):
    """
    Custom Gradient Boosting Regressor with a physics-informed loss function.
    
    This class wraps sklearn's GradientBoostingRegressor with a custom loss
    function that incorporates domain knowledge from quantum chemistry.
    
    Parameters
    ----------
    custom_loss_func : callable
        The custom loss function to use
    *args, **kwargs : 
        Arguments to pass to GradientBoostingRegressor
    """
    
    def __init__(self, custom_loss_func, *args, **kwargs):
        self.custom_loss_func = custom_loss_func
        self.gbr = GradientBoostingRegressor(*args, **kwargs)

    def fit(self, X, y):
        """
        Fit the model to training data.
        
        Parameters
        ----------
        X : array-like
            Training features
        y : array-like
            Training targets
            
        Returns
        -------
        self : object
            Returns self
        """
        self.gbr.fit(X, y)
        return self

    def predict(self, X):
        """
        Make predictions using the fitted model.
        
        Parameters
        ----------
        X : array-like
            Features to predict on
            
        Returns
        -------
        array-like
            Predictions
        """
        return self.gbr.predict(X)

    def score(self, X, y):
        """
        Calculate score using the custom loss function.
        
        Parameters
        ----------
        X : array-like
            Features to predict on
        y : array-like
            True target values
            
        Returns
        -------
        float
            Negative custom loss (higher is better)
        """
        y_pred = self.predict(X)
        return -self.custom_loss_func(y, y_pred)


def load_and_preprocess_data(data_path, selected_descriptors, target_column, test_size=0.2, random_state=5203):
    """
    Load and preprocess the dataset.
    
    Parameters
    ----------
    data_path : str
        Path to the dataset CSV file
    selected_descriptors : list
        List of descriptor column names to use as features
    target_column : str
        Name of the target column
    test_size : float, optional
        Proportion of data to use for testing
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    tuple
        (x_train, x_test, y_train, y_test, dataset, scalers)
    """
    logger.info(f"Loading data from {data_path}")
    dataset = pd.read_csv(data_path)
    logger.info(f"Dataset shape: {dataset.shape}")
    
    # Extract features and target
    predictors = dataset[selected_descriptors].values
    targets = dataset[target_column].values
    
    # Scale target
    target_scaler = StandardScaler()
    targets_scl = target_scaler.fit_transform(targets.reshape(-1, 1)).squeeze()
    
    # Scale features
    scl_inp = predictors.copy()
    feature_scalers = []
    
    for i in range(len(selected_descriptors)):
        scaler = StandardScaler()
        scl_inp[:, i] = scaler.fit_transform(predictors[:, i].reshape(-1, 1)).squeeze()
        feature_scalers.append(scaler)
    
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        scl_inp, targets_scl, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Training set size: {x_train.shape[0]}, Test set size: {x_test.shape[0]}")
    
    scalers = {
        'target': target_scaler,
        'features': feature_scalers
    }
    
    return x_train, x_test, y_train, y_test, dataset, scalers


def objective(trial, x_train, y_train, x_test, y_test, dataset, selected_descriptors):
    """
    Optuna objective function for hyperparameter optimization.
    
    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object
    x_train, y_train, x_test, y_test : array-like
        Training and testing data
    dataset : DataFrame
        Original dataset for custom loss calculation
    selected_descriptors : list
        List of feature names
        
    Returns
    -------
    float
        Pearson correlation coefficient (higher is better)
    """
    # Define hyperparameter search space
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'max_features': trial.suggest_float('max_features', 0.1, 1.0),
        'random_state': trial.suggest_int('random_state', 0, 20000)
    }
    
    # Create and train model with suggested hyperparameters
    model = CustomLossGBR(
        lambda y_true, y_pred: custom_loss(y_true, y_pred, dataset[selected_descriptors]), 
        **param
    )
    model.fit(x_train, y_train)
    
    # Make predictions
    y_pred = model.predict(x_test)
    
    # Calculate Pearson correlation
    pearson_r = pearsonr(y_test, y_pred)[0]
    
    # Log intermediate results
    logger.debug(f"Trial {trial.number}: Pearson r = {pearson_r:.4f}, params = {param}")
    
    return pearson_r


def evaluate_model(model, x_test, y_test, target_scaler):
    """
    Evaluate model performance using multiple metrics.
    
    Parameters
    ----------
    model : object
        Trained model with predict method
    x_test, y_test : array-like
        Test data
    target_scaler : object
        Scaler used to transform the target variable
        
    Returns
    -------
    dict
        Dictionary of evaluation metrics
    """
    # Get predictions
    y_pred = model.predict(x_test)
    
    # Calculate metrics
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'pearson_r': np.corrcoef(y_test, y_pred)[0, 1]
    }
    
    # Calculate metrics on original scale
    y_test_orig = target_scaler.inverse_transform(y_test.reshape(-1, 1)).squeeze()
    y_pred_orig = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).squeeze()
    
    metrics.update({
        'rmse_orig': np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)),
        'mae_orig': mean_absolute_error(y_test_orig, y_pred_orig)
    })
    
    return metrics


def plot_results(y_test, y_pred, metrics, output_dir):
    """
    Generate and save performance plots.
    
    Parameters
    ----------
    y_test, y_pred : array-like
        Test targets and predictions
    metrics : dict
        Dictionary of evaluation metrics
    output_dir : str
        Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Scatter plot of predictions vs. actual values
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    
    # Add perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Add text with metrics
    text = f"RMSE = {metrics['rmse']:.3f}\n" \
           f"MAE = {metrics['mae']:.3f}\n" \
           f"Pearson r = {metrics['pearson_r']:.3f}"
    plt.annotate(text, xy=(0.05, 0.9), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.title('Predicted vs. Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_scatter.png'), dpi=300)
    plt.close()
    
    # Residual plot
    residuals = y_pred - y_test
    plt.figure(figsize=(10, 8))
    sns.histplot(residuals, kde=True)
    plt.title('Residuals Distribution')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residuals_distribution.png'), dpi=300)
    plt.close()


def main(args):
    """
    Main function to run the optimization process.
    
    Parameters
    ----------
    args : Namespace
        Command line arguments
    """
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    # Log the start of the process
    logger.info("Starting optimization process")
    logger.info(f"Configuration: {vars(args)}")
    
    # Define feature columns and target
    selected_descriptors = ['s1_cis', 't1_cis', 'exchange_integral', 'dsp_scf', 'homo_lumo_overlap']
    target_column = 'delE_CISD'
    
    # Load and preprocess data
    x_train, x_test, y_train, y_test, dataset, scalers = load_and_preprocess_data(
        args.data_path, selected_descriptors, target_column, 
        test_size=args.test_size, random_state=args.random_seed
    )
    
    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(n_startup_trials=args.startup_trials),
        study_name=f'quantum_gbr_optimization_{timestamp}'
    )
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create callback for study checkpoints
    def save_checkpoint(study, trial):
        joblib.dump(study, os.path.join(checkpoint_dir, f'study_checkpoint_{trial.number}.pkl'))
        # Save best parameters so far
        if study.best_trial.number == trial.number:
            with open(os.path.join(output_dir, 'best_params.txt'), 'w') as f:
                f.write(f"Best Trial: {trial.number}\n")
                f.write(f"Best Value: {trial.value}\n")
                f.write("Best Parameters:\n")
                for param_name, param_value in trial.params.items():
                    f.write(f"  {param_name}: {param_value}\n")
    
    # Run optimization
    logger.info(f"Starting Optuna optimization with {args.n_trials} trials using {args.n_jobs} workers")
    study.optimize(
        lambda trial: objective(trial, x_train, y_train, x_test, y_test, dataset, selected_descriptors),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        gc_after_trial=True,
        callbacks=[save_checkpoint]
    )
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    logger.info(f"Optimization completed!")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best Pearson r: {best_value:.4f}")
    
    # Train final model with best parameters
    logger.info("Training final model with best parameters")
    final_model = CustomLossGBR(
        lambda y_true, y_pred: custom_loss(y_true, y_pred, dataset[selected_descriptors]), 
        **best_params
    )
    final_model.fit(x_train, y_train)
    
    # Save final model
    model_path = os.path.join(output_dir, 'final_model.pkl')
    joblib.dump(final_model, model_path)
    logger.info(f"Final model saved to {model_path}")
    
    # Save scalers
    scalers_path = os.path.join(output_dir, 'scalers.pkl')
    joblib.dump(scalers, scalers_path)
    logger.info(f"Scalers saved to {scalers_path}")
    
    # Evaluate model
    logger.info("Evaluating final model")
    metrics = evaluate_model(final_model, x_test, y_test, scalers['target'])
    
    # Log and save metrics
    logger.info("Final Model Evaluation Metrics:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")
    
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name}: {metric_value:.4f}\n")
    
    # Generate plots
    if args.generate_plots:
        logger.info("Generating performance plots")
        plot_results(y_test, final_model.predict(x_test), metrics, output_dir)
    
    # Save Optuna visualization
    if args.visualize_optuna:
        logger.info("Generating Optuna visualizations")
        try:
            from optuna.visualization import plot_optimization_history, plot_param_importances
            
            # Plot optimization history
            fig = plot_optimization_history(study)
            fig.write_image(os.path.join(output_dir, 'optimization_history.png'))
            
            # Plot parameter importances
            fig = plot_param_importances(study)
            fig.write_image(os.path.join(output_dir, 'parameter_importances.png'))
        except ImportError:
            logger.warning("Could not generate Optuna visualizations. Please install required dependencies with:")
            logger.warning("pip install optuna plotly kaleido")
    
    logger.info(f"All results saved to {output_dir}")


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train a custom Gradient Boosting Regressor with physics-informed loss function'
    )
    parser.add_argument('--data_path', type=str, default=r"/home/yati/bibhas/PIML/new/merged_dataset.csv",
                       help='Path to the dataset CSV file')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--n_trials', type=int, default=100,
                       help='Number of Optuna trials')
    parser.add_argument('--n_jobs', type=int, default=48,
                       help='Number of parallel jobs')
    parser.add_argument('--startup_trials', type=int, default=10,
                       help='Number of random trials before TPE starts')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Proportion of data to use for testing')
    parser.add_argument('--random_seed', type=int, default=5203,
                       help='Random seed for reproducibility')
    parser.add_argument('--generate_plots', action='store_true',
                       help='Generate performance plots')
    parser.add_argument('--visualize_optuna', action='store_true',
                       help='Generate Optuna visualization plots')
    
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        logger.exception(f"Error occurred: {e}")
        raise