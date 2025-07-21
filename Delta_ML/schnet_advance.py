import os
import re
import numpy as np
import torch
from ase import Atoms
import pandas as pd
from schnetpack.data import ASEAtomsData
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json
import torchmetrics
import schnetpack as spk
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, max_error
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
import warnings
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
import sys
import logging
import time
from datetime import datetime
import traceback
import psutil
warnings.filterwarnings('ignore')

# ==================== COMPREHENSIVE LOGGING SETUP ====================
def setup_comprehensive_logging(log_level=logging.INFO):
    """Set up comprehensive logging system with file and console output"""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create timestamp for unique log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'logs/schnet_run_{timestamp}.log'
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)8s | %(funcName)20s:%(lineno)4d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler for detailed logging
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler for user feedback
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger, log_filename

# Initialize comprehensive logging
logger, log_filename = setup_comprehensive_logging()

def log_system_info():
    """Log comprehensive system information"""
    logger.info("="*80)
    logger.info("SYSTEM INFORMATION")
    logger.info("="*80)
    
    try:
        logger.info(f"Python version: {sys.version}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"NumPy version: {np.__version__}")
        logger.info(f"Pandas version: {pd.__version__}")
        
        # CPU information
        logger.info(f"CPU count (logical): {mp.cpu_count()}")
        if hasattr(os, 'cpu_count'):
            logger.info(f"CPU count (physical): {os.cpu_count()}")
        
        # Memory information
        if 'psutil' in sys.modules:
            memory = psutil.virtual_memory()
            logger.info(f"Total memory: {memory.total / (1024**3):.2f} GB")
            logger.info(f"Available memory: {memory.available / (1024**3):.2f} GB")
            logger.info(f"Memory usage: {memory.percent:.1f}%")
        
        # GPU information
        if torch.cuda.is_available():
            logger.info(f"CUDA available: True")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            logger.info("CUDA available: False")
        
        # Environment information
        logger.info(f"Working directory: {os.getcwd()}")
        logger.info(f"Platform: {sys.platform}")
        
    except Exception as e:
        logger.error(f"Error logging system info: {e}")
    
    logger.info("="*80)

def log_function_entry(func_name, **kwargs):
    """Log function entry with parameters"""
    logger.debug(f">>> ENTERING {func_name}")
    for key, value in kwargs.items():
        if isinstance(value, (list, np.ndarray)) and len(value) > 5:
            logger.debug(f"    {key}: {type(value).__name__} with {len(value)} elements")
        else:
            logger.debug(f"    {key}: {value}")

def log_function_exit(func_name, result=None, duration=None):
    """Log function exit with result and duration"""
    if duration:
        logger.debug(f"<<< EXITING {func_name} (duration: {duration:.2f}s)")
    else:
        logger.debug(f"<<< EXITING {func_name}")
    if result is not None:
        if isinstance(result, (list, np.ndarray)) and len(result) > 5:
            logger.debug(f"    Result: {type(result).__name__} with {len(result)} elements")
        else:
            logger.debug(f"    Result: {result}")

def log_step(step_name, message=""):
    """Log a major processing step"""
    logger.info("="*60)
    logger.info(f"STEP: {step_name}")
    if message:
        logger.info(f"Description: {message}")
    logger.info("="*60)

def log_progress(current, total, item_name="items"):
    """Log progress for loops"""
    if total > 0:
        percent = (current / total) * 100
        logger.info(f"Progress: {current}/{total} {item_name} ({percent:.1f}%)")

def log_error_with_traceback(error_msg, exception=None):
    """Log error with full traceback"""
    logger.error("="*60)
    logger.error(f"ERROR: {error_msg}")
    if exception:
        logger.error(f"Exception type: {type(exception).__name__}")
        logger.error(f"Exception message: {str(exception)}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
    logger.error("="*60)

# Extended atomic number mapping for all elements in the periodic table
atomic_number_map = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
    'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
    'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
    'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
    'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110,
    'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
}

def load_optuna_parameters(params_file='best_delE_CISD_params.json'):
    """Load Optuna-optimized hyperparameters from JSON file"""
    start_time = time.time()
    log_function_entry("load_optuna_parameters", params_file=params_file)
    
    try:
        logger.info(f"Attempting to load Optuna parameters from: {params_file}")
        logger.debug(f"Checking if file exists: {os.path.exists(params_file)}")
        
        with open(params_file, 'r') as f:
            params = json.load(f)
        
        logger.info("Successfully loaded Optuna parameters")
        logger.info("="*60)
        logger.info("OPTUNA-OPTIMIZED HYPERPARAMETERS LOADED")
        logger.info("="*60)
        logger.info(f"Parameters loaded from: {params_file}")
        
        if 'optimization_date' in params:
            logger.info(f"Optimization date: {params['optimization_date']}")
        if 'best_rmse' in params:
            logger.info(f"Best RMSE achieved: {params['best_rmse']:.4f} eV")
        if 'total_trials' in params:
            logger.info(f"Total trials: {params['total_trials']}")
        
        logger.info("Loaded hyperparameters:")
        logger.info("-" * 30)
        for key, value in params.items():
            if key not in ['optimization_date', 'best_rmse', 'total_trials']:
                logger.info(f"{key:20}: {value}")
        logger.info("="*60)
        
        log_function_exit("load_optuna_parameters", result=f"Successfully loaded {len(params)} parameters", 
                         duration=time.time() - start_time)
        return params
        
    except FileNotFoundError as e:
        logger.warning(f"Optuna parameters file not found: {params_file}")
        logger.warning("Using default hyperparameters. Run Optuna optimization first:")
        logger.warning("python t1optuna.py")
        log_error_with_traceback("Optuna parameters file not found", e)
        return None
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in parameters file: {params_file}")
        log_error_with_traceback("JSON decode error in parameters file", e)
        return None
        
    except Exception as e:
        logger.error(f"Unexpected error loading parameters: {e}")
        log_error_with_traceback("Unexpected error loading Optuna parameters", e)
        return None

def get_hyperparameters(use_optuna=False):
    """Get hyperparameters either from Optuna optimization or defaults"""
    start_time = time.time()
    log_function_entry("get_hyperparameters", use_optuna=use_optuna)
    
    # Default hyperparameters (fallback)
    default_params = {
        'bs': 600,
        'lr': 5e-4,
        'patience': 15,
        'n_rbf': 18,
        'n_interactions': 3,
        'n_atom_basis': 64,
        'cutoff': 9,
        'pat2': 11
    }
    
    logger.info(f"Getting hyperparameters (use_optuna={use_optuna})")
    logger.debug(f"Default parameters: {default_params}")
    
    if use_optuna:
        logger.info("Attempting to load Optuna-optimized hyperparameters...")
        optuna_params = load_optuna_parameters()
        
        if optuna_params:
            logger.info("Optuna parameters found, processing...")
            # Extract only the hyperparameter values
            params = {}
            missing_keys = []
            
            for key in default_params.keys():
                if key in optuna_params:
                    params[key] = optuna_params[key]
                    logger.debug(f"Using Optuna value for {key}: {optuna_params[key]}")
                else:
                    params[key] = default_params[key]
                    missing_keys.append(key)
                    logger.warning(f"Key '{key}' not found in Optuna params, using default: {default_params[key]}")
            
            if missing_keys:
                logger.warning(f"Missing Optuna parameters: {missing_keys}")
            
            logger.info("Final hyperparameters:")
            for key, value in params.items():
                logger.info(f"  {key:15}: {value}")
            
            log_function_exit("get_hyperparameters", result="Optuna parameters loaded", 
                             duration=time.time() - start_time)
            return params
        else:
            logger.warning("Optuna parameters not available, falling back to defaults")
            logger.info("Final hyperparameters (defaults):")
            for key, value in default_params.items():
                logger.info(f"  {key:15}: {value}")
            
            log_function_exit("get_hyperparameters", result="Default parameters used", 
                             duration=time.time() - start_time)
            return default_params
    else:
        logger.info("Using default hyperparameters (Optuna not requested)")
        logger.info("To use Optuna-optimized parameters, run with: --use_optuna_params")
        logger.info("Final hyperparameters (defaults):")
        for key, value in default_params.items():
            logger.info(f"  {key:15}: {value}")
        
        log_function_exit("get_hyperparameters", result="Default parameters used", 
                         duration=time.time() - start_time)
        return default_params

# Simple combined model for molecular property prediction (adapted from t1optuna.py)
class SimpleMolecularModel(torch.nn.Module):
    """Simple model that combines SchNet with external descriptors for molecular property prediction"""
    
    def __init__(self, n_atom_basis=64, n_interactions=3, cutoff=5.0, n_rbf=20, n_descriptors=5):
        super(SimpleMolecularModel, self).__init__()
        
        # SchNet for processing atomic structure
        self.schnet = spk.representation.SchNet(
            n_atom_basis=n_atom_basis,
            n_interactions=n_interactions,
            radial_basis=spk.nn.GaussianRBF(n_rbf=n_rbf, cutoff=cutoff),
            cutoff_fn=spk.nn.CosineCutoff(cutoff),
            activation=F.silu
        )
        
        # MLP for processing external descriptors
        self.descriptor_mlp = torch.nn.Sequential(
            torch.nn.Linear(n_descriptors, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU()
        )
        
        # Final prediction layers
        combined_dim = n_atom_basis + 16  # SchNet output + descriptor output
        self.output_mlp = torch.nn.Sequential(
            torch.nn.Linear(combined_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
        
        self.n_atom_basis = n_atom_basis
        
    def forward(self, batch):
        # Process atomic structure with SchNet
        schnet_out = self.schnet(batch)
        atom_features = schnet_out['scalar_representation']  # [n_atoms, n_atom_basis]
        
        # Get molecular indices
        idx_m = batch.get('_idx_m', batch.get('idx_m'))
        if idx_m is None:
            idx_m = torch.zeros(atom_features.size(0), dtype=torch.long, device=atom_features.device)
        
        # Aggregate atom features to molecular features
        n_molecules = int(idx_m.max().item()) + 1
        mol_features = torch.zeros(n_molecules, atom_features.size(1), device=atom_features.device)
        
        for i in range(n_molecules):
            mask = (idx_m == i)
            if mask.sum() > 0:
                mol_features[i] = atom_features[mask].mean(dim=0)
        
        # Process external descriptors
        descriptors = torch.stack([
            batch['s1_cis'].squeeze(),
            batch['t1_cis'].squeeze(),
            batch['exchange_integral'].squeeze(),
            batch['dsp_scf'].squeeze(), 
            batch['homo_lumo_overlap'].squeeze()
        ], dim=1)
        
        if descriptors.dim() == 1:
            descriptors = descriptors.unsqueeze(0)
            
        # Ensure we have the right number of descriptors
        if descriptors.size(0) != n_molecules:
            if descriptors.size(0) == 1:
                descriptors = descriptors.expand(n_molecules, -1)
            else:
                descriptors = descriptors[:n_molecules]
        
        descriptor_features = self.descriptor_mlp(descriptors)
        
        # Combine features
        combined = torch.cat([mol_features, descriptor_features], dim=1)
        
        # Make predictions
        predictions = self.output_mlp(combined)
        
        # Flatten predictions to match target shape [batch_size] instead of [batch_size, 1]
        predictions = predictions.squeeze(-1)
        
        return {'delE_CISD': predictions}


# Custom output module that incorporates external descriptors
class DescriptorEnhancedOutput(torch.nn.Module):
    def __init__(self, n_in, n_descriptors=5, output_key='delE_CISD'):
        super(DescriptorEnhancedOutput, self).__init__()
        self.output_key = output_key
        self.n_in = n_in
        self.n_descriptors = n_descriptors
        
        # Layers to process external descriptors
        self.descriptor_layer = torch.nn.Sequential(
            torch.nn.Linear(n_descriptors, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU()
        )
        
        # Final prediction layer combining SchNet output and descriptors
        self.prediction_layer = torch.nn.Sequential(
            torch.nn.Linear(n_in + 16, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
        
    def forward(self, inputs):
        # inputs should contain both 'scalar_representation' and descriptor fields
        
        # Debug: check what we're receiving
        scalar_repr = inputs['scalar_representation']
        print(f"DEBUG: scalar_representation shape: {scalar_repr.shape}")
        
        # The scalar representation is [batch_size, n_atoms, n_features]
        # We need to aggregate over atoms to get molecular representation
        if scalar_repr.dim() == 3:
            # Sum over atoms (dim=1) to get [batch_size, n_features]
            molecular_repr = torch.sum(scalar_repr, dim=1)
        else:
            molecular_repr = scalar_repr
        
        # Ensure molecular_repr is 2D [batch_size, features]
        if molecular_repr.dim() == 1:
            molecular_repr = molecular_repr.unsqueeze(0)
        elif molecular_repr.dim() == 0:
            molecular_repr = molecular_repr.unsqueeze(0).unsqueeze(0)
        
        # Get batch size from molecular representation
        batch_size = molecular_repr.size(0)
        n_features = molecular_repr.size(1)
        
        # Debug
        print(f"DEBUG: molecular_repr shape after processing: {molecular_repr.shape}")
        print(f"DEBUG: Expected n_in={self.n_in}, actual n_features={n_features}")
        
        # If we have more features than expected, we need to project them down
        if n_features != self.n_in:
            # Add a projection layer if not already present
            if not hasattr(self, 'projection_layer'):
                self.projection_layer = torch.nn.Linear(n_features, self.n_in).to(molecular_repr.device)
            molecular_repr = self.projection_layer(molecular_repr)
        
        # Get external descriptors with proper shape handling
        # Extract each descriptor and ensure they're the right shape
        s1_vals = inputs['s1_cis'].view(-1)[:batch_size]
        t1_vals = inputs['t1_cis'].view(-1)[:batch_size]
        exchange_vals = inputs['exchange_integral'].view(-1)[:batch_size]
        dsp_vals = inputs['dsp_scf'].view(-1)[:batch_size]
        homo_lumo_vals = inputs['homo_lumo_overlap'].view(-1)[:batch_size]
        
        # Stack descriptors properly
        descriptors = torch.stack([s1_vals, t1_vals, exchange_vals, dsp_vals, homo_lumo_vals], dim=1)
        
        # Ensure descriptors is 2D [batch_size, n_descriptors]
        if descriptors.dim() == 1:
            descriptors = descriptors.unsqueeze(0)
        
        # Handle case where we have more descriptor values than molecular representations
        if descriptors.size(0) > molecular_repr.size(0):
            descriptors = descriptors[:molecular_repr.size(0)]
        elif descriptors.size(0) < molecular_repr.size(0):
            # Repeat descriptors to match batch size
            repeat_times = molecular_repr.size(0) // descriptors.size(0)
            remainder = molecular_repr.size(0) % descriptors.size(0)
            descriptors = torch.cat([descriptors.repeat(repeat_times, 1), descriptors[:remainder]], dim=0)
        
        # Process descriptors
        descriptor_features = self.descriptor_layer(descriptors)
        
        # Final dimension check before concatenation
        assert molecular_repr.dim() == 2, f"molecular_repr has wrong dimensions: {molecular_repr.shape}"
        assert descriptor_features.dim() == 2, f"descriptor_features has wrong dimensions: {descriptor_features.shape}"
        assert molecular_repr.size(0) == descriptor_features.size(0), f"Batch size mismatch: {molecular_repr.size(0)} vs {descriptor_features.size(0)}"
        assert molecular_repr.size(1) == self.n_in, f"molecular_repr feature mismatch: expected {self.n_in}, got {molecular_repr.size(1)}"
        
        # Combine features
        combined_features = torch.cat([molecular_repr, descriptor_features], dim=1)
        
        # Make prediction
        prediction = self.prediction_layer(combined_features)
        
        return {self.output_key: prediction}

def calculate_comprehensive_metrics(y_true, y_pred, dataset_name="", fold=None):
    """Calculate comprehensive regression metrics"""
    
    # Basic metrics
    r2 = r2_score(y_true, y_pred)
    pearson_r, pearson_p = pearsonr(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    max_ae = max_error(y_true, y_pred)
    
    # Additional metrics
    mse = mean_squared_error(y_true, y_pred)
    relative_mae = mae / np.mean(np.abs(y_true)) if np.mean(np.abs(y_true)) != 0 else np.inf
    
    metrics_dict = {
        'R²': r2,
        'Pearson_r': pearson_r,
        'Pearson_p_value': pearson_p,
        'RMSE': rmse,
        'MAE': mae,
        'MaxAE': max_ae,
        'MSE': mse,
        'Relative_MAE': relative_mae
    }
    
    # Print metrics
    fold_str = f"Fold {fold} - " if fold is not None else ""
    print(f"\n{fold_str}{dataset_name} Metrics:")
    print("-" * 50)
    print(f"R² Score: {r2:.4f}")
    print(f"Pearson r: {pearson_r:.4f} (p-value: {pearson_p:.2e})")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Max Absolute Error: {max_ae:.4f}")
    print(f"Relative MAE: {relative_mae:.4f}")
    
    return metrics_dict

def create_prediction_plots(y_true, y_pred, dataset_name, fold=None, save_dir="plots"):
    """Create comprehensive prediction plots in the specified format"""
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    fold_str = f"_fold_{fold}" if fold is not None else ""
    
    # Create DataFrame for comparison
    comparison_df = pd.DataFrame({
        'Experimental': y_true,
        'Predicted (SchNet)': y_pred,
        'Predicted_LR': y_pred  # Linear regression line (will be fitted below)
    })
    
    # Fit linear regression for the regression line
    from sklearn.linear_model import LinearRegression
    lr_model = LinearRegression()
    x_vals = np.array(y_true).reshape(-1, 1)
    y_vals = np.array(y_pred).reshape(-1, 1)
    lr_model.fit(x_vals, y_vals)
    lr_predictions = lr_model.predict(x_vals).flatten()
    comparison_df['Predicted_LR'] = lr_predictions
    
    # Save comparison to Excel
    excel_filename = f'{save_dir}/{dataset_name}_comparison{fold_str}.xlsx'
    comparison_df.to_excel(excel_filename, index=False)
    
    # Dark bottle green color for the scatterplot points
    dark_bottle_green = '#004953'
    
    # Increase figure size for better visibility
    plt.figure(figsize=(10, 8))
    
    # Scatter plot for Experimental vs. SchNet predictions with dark black borders
    sns.scatterplot(x='Experimental', y='Predicted (SchNet)', 
                   data=comparison_df, label='SchNet Predictions', 
                   edgecolor='black', s=100, color=dark_bottle_green, alpha=0.7)
    
    # Regression line for Linear Regression predictions
    sns.regplot(x='Experimental', y='Predicted_LR', data=comparison_df, 
                color='red', scatter=False, label='Linear Regression', 
                line_kws={'linewidth': 2})
    
    # Calculate confidence interval manually
    ci = 95  # Confidence interval percentage
    y_vals_plot = comparison_df['Predicted_LR']
    
    # Sort the DataFrame by x-values
    sorted_data = comparison_df.sort_values(by='Experimental')
    
    # Calculate standard error of the regression
    se = np.std(y_vals_plot) / np.sqrt(len(y_vals_plot))
    
    # Margin of error for the confidence interval
    from scipy import stats
    alpha = 1 - (ci / 100)
    t_value = stats.t.ppf(1 - alpha/2, len(y_vals_plot) - 1)
    margin_of_error = se * t_value
    
    # Plot shaded region for the confidence interval
    plt.fill_between(sorted_data['Experimental'], 
                    sorted_data['Predicted_LR'] - margin_of_error, 
                    sorted_data['Predicted_LR'] + margin_of_error, 
                    color='red', alpha=0.2, label='CI (95%)')
    
    # Add labels and title
    plt.title(f'Experimental vs. Predicted {dataset_name}{fold_str}', fontweight='bold', fontsize=14)
    plt.xlabel('Experimental Values', fontsize=14)
    plt.ylabel('Predicted Values', fontsize=14)
    
    # Add legend for better interpretation with a black border
    legend = plt.legend()
    legend.get_frame().set_edgecolor('black')
    
    # Customize gridlines
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add a black border around the plot on all four sides
    import matplotlib.patches as patches
    border_rect = patches.Rectangle((-0.01, -0.01), 1.02, 1.02, linewidth=1.8, 
                                   edgecolor='black', facecolor='none', 
                                   transform=plt.gca().transAxes)
    plt.gca().add_patch(border_rect)
    
    # Customize plot aesthetics
    sns.despine()
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    corr_coeff, _ = pearsonr(y_true, y_pred)
    
    # Add RMSE, MAE, and correlation to the plot
    textstr = f'Correlation (r): {corr_coeff:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    
    # Add the text box in the upper left corner
    plt.gca().text(0.8, 0.15, textstr, transform=plt.gca().transAxes, fontsize=12,
                   verticalalignment='top', bbox=props)
    
    # Print the metrics
    print(f'r for Experimental vs. SchNet: {corr_coeff:.4f}')
    print(f'RMSE for Experimental vs. SchNet: {rmse:.4f}')
    print(f'MAE for Experimental vs. SchNet: {mae:.4f}')
    
    # Save the figure in high resolution with a tight border
    output_path = f'{save_dir}/{dataset_name}_correlation_plot{fold_str}.png'
    plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
    
    # Show the plot
    plt.show()
    
    # Also create residuals analysis plot
    create_residuals_analysis(y_true, y_pred, dataset_name, fold, save_dir)

def create_residuals_analysis(y_true, y_pred, dataset_name, fold=None, save_dir="plots"):
    """Create residuals analysis plots"""
    
    fold_str = f"_fold_{fold}" if fold is not None else ""
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{dataset_name} - Residuals Analysis{fold_str}', fontsize=16, fontweight='bold')
    
    # 1. Residuals plot
    residuals = y_pred - y_true
    
    # Save residuals data to Excel
    residuals_df = pd.DataFrame({
        'true_values': y_true,
        'predicted_values': y_pred,
        'residuals': residuals
    })
    residuals_excel_filename = f'{save_dir}/{dataset_name}_residuals_data{fold_str}.xlsx'
    residuals_df.to_excel(residuals_excel_filename, index=False)
    logger.info(f"Residuals data saved to: {residuals_excel_filename}")
    
    axes[0, 0].scatter(y_true, residuals, alpha=0.6, color='green', s=30)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('True Values')
    axes[0, 0].set_ylabel('Residuals (Predicted - True)')
    axes[0, 0].set_title('Residuals Plot')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals histogram
    axes[0, 1].hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Residuals')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Q-Q plot for residuals normality
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Residuals Normality)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Predicted vs Residuals
    axes[1, 1].scatter(y_pred, residuals, alpha=0.6, color='purple', s=30)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Predicted Values')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Predicted vs Residuals')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{dataset_name}_residuals_analysis{fold_str}.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.show()

def extract_molecular_features(atoms_objects, smiles_list=None):
    """Extract molecular features for interpretability analysis"""
    
    features = []
    
    for i, atoms in enumerate(atoms_objects):
        mol_features = {}
        
        # Basic atomic features
        atomic_numbers = atoms.get_atomic_numbers()
        mol_features['n_atoms'] = len(atomic_numbers)
        mol_features['n_heavy_atoms'] = np.sum(atomic_numbers > 1)
        mol_features['n_hydrogen'] = np.sum(atomic_numbers == 1)
        mol_features['n_carbon'] = np.sum(atomic_numbers == 6)
        mol_features['n_nitrogen'] = np.sum(atomic_numbers == 7)
        mol_features['n_oxygen'] = np.sum(atomic_numbers == 8)
        mol_features['n_sulfur'] = np.sum(atomic_numbers == 16)
        mol_features['n_phosphorus'] = np.sum(atomic_numbers == 15)
        mol_features['n_halogen'] = np.sum(np.isin(atomic_numbers, [9, 17, 35, 53]))
        
        # Geometric features
        positions = atoms.get_positions()
        if len(positions) > 1:
            distances = []
            for j in range(len(positions)):
                for k in range(j+1, len(positions)):
                    dist = np.linalg.norm(positions[j] - positions[k])
                    distances.append(dist)
            
            mol_features['mean_distance'] = np.mean(distances) if distances else 0
            mol_features['max_distance'] = np.max(distances) if distances else 0
            mol_features['min_distance'] = np.min(distances) if distances else 0
            mol_features['std_distance'] = np.std(distances) if distances else 0
        else:
            mol_features['mean_distance'] = 0
            mol_features['max_distance'] = 0
            mol_features['min_distance'] = 0
            mol_features['std_distance'] = 0
        
        # Molecular weight
        total_weight = 0
        for an in atomic_numbers:
            element_symbol = atomic_number_map_reverse.get(an, 'H')  # Default to hydrogen if unknown
            weight = atomic_weights.get(element_symbol, 1.0)  # Default to 1.0 if unknown
            total_weight += weight
        mol_features['molecular_weight'] = total_weight
        
        # RDKit features if SMILES available
        if smiles_list and i < len(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smiles_list[i])
                if mol:
                    mol_features['mw'] = rdMolDescriptors.CalcExactMolWt(mol)
                    mol_features['logp'] = rdMolDescriptors.CalcCrippenDescriptors(mol)[0]
                    mol_features['tpsa'] = rdMolDescriptors.CalcTPSA(mol)
                    mol_features['n_rings'] = rdMolDescriptors.CalcNumRings(mol)
                    mol_features['n_aromatic_rings'] = rdMolDescriptors.CalcNumAromaticRings(mol)
                    mol_features['n_rotatable_bonds'] = rdMolDescriptors.CalcNumRotatableBonds(mol)
                    mol_features['n_hbd'] = rdMolDescriptors.CalcNumHBD(mol)
                    mol_features['n_hba'] = rdMolDescriptors.CalcNumHBA(mol)
            except:
                pass
        
        features.append(mol_features)
    
    # Convert to DataFrame
    df = pd.DataFrame(features)
    
    # Fill NaN values
    df = df.fillna(0)
    
    return df.values, list(df.columns)

# Create reverse mapping for molecular weight calculation
atomic_number_map_reverse = {v: k for k, v in atomic_number_map.items()}
atomic_weights = {
    'H': 1.008, 'He': 4.003, 'Li': 6.94, 'Be': 9.012, 'B': 10.81, 'C': 12.01, 'N': 14.01, 'O': 16.00,
    'F': 19.00, 'Ne': 20.18, 'Na': 22.99, 'Mg': 24.31, 'Al': 26.98, 'Si': 28.09, 'P': 30.97, 'S': 32.06,
    'Cl': 35.45, 'Ar': 39.95, 'K': 39.10, 'Ca': 40.08, 'Sc': 44.96, 'Ti': 47.87, 'V': 50.94, 'Cr': 51.996,
    'Mn': 54.94, 'Fe': 55.85, 'Co': 58.93, 'Ni': 58.69, 'Cu': 63.55, 'Zn': 65.38, 'Ga': 69.72, 'Ge': 72.63,
    'As': 74.92, 'Se': 78.97, 'Br': 79.90, 'Kr': 83.80, 'Rb': 85.47, 'Sr': 87.62, 'Y': 88.91, 'Zr': 91.22,
    'Nb': 92.91, 'Mo': 95.95, 'Tc': 98.00, 'Ru': 101.07, 'Rh': 102.91, 'Pd': 106.42, 'Ag': 107.87, 'Cd': 112.41,
    'In': 114.82, 'Sn': 118.71, 'Sb': 121.76, 'Te': 127.60, 'I': 126.90, 'Xe': 131.29, 'Cs': 132.91, 'Ba': 137.33,
    'La': 138.91, 'Ce': 140.12, 'Pr': 140.91, 'Nd': 144.24, 'Pm': 145.00, 'Sm': 150.36, 'Eu': 151.96, 'Gd': 157.25,
    'Tb': 158.93, 'Dy': 162.50, 'Ho': 164.93, 'Er': 167.26, 'Tm': 168.93, 'Yb': 173.05, 'Lu': 174.97, 'Hf': 178.49,
    'Ta': 180.95, 'W': 183.84, 'Re': 186.21, 'Os': 190.23, 'Ir': 192.22, 'Pt': 195.08, 'Au': 196.97, 'Hg': 200.59,
    'Tl': 204.38, 'Pb': 207.20, 'Bi': 208.98, 'Po': 209.00, 'At': 210.00, 'Rn': 222.00, 'Fr': 223.00, 'Ra': 226.00,
    'Ac': 227.00, 'Th': 232.04, 'Pa': 231.04, 'U': 238.03, 'Np': 237.00, 'Pu': 244.00, 'Am': 243.00, 'Cm': 247.00,
    'Bk': 247.00, 'Cf': 251.00, 'Es': 252.00, 'Fm': 257.00, 'Md': 258.00, 'No': 259.00, 'Lr': 266.00, 'Rf': 267.00,
    'Db': 268.00, 'Sg': 269.00, 'Bh': 270.00, 'Hs': 277.00, 'Mt': 278.00, 'Ds': 281.00, 'Rg': 282.00, 'Cn': 285.00,
    'Nh': 286.00, 'Fl': 289.00, 'Mc': 290.00, 'Lv': 293.00, 'Ts': 294.00, 'Og': 294.00
}

def create_feature_importance_plots(features, feature_names, y_true, y_pred, dataset_name, fold=None, save_dir="plots"):
    """Create feature importance analysis using surrogate models"""
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    fold_str = f"_fold_{fold}" if fold is not None else ""
    
    # Use Random Forest as surrogate model for feature importance
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(features, y_pred)  # Fit to predictions to understand model behavior
    
    # Get feature importance
    importance = rf_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Save feature importance data to Excel
    excel_filename = f'{save_dir}/{dataset_name}_feature_importance_data{fold_str}.xlsx'
    importance_df.to_excel(excel_filename, index=False)
    logger.info(f"Feature importance data saved to: {excel_filename}")
    
    # Create feature importance plot
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(15)
    
    bars = plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Feature Importance Analysis - {dataset_name}{fold_str}')
    plt.gca().invert_yaxis()
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_features['importance'])):
        plt.text(value + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{dataset_name}_feature_importance{fold_str}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return importance_df

def create_shap_plots(features, feature_names, y_pred, dataset_name, fold=None, save_dir="plots", max_samples=500):
    """Create publication-ready SHAP plots for model interpretability"""
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    fold_str = f"_fold_{fold}" if fold is not None else ""
    
    # Limit samples for SHAP analysis (computational efficiency)
    if len(features) > max_samples:
        indices = np.random.choice(len(features), max_samples, replace=False)
        features_sample = features[indices]
        y_pred_sample = np.array(y_pred)[indices]
    else:
        features_sample = features
        y_pred_sample = y_pred
    
    # Train a simple model for SHAP analysis
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15)
    rf_model.fit(features_sample, y_pred_sample)
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(features_sample)
    
    # Create feature DataFrame for SHAP
    features_df = pd.DataFrame(features_sample, columns=feature_names)
    
    # Save SHAP values to Excel
    shap_values_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_excel_filename = f'{save_dir}/{dataset_name}_shap_values{fold_str}.xlsx'
    shap_values_df.to_excel(shap_excel_filename, index=False)
    logger.info(f"SHAP values saved to: {shap_excel_filename}")
    
    # Calculate mean absolute SHAP values for each feature and save
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_shap_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap_value': mean_abs_shap
    }).sort_values('mean_abs_shap_value', ascending=False)
    
    mean_shap_excel_filename = f'{save_dir}/{dataset_name}_mean_shap_values{fold_str}.xlsx'
    mean_shap_df.to_excel(mean_shap_excel_filename, index=False)
    logger.info(f"Mean SHAP values saved to: {mean_shap_excel_filename}")
    
    # Create single high-quality SHAP summary plot
    try:
        # Set publication style
        plt.style.use('default')
        
        # Get top features for display (handle case where we have fewer than 15 features)
        feature_importance = np.abs(shap_values).mean(0)
        n_features = len(feature_importance)
        max_display = min(15, n_features)  # Show at most 15 features or all if fewer
        top_indices = np.argsort(feature_importance)[-max_display:]  
        
        # Create single comprehensive SHAP summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values[:, top_indices], features_df.iloc[:, top_indices], 
                         show=False, max_display=max_display, alpha=0.8)
        
        plt.title(f'SHAP Feature Importance Analysis - {dataset_name}{fold_str}', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('SHAP value (impact on model output)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{dataset_name}_shap_analysis{fold_str}.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        logger.info("Publication-ready SHAP summary plot created successfully")
        
    except Exception as shap_error:
        logger.warning(f"Could not create publication-ready SHAP plots: {shap_error}")
        logger.info("Falling back to simple SHAP plots...")
        
        # Fallback to simple plots
        try:
            # Simple summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, features_df, max_display=15)
            plt.title(f'SHAP Summary Plot - {dataset_name}{fold_str}')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/{dataset_name}_shap_summary{fold_str}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Simple bar plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, features_df, plot_type="bar", max_display=15)
            plt.title(f'SHAP Feature Importance - {dataset_name}{fold_str}')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/{dataset_name}_shap_bar{fold_str}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as fallback_error:
            logger.warning(f"Fallback SHAP plots also failed: {fallback_error}")
    
    return shap_values, explainer

def smiles_to_xyz(smiles, output_file=None, conf_id=0):
    """Convert SMILES to XYZ coordinates using RDKit with robust error handling"""
    try:
        # Try to fix common SMILES issues
        import re
        cleaned_smiles = smiles
        
        # The SMILES seem to use non-standard notation for ring fusion
        # Let's try multiple approaches to fix them
        
        # Approach 1: Replace -N with just N (remove negative ring numbers)
        cleaned_smiles = re.sub(r'-(\d+)', r'\1', cleaned_smiles)
        
        # Approach 2: Replace =N with just N  
        cleaned_smiles = re.sub(r'=(\d+)', r'\1', cleaned_smiles)
        
        # Approach 3: Handle specific patterns common in the dataset
        # Replace patterns like c-2 with c2, c=2 with c2, etc.
        cleaned_smiles = re.sub(r'([a-zA-Z])-(\d+)', r'\1\2', cleaned_smiles)
        cleaned_smiles = re.sub(r'([a-zA-Z])=(\d+)', r'\1\2', cleaned_smiles)
        
        # Try original SMILES first
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Try cleaned SMILES
            mol = Chem.MolFromSmiles(cleaned_smiles)
            if mol is None:
                # Try to sanitize the molecule
                try:
                    mol = Chem.MolFromSmiles(smiles, sanitize=False)
                    if mol is not None:
                        Chem.SanitizeMol(mol)
                except:
                    pass
        
        if mol is None:
            return None, None
        
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Try multiple 3D embedding methods
        embed_success = False
        
        # Method 1: Standard embedding
        try:
            result = AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=100)
            if result == 0:  # Success
                AllChem.OptimizeMolecule(mol, maxIters=500)
                embed_success = True
        except:
            pass
        
        # Method 2: ETKDG method (if standard fails)
        if not embed_success:
            try:
                params = AllChem.ETKDG()
                params.randomSeed = 42
                params.maxAttempts = 100
                result = AllChem.EmbedMolecule(mol, params)
                if result == 0:
                    AllChem.OptimizeMolecule(mol, maxIters=500)
                    embed_success = True
            except:
                pass
        
        # Method 3: Use 2D coordinates as fallback (if 3D embedding fails)
        if not embed_success:
            try:
                # Generate 2D coordinates and extend to 3D
                AllChem.Compute2DCoords(mol)
                # Add a z-coordinate of 0 to make it 3D
                embed_success = True
            except:
                pass
        
        # Method 4: Last resort - generate random coordinates
        if not embed_success:
            try:
                # Generate random coordinates
                AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True, maxAttempts=10)
                embed_success = True
            except:
                pass
        
        if not embed_success:
            return None, None
        
        # Get atomic symbols and positions
        atoms = []
        positions = []
        
        try:
            conf = mol.GetConformer(conf_id)
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                pos = conf.GetAtomPosition(atom.GetIdx())
                atoms.append(symbol)
                positions.append([pos.x, pos.y, pos.z])
        except:
            return None, None
        
        # Validate that we have reasonable coordinates
        if len(atoms) == 0 or len(positions) == 0:
            return None, None
        
        # Check for NaN or infinite coordinates
        for pos in positions:
            if any(np.isnan(pos)) or any(np.isinf(pos)):
                return None, None
        
        # Save to XYZ file if specified
        if output_file:
            write_xyz(atoms, positions, output_file)
        
        return atoms, positions
    
    except Exception as e:
        return None, None

def write_xyz(atoms, positions, filename):
    """Write atoms and positions to XYZ file"""
    with open(filename, 'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write("Generated from SMILES\n")
        for atom, pos in zip(atoms, positions):
            f.write(f"{atom} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

def process_smiles_batch(smiles_batch, output_dir, start_idx):
    """Process a batch of SMILES in parallel"""
    results = []
    for i, smiles in enumerate(smiles_batch):
        atoms, positions = smiles_to_xyz(smiles)
        if atoms is not None and positions is not None:
            # Create filename
            filename = os.path.join(output_dir, f"mol_{start_idx + i}.xyz")
            write_xyz(atoms, positions, filename)
            results.append((atoms, positions, start_idx + i))
        else:
            logger.debug(f"Failed to process SMILES at index {start_idx + i}: {smiles}")
    return results

def process_smiles_parallel(smiles_list, output_dir, n_processes=30):
    """Process SMILES list in parallel using multiprocessing"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Split SMILES into batches for parallel processing
    batch_size = len(smiles_list) // n_processes + 1
    batches = [smiles_list[i:i + batch_size] for i in range(0, len(smiles_list), batch_size)]
    
    # Prepare arguments for parallel processing
    args = [(batch, output_dir, i * batch_size) for i, batch in enumerate(batches)]
    
    # Process in parallel
    with Pool(n_processes) as pool:
        batch_results = pool.starmap(process_smiles_batch, args)
    
    # Flatten results
    all_results = []
    for batch_result in batch_results:
        all_results.extend(batch_result)
    
    return all_results

# Read XYZ files
def read_xyz(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    num_atoms = int(lines[0])
    atoms = []
    positions = []
    
    for line in lines[2:2 + num_atoms]:
        parts = line.split()
        element = parts[0]
        x, y, z = map(float, parts[1:4])
        atoms.append(element)
        positions.append([x, y, z])
    
    return atoms, positions

# Convert symbols to atomic numbers with error handling
def symbols_to_atomic_numbers(symbols):
    atomic_numbers = []
    for symbol in symbols:
        if symbol in atomic_number_map:
            atomic_numbers.append(atomic_number_map[symbol])
        else:
            print(f"Warning: Unknown element symbol '{symbol}', skipping...")
            continue
    return atomic_numbers

# Split Roman numerals and integers from filename for Dataset-I. For Dataset-II, use only integers for matching
def split_roman_and_integer(file_name):
    match = re.match(r'([IVXLCDM]+)_([0-9]+)\.xyz', file_name)
    if match:
        roman = match.group(1)
        integer = int(match.group(2))
        return roman, integer
    return None, None

# Extract XYZ files from a directory
def get_xyz_files(directory):
    xyz_files = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.xyz'):
            if '_' in file_name:  # Roman numeral format
                roman, integer = split_roman_and_integer(file_name)
                if roman is not None and integer is not None:
                    xyz_files.append((roman, integer, os.path.join(directory, file_name)))
            else:  # Simple numbering
                xyz_files.append((0, int(file_name.split('.')[0].split('_')[-1]), os.path.join(directory, file_name)))
    
    xyz_files.sort(key=lambda x: (x[0], x[1]))
    return [file[2] for file in xyz_files]

# Process XYZ files into a format compatible with SchNetPack
def process_xyz_files(file_paths):
    dataset_entries = []
    
    for file_path in file_paths:
        atoms, positions = read_xyz(file_path)
        atomic_numbers = symbols_to_atomic_numbers(atoms)
        if len(atomic_numbers) == 0:  # Skip if no valid atoms
            continue
            
        positions = np.array(positions)
        
        entry = {
            'atomic_numbers': torch.tensor(atomic_numbers, dtype=torch.long),
            'positions': torch.tensor(positions, dtype=torch.float32),
        }
        
        dataset_entries.append(entry)
    
    return dataset_entries

# Process the folder containing XYZ files
def process_xyz_folder_to_schnet_format(directory):
    xyz_files = get_xyz_files(directory)
    dataset_entries = process_xyz_files(xyz_files)
    
    all_atomic_numbers = []
    all_positions = []
    
    for entry in dataset_entries:
        all_atomic_numbers.append(entry['atomic_numbers'].tolist())
        all_positions.append(entry['positions'].tolist())
    
    return all_atomic_numbers, all_positions

def process_smiles_to_schnet_format(smiles_list, temp_dir='temp_xyz', n_processes=30):
    """Process SMILES list to SchNet format"""
    logger.info(f"Processing {len(smiles_list)} SMILES strings with {n_processes} processes...")
    
    # Test basic RDKit functionality first
    logger.info(f"🧪 Testing RDKit functionality...")
    try:
        test_mol = Chem.MolFromSmiles('CCO')  # Simple ethanol
        if test_mol is None:
            logger.error("❌ CRITICAL: RDKit cannot parse simple SMILES")
            raise ValueError("RDKit basic functionality test failed")
        logger.info("✓ RDKit basic functionality test passed")
        
        # Test with first actual SMILES from dataset
        if smiles_list:
            logger.info(f"🧪 Testing with actual dataset SMILES...")
            test_smiles = smiles_list[0]
            logger.info(f"   First SMILES: {test_smiles}")
            test_atoms, test_positions = smiles_to_xyz(test_smiles)
            if test_atoms is not None:
                logger.info(f"   ✓ Successfully processed first SMILES: {len(test_atoms)} atoms")
            else:
                logger.warning(f"   ❌ Failed to process first SMILES - will attempt with fallback strategies")
                
    except Exception as rdkit_test_error:
        logger.error(f"❌ RDKit test failed: {rdkit_test_error}")
        # Don't raise here, continue with processing as the robust function might still work
    
    # Try parallel processing first
    try:
        logger.info(f"🔄 Starting parallel SMILES processing with {n_processes} processes...")
        results = process_smiles_parallel(smiles_list, temp_dir, n_processes)
        logger.info(f"Parallel processing returned {len(results)} results")
    except Exception as parallel_error:
        logger.warning(f"Parallel SMILES processing failed: {parallel_error}")
        logger.info("Trying single-threaded SMILES processing...")
        try:
            results = process_smiles_parallel(smiles_list, temp_dir, n_processes=1)
            logger.info(f"Single-threaded processing returned {len(results)} results")
        except Exception as single_error:
            logger.error(f"Single-threaded processing also failed: {single_error}")
            results = []
    
    all_atomic_numbers = []
    all_positions = []
    failed_count = 0
    
    for atoms, positions, idx in results:
        atomic_numbers = symbols_to_atomic_numbers(atoms)
        if len(atomic_numbers) > 0:
            all_atomic_numbers.append(atomic_numbers)
            all_positions.append(positions)
        else:
            failed_count += 1
    
    success_rate = len(all_atomic_numbers) / len(smiles_list) * 100 if len(smiles_list) > 0 else 0
    logger.info(f"📊 SMILES Processing Results:")
    logger.info(f"   ✅ Successfully processed: {len(all_atomic_numbers)}")
    logger.info(f"   ❌ Failed: {len(smiles_list) - len(all_atomic_numbers)}")
    logger.info(f"   📈 Success rate: {success_rate:.1f}%")
    
    if len(all_atomic_numbers) == 0:
        logger.error(f"❌ CRITICAL ERROR: No valid molecules could be processed from SMILES")
        logger.error(f"   This indicates issues with the SMILES format in the dataset")
        logger.error(f"   Sample SMILES: {smiles_list[:5] if smiles_list else 'None'}")
        raise ValueError("No valid molecules could be processed from SMILES")
    
    return all_atomic_numbers, all_positions

# Main processing logic
def main():
    """Main function for SchNet training with comprehensive logging"""
    main_start_time = time.time()
    
    # Set tensor core optimization for better GPU performance
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
        logger.info("Set tensor core precision to 'medium' for optimal GPU performance")
    
    # Log startup information
    logger.info("="*80)
    logger.info("SCHNET DELTA ML TRAINING STARTED")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_filename}")
    logger.info(f"Script: {__file__}")
    logger.info(f"Command line: {' '.join(sys.argv)}")
    
    # Log system information
    log_system_info()
    
    try:
        log_step("ARGUMENT PARSING", "Parsing command line arguments")
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Train SchNet model for delE_CISD prediction')
        parser.add_argument('--use_optuna_params', action='store_true', 
                           help='Use Optuna-optimized hyperparameters from best_delE_CISD_params.json')
        parser.add_argument('--dataset', type=str, default='dataset.csv',
                           help='Path to dataset CSV file (default: dataset.csv)')
        parser.add_argument('--params_file', type=str, default='best_delE_CISD_params.json',
                           help='Path to Optuna parameters file (default: best_delE_CISD_params.json)')
        
        args = parser.parse_args()
        
        logger.info("Command line arguments:")
        logger.info(f"  use_optuna_params: {args.use_optuna_params}")
        logger.info(f"  dataset: {args.dataset}")
        logger.info(f"  params_file: {args.params_file}")
        
        log_step("HYPERPARAMETER LOADING", "Loading model hyperparameters")
        
        # Get hyperparameters
        hyperparams = get_hyperparameters(use_optuna=args.use_optuna_params)
        logger.info(f"Hyperparameters loaded successfully")
        
        # Calculate optimal parallelization settings
        available_cores = mp.cpu_count()
        requested_cores = min(62, available_cores)
        logger.info(f"Parallelization settings:")
        logger.info(f"  Available CPU cores: {available_cores}")
        logger.info(f"  Requested cores: 62")
        logger.info(f"  Using cores: {requested_cores}")
        
        log_step("DATASET LOADING", f"Loading dataset from {args.dataset}")
    
        # Read the dataset
        data_file = args.dataset
        logger.info(f"Checking dataset file: {data_file}")
        logger.debug(f"File exists: {os.path.exists(data_file)}")
        logger.debug(f"File size: {os.path.getsize(data_file) if os.path.exists(data_file) else 'N/A'} bytes")
        
        if not os.path.exists(data_file):
            error_msg = f"Dataset file '{data_file}' not found. Please provide the correct filename."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.info(f"Reading dataset from {data_file}...")
        start_read = time.time()
        data = pd.read_csv(data_file)
        read_duration = time.time() - start_read
        logger.info(f"Dataset loaded successfully in {read_duration:.2f} seconds")
        
        # Log dataset basic info
        logger.info(f"Dataset shape: {data.shape}")
        logger.info(f"Dataset columns: {list(data.columns)}")
        logger.info(f"Dataset size: {len(data)} samples")
        logger.info(f"Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Check required columns
        required_columns = ['smiles', 's1_cis', 't1_cis', 'exchange_integral', 'dsp_scf', 'homo_lumo_overlap', 'delE_CISD']
        logger.info("Validating required columns...")
        logger.debug(f"Required columns: {required_columns}")
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            logger.error(error_msg)
            logger.error(f"Available columns: {list(data.columns)}")
            raise ValueError(error_msg)
        
        logger.info("✓ All required columns found")
        
        # Log data statistics
        logger.info("Dataset statistics:")
        for col in ['s1_cis', 't1_cis', 'exchange_integral', 'dsp_scf', 'homo_lumo_overlap', 'delE_CISD']:
            if col in data.columns:
                stats = data[col].describe()
                logger.info(f"  {col}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, range=[{stats['min']:.4f}, {stats['max']:.4f}]")
        
        # Check for missing values
        logger.info("Checking for missing values...")
        null_counts = data[required_columns].isnull().sum()
        total_nulls = null_counts.sum()
        if total_nulls > 0:
            logger.warning(f"Found {total_nulls} missing values:")
            for col, count in null_counts.items():
                if count > 0:
                    logger.warning(f"  {col}: {count} missing values")
        else:
            logger.info("✓ No missing values found")
        
        log_step("DATA EXTRACTION", "Extracting data arrays from dataset")
        
        # Extract data
        smiles_list = data['smiles'].tolist()
        s1_cis = data['s1_cis'].tolist()
        t1_cis = data['t1_cis'].tolist()
        exchange_integral = data['exchange_integral'].tolist()
        dsp_scf = data['dsp_scf'].tolist()
        homo_lumo_overlap = data['homo_lumo_overlap'].tolist()
        delE_CISD = data['delE_CISD'].tolist()
        
        logger.info("Data extraction completed:")
        logger.info(f"  SMILES: {len(smiles_list)} entries")
        logger.info(f"  Properties: {len(s1_cis)} entries each")
        
        # Check for empty or invalid SMILES
        empty_smiles = sum(1 for s in smiles_list if not s or s.strip() == '')
        if empty_smiles > 0:
            logger.warning(f"Found {empty_smiles} empty SMILES strings")
        
        logger.info(f"Sample SMILES: {smiles_list[:5]}")  # Show first 5 SMILES
        
        log_step("SMILES PROCESSING", "Converting SMILES to 3D molecular structures")
        
        # Process SMILES to get atomic numbers and positions
        logger.info(f"Processing {len(smiles_list)} SMILES strings with {min(62, mp.cpu_count())} parallel processes...")
        logger.info(f"First 3 SMILES: {smiles_list[:3]}")
        smiles_start_time = time.time()
        
        # Try single-threaded first if parallel fails
        try:
            schnet_compatible_data = process_smiles_to_schnet_format(smiles_list, n_processes=min(62, mp.cpu_count()))
        except Exception as e:
            logger.warning(f"Parallel SMILES processing failed: {e}")
            logger.info("Trying single-threaded SMILES processing...")
            schnet_compatible_data = process_smiles_to_schnet_format(smiles_list, n_processes=1)
        atoms_list = schnet_compatible_data[0]
        positions_list = schnet_compatible_data[1]
        smiles_duration = time.time() - smiles_start_time
        logger.info(f"SMILES processing completed in {smiles_duration:.2f} seconds")
        logger.info(f"Successfully processed {len(atoms_list)} molecules")

        log_step("DATA ALIGNMENT", "Aligning molecular data arrays")
        
        # Ensure data alignment
        min_length = min(len(atoms_list), len(positions_list), len(s1_cis), len(t1_cis), 
                         len(exchange_integral), len(dsp_scf), len(homo_lumo_overlap), len(delE_CISD))
        
        logger.info(f"Data alignment: minimum length = {min_length} samples")
        
        atoms_list = atoms_list[:min_length]
        positions_list = positions_list[:min_length]
        s1_cis = s1_cis[:min_length]
        t1_cis = t1_cis[:min_length]
        exchange_integral = exchange_integral[:min_length]
        dsp_scf = dsp_scf[:min_length]
        homo_lumo_overlap = homo_lumo_overlap[:min_length]
        delE_CISD = delE_CISD[:min_length]
        smiles_list = smiles_list[:min_length]
        
        logger.info(f"Aligned all arrays to {min_length} samples")

        log_step("ASE OBJECTS CREATION", "Creating ASE atoms objects with properties")
        
        atoms_objects = []
        property_list = []
        
        for i, (atomic_numbers, positions, s1, t1, exchange, dsp, homo_lumo, target) in enumerate(
            zip(atoms_list, positions_list, s1_cis, t1_cis, exchange_integral, dsp_scf, homo_lumo_overlap, delE_CISD)):
            
            if (i + 1) % 100 == 0:
                log_progress(i + 1, min_length, "molecules")
            
            atoms = Atoms(numbers=atomic_numbers, positions=positions)
            properties = {
                's1_cis': np.array([s1]),
                't1_cis': np.array([t1]),
                'exchange_integral': np.array([exchange]),
                'dsp_scf': np.array([dsp]),
                'homo_lumo_overlap': np.array([homo_lumo]),
                'delE_CISD': np.array([target])
            }
            atoms_objects.append(atoms)
            property_list.append(properties)
        
        logger.info(f"Created {len(atoms_objects)} ASE atoms objects")

        # Create a new ASE dataset
        log_step("SCHNET DATABASE CREATION", "Creating SchNetPack database")
        
        new_datatut = './delE_CISD_dataset'
        if not os.path.exists(new_datatut):
            os.makedirs(new_datatut)
            logger.info(f"Created directory: {new_datatut}")

        db_path = './delE_CISD_dataset.db'
        if os.path.exists(db_path):
            os.remove(db_path)
            logger.info(f"Removed existing database: {db_path}")

        logger.info("Creating SchNetPack ASEAtomsData database...")
        new_dataset = spk.data.ASEAtomsData.create(
            db_path,
            distance_unit='Ang',
            property_unit_dict={
                's1_cis': 'eV',
                't1_cis': 'eV',
                'exchange_integral': '1',  # Dimensionless
                'dsp_scf': '1',  # Dimensionless
                'homo_lumo_overlap': '1',  # Dimensionless
                'delE_CISD': 'eV'
            }
        )

        logger.info("Adding molecular systems to database...")
        logger.info(f"Atoms objects: {len(atoms_objects)}, Property list: {len(property_list)}")
        if len(atoms_objects) == 0:
            raise ValueError("No atoms objects created. Check SMILES processing.")
        new_dataset.add_systems(property_list, atoms_objects)
        logger.info(f'Number of reference calculations: {len(new_dataset)}')
        
        if len(new_dataset) == 0:
            raise ValueError("No systems were added to the database. Check your input data.")
        
        logger.info('Available properties:')
        for p in new_dataset.available_properties:
            logger.info(f'  - {p}')

        example = new_dataset[0]
        logger.info('Properties of molecule with id 0:')
        for k, v in example.items():
            logger.info(f'  - {k}: {v.shape}')

        log_step("DATA SPLITTING", "Creating train-test splits")
        
        total_samples = len(new_dataset)
        logger.info(f"Total samples for splitting: {total_samples}")

        # 80-20 train-test split
        train_indices, test_indices = train_test_split(
            np.arange(total_samples), 
            test_size=0.2, 
            random_state=42
        )

        logger.info(f"Training set size: {len(train_indices)}")
        logger.info(f"Test set size: {len(test_indices)}")

        # Extract molecular features for interpretability analysis
        log_step("FEATURE EXTRACTION", "Extracting molecular features for interpretability")
        
        logger.info("Extracting molecular features for interpretability analysis...")
        molecular_features, feature_names = extract_molecular_features(atoms_objects, smiles_list)
        logger.info(f"Extracted {len(feature_names)} molecular features")

        # Create split file for SchNetPack (numpy format)
        split_file_path = os.path.join(new_datatut, 'train_test_split.npz')
        
        # Save as numpy arrays (SchNetPack expected format)
        np.savez(split_file_path,
                 train_idx=train_indices,
                 val_idx=np.array([]),  # Empty validation set
                 test_idx=test_indices)

        logger.info(f"Split file saved to {split_file_path}")

        log_step("DATA MODULE SETUP", "Setting up SchNetPack data module")
        
        # Setup data module with optimized hyperparameters
        logger.info("Creating AtomsDataModule with hyperparameters:")
        logger.info(f"  Batch size: {hyperparams['bs']}")
        logger.info(f"  Cutoff: {hyperparams['cutoff']}")
        logger.info(f"  Workers: {min(62, mp.cpu_count())}")
        
        newdata = spk.data.AtomsDataModule(
            db_path,
            batch_size=min(32, hyperparams['bs']),  # Reduce batch size to prevent memory issues
            num_train=len(train_indices),
            num_val=0,  # No validation set
            distance_unit='Ang',
            property_units={'delE_CISD': 'eV'},
            transforms=[
                spk.transform.ASENeighborList(cutoff=hyperparams['cutoff']),
                spk.transform.RemoveOffsets('delE_CISD', remove_mean=True, remove_atomrefs=False),
                spk.transform.CastTo32()
            ],
            num_workers=4,  # Reduce workers to prevent multiprocessing crashes
            pin_memory=False,  # Disable pin_memory to reduce memory pressure
            split_file=split_file_path,
            load_properties=['delE_CISD', 's1_cis', 't1_cis', 'exchange_integral', 'dsp_scf', 'homo_lumo_overlap']
        )

        logger.info("Preparing data...")
        newdata.prepare_data()
        logger.info("Setting up data...")
        newdata.setup()
        logger.info("Data module setup completed")

        log_step("MODEL INITIALIZATION", "Initializing SchNet model with external descriptors")
        
        # Create the simple combined model (adapted from t1optuna.py working approach)
        logger.info("Model Configuration:")
        logger.info("-" * 40)
        logger.info(f"Cutoff distance: {hyperparams['cutoff']} Å")
        logger.info(f"Atom basis dimension: {hyperparams['n_atom_basis']}")
        logger.info(f"Number of interactions: {hyperparams['n_interactions']}")
        logger.info(f"Number of RBF functions: {hyperparams['n_rbf']}")
        logger.info(f"Batch size: {hyperparams['bs']}")
        logger.info(f"Learning rate: {hyperparams['lr']}")

        # Create the simple combined model
        model = SimpleMolecularModel(
            n_atom_basis=hyperparams['n_atom_basis'],
            n_interactions=hyperparams['n_interactions'],
            cutoff=float(hyperparams['cutoff']),
            n_rbf=hyperparams['n_rbf'],
            n_descriptors=5
        )
        logger.info("SimpleMolecularModel created")

        # Create a simple wrapper that works with SchNetPack
        pairwise_distance = spk.atomistic.PairwiseDistances()
        
        class SimpleNNPot(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.pairwise_distance = pairwise_distance
                self.postprocessors = torch.nn.ModuleList([])
                
                # Required SchNetPack attributes
                self.required_derivatives = []
                self.model_outputs = ['delE_CISD']
                self.do_postprocessing = True
                self.input_modules = []
            
            def initialize_transforms(self, datamodule):
                """Initialize transforms - required by SchNetPack"""
                logger.info("Initializing transforms for SimpleNNPot...")
                if len(self.postprocessors) == 0:
                    logger.info("No postprocessors to initialize (simplified model)")
                logger.info("Transform initialization completed")
            
            def reset_parameters(self):
                """Reset parameters - sometimes called by SchNetPack"""
                if hasattr(self.model, 'reset_parameters'):
                    self.model.reset_parameters()
            
            def calculate_interatomic_vectors(self, inputs):
                """Calculate interatomic vectors - compatibility method"""
                return self.pairwise_distance(inputs)
            
            def forward(self, inputs):
                # Apply pairwise distance calculation
                inputs = self.pairwise_distance(inputs)
                
                # Get model predictions
                outputs = self.model(inputs)
                
                # Apply postprocessors (currently none to keep things simple)
                for postprocessor in self.postprocessors:
                    outputs = postprocessor(outputs)
                
                return outputs
        
        nnpot = SimpleNNPot(model)
        logger.info("Simple neural network potential wrapper created")

        output_delE = spk.task.ModelOutput(
            name='delE_CISD',
            loss_fn=torch.nn.MSELoss(),
            loss_weight=1.,
            metrics={
                'MAE': torchmetrics.MeanAbsoluteError(),
                'RMSE': torchmetrics.MeanSquaredError()
            }
        )

        task = spk.task.AtomisticTask(
            model=nnpot,
            outputs=[output_delE],
            optimizer_cls=torch.optim.AdamW,
            optimizer_args={'lr': hyperparams['lr']},
            scheduler_cls=spk.train.ReduceLROnPlateau,
            scheduler_monitor='train_loss',  # Monitor train loss since no validation
            scheduler_args={'mode': 'min', 'factor': 0.5, 'patience': hyperparams['pat2'], 'threshold_mode': 'rel', 'cooldown': 5},
        )
        logger.info("Atomistic task created")
        
        log_step("TRAINER SETUP", "Setting up PyTorch Lightning trainer")
        
        # TensorBoard Logger to visualize the model performance
        tb_logger = pl.loggers.TensorBoardLogger(save_dir=new_datatut)
        callbacks = [
            pl.callbacks.ModelCheckpoint(
                dirpath=new_datatut,
                filename='best_model_schnet',
                save_top_k=1,
                monitor='train_loss',
                mode='min',
                save_last=True
            ),
            pl.callbacks.EarlyStopping(monitor='train_loss', mode='min', patience=hyperparams['patience'], min_delta=0),
            pl.callbacks.LearningRateMonitor(logging_interval='step')
        ]

        # Fit the model with parallel processing support
        trainer = pl.Trainer(
            max_epochs=2000,
            logger=tb_logger,
            callbacks=callbacks,
            default_root_dir=new_datatut,
            accelerator='auto',
            devices='auto',
            strategy='ddp' if torch.cuda.device_count() > 1 else 'auto',
            accumulate_grad_batches=1,
            val_check_interval=1.0,
            check_val_every_n_epoch=1,
            num_sanity_val_steps=0,
            fast_dev_run=False,
            enable_checkpointing=True,
            overfit_batches=0,
            limit_train_batches=1.0,
            limit_test_batches=1.0,
            log_every_n_steps=5
        )
        logger.info("Trainer configured successfully")

        log_step("MODEL TRAINING", "Starting model training")
        
        logger.info("Starting model training...")
        training_start_time = time.time()
        trainer.fit(task, datamodule=newdata)
        training_duration = time.time() - training_start_time
        logger.info(f"Training completed in {training_duration:.2f} seconds")

        log_step("MODEL EVALUATION", "Loading best model and evaluating performance")
        
        # Load the best model
        best_model_path = os.path.join(new_datatut, 'best_inference_model')
        
        # Look for checkpoint files with PyTorch Lightning naming convention
        checkpoint_files = []
        if os.path.exists(new_datatut):
            for f in os.listdir(new_datatut):
                if f.endswith('.ckpt'):
                    checkpoint_files.append(os.path.join(new_datatut, f))
        
        if checkpoint_files:
            best_model_path = checkpoint_files[0]  # Use first checkpoint found
            logger.info(f"Loading model checkpoint from: {best_model_path}")
            
            # Load the checkpoint
            checkpoint = torch.load(best_model_path, map_location='cpu')
            
            # Recreate the model architecture
            eval_model = SimpleMolecularModel(
                n_atom_basis=hyperparams['n_atom_basis'],
                n_interactions=hyperparams['n_interactions'],
                cutoff=float(hyperparams['cutoff']),
                n_rbf=hyperparams['n_rbf'],
                n_descriptors=5
            )
            
            # Load state dict from checkpoint
            if 'state_dict' in checkpoint:
                # Remove the 'model.model.' prefix from checkpoint keys
                state_dict = {}
                for k, v in checkpoint['state_dict'].items():
                    if k.startswith('model.model.'):
                        new_key = k.replace('model.model.', '')
                        state_dict[new_key] = v
                
                eval_model.load_state_dict(state_dict, strict=False)
                logger.info("Successfully loaded model weights from checkpoint")
            else:
                logger.warning("Could not find state_dict in checkpoint - using random weights")
            
            best_model = eval_model
        else:
            logger.warning(f"No checkpoint files found in {new_datatut}")
            # Create a new model with random weights as fallback
            best_model = SimpleMolecularModel(
                n_atom_basis=hyperparams['n_atom_basis'],
                n_interactions=hyperparams['n_interactions'],
                cutoff=float(hyperparams['cutoff']),
                n_rbf=hyperparams['n_rbf'],
                n_descriptors=5
            )
        
        actual_delE_CISD_train = []
        predicted_delE_CISD_train = []
        
        best_model.eval()
        logger.info("Evaluating on training set...")

        with torch.no_grad():
            # Prediction on training set
            for batch in newdata.train_dataloader():
                actual_batch = batch['delE_CISD'].cpu().numpy().tolist()
                actual_delE_CISD_train.extend(actual_batch)

                # Apply pairwise distances for the model
                pairwise_distance = spk.atomistic.PairwiseDistances()
                batch = pairwise_distance(batch)
                
                result = best_model(batch)
                predicted_batch = result['delE_CISD'].cpu().detach().numpy().tolist()
                predicted_delE_CISD_train.extend(predicted_batch)

        # Calculate comprehensive metrics for training
        train_metrics = calculate_comprehensive_metrics(
            actual_delE_CISD_train, 
            predicted_delE_CISD_train,
            "Train delE_CISD"
        )
        
        # Create prediction plots for training
        create_prediction_plots(
            np.array(actual_delE_CISD_train),
            np.array(predicted_delE_CISD_train),
            "Train_delE_CISD"
        )

        logger.info("Evaluating on test set...")
        actual_delE_CISD_test = []
        predicted_delE_CISD_test = []
        
        with torch.no_grad():
            # Prediction on test set
            for batch in newdata.test_dataloader():
                actual_batch = batch['delE_CISD'].cpu().numpy().tolist()
                actual_delE_CISD_test.extend(actual_batch)

                # Apply pairwise distances for the model
                pairwise_distance = spk.atomistic.PairwiseDistances()
                batch = pairwise_distance(batch)
                
                result = best_model(batch)
                predicted_batch = result['delE_CISD'].cpu().detach().numpy().tolist()
                predicted_delE_CISD_test.extend(predicted_batch)

        # Calculate comprehensive metrics for test
        test_metrics = calculate_comprehensive_metrics(
            actual_delE_CISD_test,
            predicted_delE_CISD_test,
            "Test delE_CISD"
        )
        
        # Create prediction plots for test
        create_prediction_plots(
            np.array(actual_delE_CISD_test),
            np.array(predicted_delE_CISD_test),
            "Test_delE_CISD"
        )
        
        log_step("INTERPRETABILITY ANALYSIS", "Creating feature importance and SHAP analysis")
        
        # Extract features for test set (full molecular features for general feature importance)
        test_molecular_features = molecular_features[test_indices]
        
        # Feature importance analysis for all molecular features
        logger.info("Creating feature importance analysis...")
        importance_df = create_feature_importance_plots(
            test_molecular_features, 
            feature_names,
            np.array(actual_delE_CISD_test),
            np.array(predicted_delE_CISD_test),
            "Test_delE_CISD"
        )
        
        # Extract only the 5 external descriptors for SHAP analysis
        logger.info("Preparing external descriptors for SHAP analysis...")
        external_descriptors_test = []
        external_descriptor_names = ['s1_cis', 't1_cis', 'exchange_integral', 'dsp_scf', 'homo_lumo_overlap']
        
        # Get the 5 external descriptors for test samples
        test_s1_cis = np.array(s1_cis)[test_indices]
        test_t1_cis = np.array(t1_cis)[test_indices] 
        test_exchange_integral = np.array(exchange_integral)[test_indices]
        test_dsp_scf = np.array(dsp_scf)[test_indices]
        test_homo_lumo_overlap = np.array(homo_lumo_overlap)[test_indices]
        
        # Stack the 5 external descriptors
        external_descriptors_test = np.column_stack([
            test_s1_cis,
            test_t1_cis, 
            test_exchange_integral,
            test_dsp_scf,
            test_homo_lumo_overlap
        ])
        
        logger.info(f"External descriptors shape: {external_descriptors_test.shape}")
        logger.info(f"External descriptor names: {external_descriptor_names}")
        
        # SHAP analysis for external descriptors only
        logger.info("Creating SHAP analysis for external descriptors...")
        shap_values, explainer = create_shap_plots(
            external_descriptors_test,
            external_descriptor_names,
            predicted_delE_CISD_test,
            "Test_delE_CISD_External_Descriptors"
        )
        logger.info(f"SHAP analysis completed with {len(shap_values)} samples")
        logger.info(f"SHAP explainer type: {type(explainer).__name__}")
        
        log_step("RESULTS SAVING", "Saving comprehensive results")
        
        # Save detailed results
        results = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': importance_df.to_dict('records'),
            'external_descriptors_used': ['s1_cis', 't1_cis', 'exchange_integral', 'dsp_scf', 'homo_lumo_overlap']
        }
        
        # Save results to JSON
        with open('delE_CISD_prediction_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save comprehensive metrics to Excel
        metrics_df = pd.DataFrame({
            'Dataset': ['Train', 'Test'],
            'R²': [train_metrics['R²'], test_metrics['R²']],
            'Pearson_r': [train_metrics['Pearson_r'], test_metrics['Pearson_r']],
            'Pearson_p_value': [train_metrics['Pearson_p_value'], test_metrics['Pearson_p_value']],
            'RMSE': [train_metrics['RMSE'], test_metrics['RMSE']],
            'MAE': [train_metrics['MAE'], test_metrics['MAE']],
            'MaxAE': [train_metrics['MaxAE'], test_metrics['MaxAE']],
            'MSE': [train_metrics['MSE'], test_metrics['MSE']],
            'Relative_MAE': [train_metrics['Relative_MAE'], test_metrics['Relative_MAE']]
        })
        metrics_excel_filename = 'plots/delE_CISD_metrics_summary.xlsx'
        metrics_df.to_excel(metrics_excel_filename, index=False)
        logger.info(f"Metrics summary saved to: {metrics_excel_filename}")
        
        # Save predictions vs actual values for both train and test sets
        train_predictions_df = pd.DataFrame({
            'actual': actual_delE_CISD_train,
            'predicted': predicted_delE_CISD_train
        })
        train_predictions_excel = 'plots/Train_delE_CISD_predictions.xlsx'
        train_predictions_df.to_excel(train_predictions_excel, index=False)
        logger.info(f"Training predictions saved to: {train_predictions_excel}")
        
        test_predictions_df = pd.DataFrame({
            'actual': actual_delE_CISD_test,
            'predicted': predicted_delE_CISD_test
        })
        test_predictions_excel = 'plots/Test_delE_CISD_predictions.xlsx'
        test_predictions_df.to_excel(test_predictions_excel, index=False)
        logger.info(f"Test predictions saved to: {test_predictions_excel}")
        
        logger.info("Results saved successfully")
        
        # Log successful completion
        main_duration = time.time() - main_start_time
        logger.info("="*80)
        logger.info("SCHNET DELTA ML TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"Total execution time: {main_duration:.2f} seconds")
        logger.info(f"Final Test Pearson r: {test_metrics['Pearson_r']:.4f}")
        logger.info(f"Final Test RMSE: {test_metrics['RMSE']:.4f}")
        logger.info(f"Final Test MAE: {test_metrics['MAE']:.4f}")
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if args.use_optuna_params:
            logger.info(f"✅ Used Optuna-optimized hyperparameters from: {args.params_file}")
        else:
            logger.info("💡 To use Optuna-optimized hyperparameters:")
            logger.info("   1. Run: python t1optuna.py")
            logger.info("   2. Then: python schnet.py --use_optuna_params")
        
        logger.info("="*80)

    except Exception as e:
        # Log any critical errors
        main_duration = time.time() - main_start_time
        logger.error("="*80)
        logger.error("CRITICAL ERROR IN MAIN EXECUTION")
        logger.error("="*80)
        logger.error(f"Execution time before error: {main_duration:.2f} seconds")
        log_error_with_traceback("Critical error in main function", e)
        
        # Clean up any partial files
        try:
            cleanup_files = [
                './delE_CISD_dataset.db',
                './delE_CISD_dataset/train_test_split.npz',
                'temp_xyz'
            ]
            logger.info("Attempting cleanup of partial files...")
            for file_path in cleanup_files:
                if os.path.exists(file_path):
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        logger.info(f"Cleaned up file: {file_path}")
                    elif os.path.isdir(file_path):
                        import shutil
                        shutil.rmtree(file_path)
                        logger.info(f"Cleaned up directory: {file_path}")
        except Exception as cleanup_error:
            logger.warning(f"Cleanup failed: {cleanup_error}")
        
        # Re-raise the exception to ensure proper exit code
        raise

if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    main()    
