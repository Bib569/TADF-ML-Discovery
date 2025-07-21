import optuna
import torch
import schnetpack as spk
import pytorch_lightning as pl
import schnetpack.transform as trn
import torchmetrics
import os
import re
import numpy as np
import torch
from ase import Atoms
import pandas as pd
from schnetpack.data import ASEAtomsData
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
import warnings
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, max_error
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch.nn.functional as F
import argparse
import sys
import json
import logging
from datetime import datetime
try:
    import psutil
except ImportError:
    psutil = None
warnings.filterwarnings('ignore')

# Set up comprehensive logging
def setup_logging():
    """Set up comprehensive logging for debugging"""
    # Create logs directory if it doesn't exist
    if not os.path.exists('optuna_logs'):
        os.makedirs('optuna_logs')
    
    # Create timestamp for unique log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'optuna_logs/optuna_debug_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w'),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - log file: {log_filename}")
    return logger, log_filename

# Initialize logger with unique name to avoid conflicts
debug_logger, log_file = setup_logging()

# Simple combined model for molecular property prediction
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
        # Debug logging
        if not hasattr(self, '_debug_logged'):
            debug_logger.info(f"SimpleMolecularModel - Batch keys: {list(batch.keys())}")
            self._debug_logged = True
        
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


# Custom SchNet representation with external descriptors
class SchNetWithDescriptors(torch.nn.Module):
    def __init__(self, n_atom_basis=64, n_interactions=3, cutoff=5.0, n_rbf=20, n_descriptors=5):
        super(SchNetWithDescriptors, self).__init__()
        
        # SchNet representation
        self.schnet = spk.representation.SchNet(
            n_atom_basis=n_atom_basis,
            n_interactions=n_interactions,
            radial_basis=spk.nn.GaussianRBF(n_rbf=n_rbf, cutoff=cutoff),
            cutoff_fn=spk.nn.CosineCutoff(cutoff),
            activation=F.silu
        )
        
        # Layers to process external descriptors
        self.descriptor_layer = torch.nn.Sequential(
            torch.nn.Linear(n_descriptors, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU()
        )
        
        self.n_atom_basis = n_atom_basis
        
    def forward(self, batch):
        # Debug: Print available batch keys on first call
        if not hasattr(self, '_debug_printed'):
            debug_logger.info(f"   🔍 SchNetWithDescriptors batch keys: {list(batch.keys())}")
            print(f"   🔍 SchNetWithDescriptors batch keys: {list(batch.keys())}")
            self._debug_printed = True
        
        # Get SchNet representation
        schnet_output = self.schnet(batch)
        
        # Debug: Print SchNet output shape
        if not hasattr(self, '_debug_shape_printed'):
            debug_logger.info(f"   🔍 SchNet output shape: {schnet_output['scalar_representation'].shape}")
            print(f"   🔍 SchNet output shape: {schnet_output['scalar_representation'].shape}")
            self._debug_shape_printed = True
        
        # Process molecular representation - proper aggregation using idx_m
        atom_repr = schnet_output['scalar_representation']  # [n_atoms, n_features]
        
        # Get molecular indices for proper aggregation
        idx_m = batch.get('_idx_m', batch.get('idx_m'))
        if idx_m is None:
            # Fallback: assume all atoms belong to molecule 0
            idx_m = torch.zeros(atom_repr.size(0), dtype=torch.long, device=atom_repr.device)
        
        # Get number of molecules
        n_molecules = int(idx_m.max().item()) + 1
        
        # Aggregate atom representations to molecular representations
        molecular_repr = torch.zeros(n_molecules, atom_repr.size(1), device=atom_repr.device)
        for i in range(n_molecules):
            mask = (idx_m == i)
            if mask.sum() > 0:
                molecular_repr[i] = atom_repr[mask].mean(dim=0)  # Use mean instead of sum for better scaling
        
        # Get external descriptors
        descriptors = torch.stack([
            batch['s1_cis'].squeeze(),
            batch['t1_cis'].squeeze(),
            batch['exchange_integral'].squeeze(),
            batch['dsp_scf'].squeeze(),
            batch['homo_lumo_overlap'].squeeze()
        ], dim=1)
        
        # Ensure descriptors is 2D [batch_size, n_descriptors]
        if descriptors.dim() == 1:
            descriptors = descriptors.unsqueeze(0)
        
        # Ensure we have the right number of descriptors for each molecule
        if descriptors.size(0) != n_molecules:
            if descriptors.size(0) == 1:
                descriptors = descriptors.expand(n_molecules, -1)
            else:
                # Take only the first n_molecules descriptors
                descriptors = descriptors[:n_molecules]
        
        # Process descriptors
        descriptor_features = self.descriptor_layer(descriptors)
        
        # Combine features - this is the final representation
        combined_features = torch.cat([molecular_repr, descriptor_features], dim=1)
        
        # Debug: Print final shape
        if not hasattr(self, '_debug_final_printed'):
            debug_logger.info(f"   🔍 Final combined features shape: {combined_features.shape}")
            print(f"   🔍 Final combined features shape: {combined_features.shape}")
            debug_logger.info(f"      - Molecular repr: {molecular_repr.shape}")
            print(f"      - Molecular repr: {molecular_repr.shape}")
            debug_logger.info(f"      - Descriptor features: {descriptor_features.shape}")
            print(f"      - Descriptor features: {descriptor_features.shape}")
            self._debug_final_printed = True
        
        # Return representation in SchNet format - return per-atom representation!
        # We need to expand the molecular features back to per-atom for SchNet compatibility
        expanded_features = combined_features[idx_m]  # Map molecular features back to atoms
        
        # IMPORTANT: Pass through the molecular indices key that Atomwise expects
        output = {'scalar_representation': expanded_features}
        
        # Add the idx_m key (without underscore) that Atomwise expects
        if '_idx_m' in batch:
            output['idx_m'] = batch['_idx_m']
        elif 'idx_m' in batch:
            output['idx_m'] = batch['idx_m']
        
        return output


# Custom output module that works with our combined representation
class CombinedOutput(torch.nn.Module):
    """Output module that processes combined SchNet + descriptor features"""
    
    def __init__(self, n_in, n_out=1, n_layers=3, n_hidden=None, activation=F.silu):
        super(CombinedOutput, self).__init__()
        
        if n_hidden is None:
            n_hidden = [64, 32]
        
        layers = []
        in_features = n_in
        
        # Build hidden layers
        for hidden_size in n_hidden:
            layers.append(torch.nn.Linear(in_features, hidden_size))
            layers.append(torch.nn.SiLU())
            in_features = hidden_size
        
        # Output layer
        layers.append(torch.nn.Linear(in_features, n_out))
        
        self.mlp = torch.nn.Sequential(*layers)
        self.output_key = 'delE_CISD'
        
    def forward(self, inputs):
        # Get the representation
        repr = inputs['scalar_representation']
        
        # Get molecular indices - handle both with and without underscore
        idx_m = inputs.get('idx_m', inputs.get('_idx_m'))
        
        if idx_m is None:
            # If no molecular indices, treat all atoms as one molecule
            idx_m = torch.zeros(repr.size(0), dtype=torch.long, device=repr.device)
        
        # Get number of molecules
        n_molecules = int(idx_m.max().item()) + 1
        
        # Aggregate per-atom features to per-molecule features
        mol_features = torch.zeros(n_molecules, repr.size(1), device=repr.device)
        for i in range(n_molecules):
            mask = (idx_m == i)
            if mask.sum() > 0:
                mol_features[i] = repr[mask].mean(dim=0)
        
        # Apply MLP to get predictions
        predictions = self.mlp(mol_features)
        
        # Flatten predictions to match target shape
        predictions = predictions.squeeze(-1)
        
        return {self.output_key: predictions}


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
            print(f"Failed to process SMILES at index {start_idx + i}: {smiles}")
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

def read_xyz(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    num_atoms = int(lines[0])
    atoms = []
    positions = []
    
    for line in lines[2:2+num_atoms]:
        parts = line.split()
        element = parts[0]
        x, y, z = map(float, parts[1:4])
        atoms.append(element)
        positions.append([x, y, z])
    
    return atoms, positions

def symbols_to_atomic_numbers(symbols):
    atomic_numbers = []
    for symbol in symbols:
        if symbol in atomic_number_map:
            atomic_numbers.append(atomic_number_map[symbol])
        else:
            print(f"Warning: Unknown element symbol '{symbol}', skipping...")
            continue
    return atomic_numbers

def split_roman_and_integer(file_name):
    match = re.match(r'([IVXLCDM]+)_([0-9]+)\.xyz', file_name)
    if match:
        roman = match.group(1)
        integer = int(match.group(2))
        return roman, integer
    return None, None

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
    
    # Sort files by Roman numeral first, then by the integer part
    xyz_files.sort(key=lambda x: (x[0], x[1]))
    return [file[2] for file in xyz_files]

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
    print(f"Processing {len(smiles_list)} SMILES strings with {n_processes} processes...")
    
    # Process SMILES in parallel
    results = process_smiles_parallel(smiles_list, temp_dir, n_processes)
    
    all_atomic_numbers = []
    all_positions = []
    
    for atoms, positions, idx in results:
        atomic_numbers = symbols_to_atomic_numbers(atoms)
        if len(atomic_numbers) > 0:
            all_atomic_numbers.append(atomic_numbers)
            all_positions.append(positions)
    
    return all_atomic_numbers, all_positions

# Define the improved objective function with comprehensive logging
def objective(trial, dataset_file):
    global debug_logger  # Make logger accessible within this function
    logger = debug_logger  # Alias for easier use - this will be our debug logger throughout
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 STARTING TRIAL {trial.number}")
    logger.info(f"{'='*60}")
    logger.info(f"⏰ Trial start time: {pd.Timestamp.now()}")
    logger.info(f"📁 Dataset file: {dataset_file}")
    
    # Log system information
    logger.info(f"💻 System info:")
    logger.info(f"   - CPU count: {mp.cpu_count()}")
    if psutil:
        logger.info(f"   - Available memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    logger.info(f"   - Python version: {sys.version}")
    
    # Suggest hyperparameters FIRST (before any data processing)
    logger.info(f"🎛️  Suggesting hyperparameters...")
    bs = trial.suggest_categorical('bs', [300, 400, 500, 600, 700, 800, 900, 1000])
    logger.info(f"   ✓ batch_size: {bs}")
    
    lr = trial.suggest_categorical('lr', [1e-4, 5e-4, 1e-5, 5e-5])
    logger.info(f"   ✓ learning_rate: {lr}")
    
    patience = trial.suggest_int('patience', 1, 10)
    logger.info(f"   ✓ patience: {patience}")
    
    n_rbf = trial.suggest_int('n_rbf', 15, 20)
    logger.info(f"   ✓ n_rbf: {n_rbf}")
    
    n_interactions = trial.suggest_categorical('n_interactions', [3, 4, 5, 6])
    logger.info(f"   ✓ n_interactions: {n_interactions}")
    
    n_atom_basis = trial.suggest_categorical('n_atom_basis', [30, 64, 128])
    logger.info(f"   ✓ n_atom_basis: {n_atom_basis}")
    
    cutoff = trial.suggest_int('cutoff', 3, 8)
    logger.info(f"   ✓ cutoff: {cutoff}")
    
    pat2 = trial.suggest_int('pat2', 1, 10)
    logger.info(f"   ✓ scheduler_patience: {pat2}")
    
    logger.info(f"\n📋 HYPERPARAMETERS SUMMARY:")
    logger.info(f"   batch_size={bs}, lr={lr}, cutoff={cutoff}")
    logger.info(f"   n_rbf={n_rbf}, n_interactions={n_interactions}, n_atom_basis={n_atom_basis}")
    logger.info(f"   patience={patience}, scheduler_patience={pat2}")
    
    # Immediate basic checks before entering main try block
    logger.info(f"\n🔍 PRE-FLIGHT CHECKS")
    logger.info(f"{'='*40}")
    logger.info(f"✓ Trial number: {trial.number}")
    logger.info(f"✓ Dataset file: {dataset_file}")
    logger.info(f"✓ File exists: {os.path.exists(dataset_file)}")
    logger.info(f"✓ Current working directory: {os.getcwd()}")
    logger.info(f"✓ Python executable: {sys.executable}")
    
    try:
        # Step 1: Read the dataset
        logger.info(f"\n📊 STEP 1: READING DATASET")
        logger.info(f"{'='*50}")
        logger.info(f"📁 Checking file existence: {dataset_file}")
        if not os.path.exists(dataset_file):
            logger.error(f"❌ ERROR: Dataset file not found!")
            raise FileNotFoundError(f"Dataset file '{dataset_file}' not found.")
        logger.info(f"✓ File exists")
        
        logger.info(f"📖 Loading CSV file...")
        try:
            data = pd.read_csv(dataset_file)
            logger.info(f"✓ CSV loaded successfully")
        except Exception as csv_error:
            logger.error(f"❌ ERROR loading CSV: {csv_error}")
            raise
        
        logger.info(f"📏 Dataset shape: {data.shape}")
        logger.info(f"📋 Dataset columns: {list(data.columns)}")
        logger.info(f"🔢 Total samples: {len(data)}")
        
        # Check required columns
        logger.info(f"🔍 Validating required columns...")
        required_columns = ['smiles', 's1_cis', 't1_cis', 'exchange_integral', 'dsp_scf', 'homo_lumo_overlap', 'delE_CISD']
        logger.info(f"   Required: {required_columns}")
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"❌ ERROR: Missing required columns: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")
        logger.info(f"✓ All required columns present")
        
        # Check for optional columns (that we recognize but don't use)
        optional_columns = ['oscillator_strength']
        found_optional = [col for col in optional_columns if col in data.columns]
        if found_optional:
            logger.info(f"📋 Found optional columns (will be ignored): {found_optional}")
        
        # Check for null values
        logger.info(f"🔍 Checking for null values...")
        null_counts = data[required_columns].isnull().sum()
        total_nulls = null_counts.sum()
        logger.info(f"   Null values per column: {dict(null_counts)}")
        logger.info(f"   Total null values: {total_nulls}")
        
        # Take only first N samples for faster processing during optimization
        original_len = len(data)
        max_samples = min(100, len(data))  # Limit samples for faster optimization
        data = data.head(max_samples)
        logger.info(f"📊 Dataset sampling: {original_len} → {len(data)} samples for optimization")
        
        # Show sample data statistics
        logger.info(f"📈 Data statistics for target variable (delE_CISD):")
        target_stats = data['delE_CISD'].describe()
        for stat, value in target_stats.items():
            logger.info(f"   {stat}: {value:.4f}")
        
        logger.info(f"✅ STEP 1 COMPLETED: Dataset ready with {len(data)} samples")
        
        # Extract data
        logger.info(f"🔄 Extracting data arrays...")
        smiles_list = data['smiles'].tolist()
        logger.info(f"   ✓ SMILES: {len(smiles_list)} entries")
        s1_cis = data['s1_cis'].tolist()
        logger.info(f"   ✓ s1_cis: {len(s1_cis)} entries")
        t1_cis = data['t1_cis'].tolist()
        logger.info(f"   ✓ t1_cis: {len(t1_cis)} entries")
        exchange_integral = data['exchange_integral'].tolist()
        logger.info(f"   ✓ exchange_integral: {len(exchange_integral)} entries")
        dsp_scf = data['dsp_scf'].tolist()
        logger.info(f"   ✓ dsp_scf: {len(dsp_scf)} entries")
        homo_lumo_overlap = data['homo_lumo_overlap'].tolist()
        logger.info(f"   ✓ homo_lumo_overlap: {len(homo_lumo_overlap)} entries")
        delE_CISD = data['delE_CISD'].tolist()
        logger.info(f"   ✓ delE_CISD: {len(delE_CISD)} entries")
        
        # Step 2: Process SMILES to 3D structures (single-threaded for stability)
        logger.info(f"\n🧪 STEP 2: PROCESSING SMILES TO 3D STRUCTURES")
        logger.info(f"{'='*50}")
        logger.info(f"🔬 Processing mode: Single-threaded for stability")
        logger.info(f"📝 Total SMILES to process: {len(smiles_list)}")
        
        # Test basic RDKit functionality first
        logger.info(f"🧪 Testing RDKit functionality...")
        try:
            test_mol = Chem.MolFromSmiles('CCO')  # Simple ethanol
            if test_mol is None:
                logger.error("❌ CRITICAL: RDKit cannot parse simple SMILES")
                raise ValueError("RDKit basic functionality test failed")
            logger.info("✓ RDKit basic functionality test passed")
            
            # Test with first actual SMILES from dataset
            logger.info(f"🧪 Testing with actual dataset SMILES...")
            test_smiles = smiles_list[0]
            logger.info(f"   First SMILES: {test_smiles}")
            test_atoms, test_positions = smiles_to_xyz(test_smiles)
            if test_atoms is not None:
                logger.info(f"   ✓ Successfully processed first SMILES: {len(test_atoms)} atoms")
            else:
                logger.warning(f"   ❌ Failed to process first SMILES - this indicates dataset SMILES issues")
                
        except Exception as rdkit_test_error:
            logger.error(f"❌ RDKit test failed: {rdkit_test_error}")
            raise
        
        atoms_list = []
        positions_list = []
        valid_indices = []
        failed_smiles = []
        
        logger.info(f"🔄 Starting SMILES processing...")
        for i, smiles in enumerate(smiles_list):
            logger.info(f"   Processing {i+1}/{len(smiles_list)}: {smiles[:50]}{'...' if len(smiles) > 50 else ''}")
            try:
                atoms, positions = smiles_to_xyz(smiles)
                if atoms is not None and positions is not None:
                    logger.info(f"      ✓ 3D structure generated: {len(atoms)} atoms")
                    atomic_numbers = symbols_to_atomic_numbers(atoms)
                    if len(atomic_numbers) > 0:
                        atoms_list.append(atomic_numbers)
                        positions_list.append(positions)
                        valid_indices.append(i)
                        logger.info(f"      ✓ Valid atomic numbers: {atomic_numbers[:5]}{'...' if len(atomic_numbers) > 5 else ''}")
                    else:
                        logger.warning(f"      ❌ No valid atomic numbers generated")
                        failed_smiles.append((i, smiles, "No valid atomic numbers"))
                else:
                    logger.warning(f"      ❌ 3D structure generation failed")
                    failed_smiles.append((i, smiles, "3D generation failed"))
            except Exception as e:
                logger.error(f"      ❌ Exception: {str(e)}")
                failed_smiles.append((i, smiles, str(e)))
                continue
        
        print(f"\n📊 SMILES Processing Results:")
        print(f"   ✅ Successfully processed: {len(atoms_list)}")
        print(f"   ❌ Failed: {len(failed_smiles)}")
        print(f"   📈 Success rate: {len(atoms_list)/len(smiles_list)*100:.1f}%")
        
        if len(failed_smiles) > 0:
            print(f"   ⚠️  Failed SMILES details:")
            for idx, smiles, error in failed_smiles[:3]:  # Show first 3 failures
                print(f"      {idx}: {smiles[:30]}... → {error}")
            if len(failed_smiles) > 3:
                print(f"      ... and {len(failed_smiles)-3} more")
        
        if len(atoms_list) == 0:
            print(f"❌ CRITICAL ERROR: No valid molecules could be processed from SMILES")
            raise ValueError("No valid molecules could be processed from SMILES")
        
        print(f"✅ STEP 2 COMPLETED: {len(atoms_list)} molecules ready for training")
        
        # Filter other data to match valid molecules
        print(f"\n🔄 Filtering data to match valid molecules...")
        print(f"   Valid molecule indices: {valid_indices[:10]}{'...' if len(valid_indices) > 10 else ''}")
        
        s1_cis = [s1_cis[i] for i in valid_indices]
        print(f"   ✓ s1_cis filtered: {len(s1_cis)} values")
        t1_cis = [t1_cis[i] for i in valid_indices]
        print(f"   ✓ t1_cis filtered: {len(t1_cis)} values")
        exchange_integral = [exchange_integral[i] for i in valid_indices]
        print(f"   ✓ exchange_integral filtered: {len(exchange_integral)} values")
        dsp_scf = [dsp_scf[i] for i in valid_indices]
        print(f"   ✓ dsp_scf filtered: {len(dsp_scf)} values")
        homo_lumo_overlap = [homo_lumo_overlap[i] for i in valid_indices]
        print(f"   ✓ homo_lumo_overlap filtered: {len(homo_lumo_overlap)} values")
        delE_CISD = [delE_CISD[i] for i in valid_indices]
        print(f"   ✓ delE_CISD filtered: {len(delE_CISD)} values")
        
        # Step 3: Create SchNet dataset
        print(f"\n⚛️  STEP 3: CREATING SCHNET DATASET")
        print(f"{'='*50}")
        print(f"🔬 Creating ASE Atoms objects with properties...")
        
        atoms_objects = []
        property_list = []
        failed_objects = []

        for i, (atomic_numbers, positions, s1, t1, exchange, dsp, homo_lumo, target) in enumerate(
            zip(atoms_list, positions_list, s1_cis, t1_cis, exchange_integral, dsp_scf, homo_lumo_overlap, delE_CISD)):
            
            print(f"   Creating object {i+1}/{len(atoms_list)}...")
            print(f"      🧮 Atomic numbers: {len(atomic_numbers)} atoms")
            print(f"      📍 Positions shape: {np.array(positions).shape}")
            print(f"      🎯 Properties: s1={s1:.4f}, t1={t1:.4f}, target={target:.4f}")
            
            try:
                atoms = Atoms(numbers=atomic_numbers, positions=positions)
                print(f"      ✓ ASE Atoms object created")
                
                properties = {
                    's1_cis': np.array([s1]),
                    't1_cis': np.array([t1]),
                    'exchange_integral': np.array([exchange]),
                    'dsp_scf': np.array([dsp]),
                    'homo_lumo_overlap': np.array([homo_lumo]),
                    'delE_CISD': np.array([target])
                }
                print(f"      ✓ Properties dictionary created")
                
                atoms_objects.append(atoms)
                property_list.append(properties)
                print(f"      ✅ Object {i+1} successfully added")
                
            except Exception as e:
                print(f"      ❌ Error creating atom object {i}: {e}")
                failed_objects.append((i, str(e)))
                continue
        
        print(f"\n📊 Atom Object Creation Results:")
        print(f"   ✅ Successfully created: {len(atoms_objects)}")
        print(f"   ❌ Failed: {len(failed_objects)}")
        if len(failed_objects) > 0:
            print(f"   ⚠️  Failed objects:")
            for idx, error in failed_objects[:3]:
                print(f"      {idx}: {error}")
        
        if len(atoms_objects) == 0:
            print(f"❌ CRITICAL ERROR: No valid atom objects could be created")
            raise ValueError("No valid atom objects could be created")
        
        print(f"✅ STEP 3 COMPLETED: {len(atoms_objects)} atom objects ready")
        
        # Step 4: Create SchNet database
        print(f"\n💾 STEP 4: CREATING SCHNET DATABASE")
        print(f"{'='*50}")
        
        db_path = f'./optuna_dataset_trial_{trial.number}.db'
        print(f"📁 Database path: {db_path}")
        
        if os.path.exists(db_path):
            print(f"🗑️  Removing existing database file...")
            os.remove(db_path)
            print(f"   ✓ Existing file removed")
        
        print(f"🏗️  Creating new ASEAtomsData database...")
        try:
            new_dataset = spk.data.ASEAtomsData.create(
                db_path,
                distance_unit='Ang',
                property_unit_dict={
                    's1_cis': 'eV',
                    't1_cis': 'eV',
                    'exchange_integral': '1',
                    'dsp_scf': '1',
                    'homo_lumo_overlap': '1',
                    'delE_CISD': 'eV'
                }
            )
            print(f"   ✓ Database structure created")
        except Exception as db_create_error:
            print(f"   ❌ Database creation failed: {db_create_error}")
            raise

        print(f"📝 Adding systems to database...")
        print(f"   Property list length: {len(property_list)}")
        print(f"   Atoms objects length: {len(atoms_objects)}")
        
        try:
            new_dataset.add_systems(property_list, atoms_objects)
            print(f"   ✓ Systems added successfully")
        except Exception as add_error:
            print(f"   ❌ Adding systems failed: {add_error}")
            raise
        
        dataset_length = len(new_dataset)
        print(f"📊 Database statistics:")
        print(f"   Total samples: {dataset_length}")
        print(f"   Database file size: {os.path.getsize(db_path) / 1024:.2f} KB")
        
        # Test database access
        print(f"🔍 Testing database access...")
        try:
            first_sample = new_dataset[0]
            print(f"   ✓ First sample accessible")
            print(f"   Available properties: {list(first_sample.keys())}")
        except Exception as access_error:
            print(f"   ❌ Database access failed: {access_error}")
            raise
        
        print(f"✅ STEP 4 COMPLETED: SchNet database with {dataset_length} samples")

        # Step 5: Create data splits
        print(f"\n🎯 STEP 5: CREATING DATA SPLITS")
        print(f"{'='*50}")
        
        new_datatut = f'./optuna_dataset_trial_{trial.number}'
        print(f"📁 Working directory: {new_datatut}")
        
        if not os.path.exists(new_datatut):
            print(f"🏗️  Creating working directory...")
            os.makedirs(new_datatut) 
            print(f"   ✓ Directory created")
        else:
            print(f"   ✓ Directory already exists")
            
        total_samples = len(new_dataset)
        print(f"📊 Total samples available: {total_samples}")
        
        if total_samples < 10:
            print(f"❌ CRITICAL ERROR: Too few samples for training")
            raise ValueError(f"Too few samples ({total_samples}) for training")

        # 80-20 train-test split
        test_ratio = 0.2
        print(f"🎲 Creating train-test split (ratio: {1-test_ratio:.0%}-{test_ratio:.0%})...")
        
        num_test = max(1, int(total_samples * test_ratio))  # At least 1 test sample
        num_train = total_samples - num_test
        
        print(f"   Train samples: {num_train}")
        print(f"   Test samples: {num_test}")

        print(f"🔀 Shuffling indices...")
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        print(f"   ✓ Indices shuffled")

        train_indices = indices[:num_train].tolist()
        test_indices = indices[num_train:].tolist()
        
        logger.info(f"📋 Split summary:")
        logger.info(f"   Train indices: {train_indices[:5]}{'...' if len(train_indices) > 5 else ''}")
        logger.info(f"   Test indices: {test_indices[:5]}{'...' if len(test_indices) > 5 else ''}")

        split_data = {
            'train_idx': np.array(train_indices),
            'val_idx': np.array([]),  # Empty validation set
            'test_idx': np.array(test_indices)
        }
        logger.info(f"   ✓ Split data structure created")

        # Save split file in NumPy format (not JSON) as SchNet expects
        split_file_path = f'split_file_trial_{trial.number}.npz'
        split_file_full_path = os.path.join(new_datatut, split_file_path)
        logger.info(f"💾 Saving split file: {split_file_full_path}")
        
        try:
            np.savez(split_file_full_path, **split_data)
            logger.info(f"   ✓ Split file saved successfully in NumPy format")
        except Exception as save_error:
            logger.error(f"   ❌ Failed to save split file: {save_error}")
            raise

        logger.info(f"✅ STEP 5 COMPLETED: Train={len(train_indices)}, Test={len(test_indices)}")

        # Step 6: Create data module
        logger.info(f"\n📦 STEP 6: CREATING DATA MODULE")
        logger.info(f"{'='*50}")
        
        effective_batch_size = min(bs, num_train)
        logger.info(f"⚙️  DataModule configuration:")
        logger.info(f"   Database path: {db_path}")
        logger.info(f"   Requested batch size: {bs}")
        logger.info(f"   Effective batch size: {effective_batch_size}")
        logger.info(f"   Number of train samples: {num_train}")
        logger.info(f"   Cutoff distance: {cutoff} Å")
        logger.info(f"   Split file: {split_file_full_path}")
        
        logger.info(f"🔧 Creating transforms...")
        transforms = [
            trn.ASENeighborList(cutoff=float(cutoff)),
            trn.CastTo32()
        ]
        logger.info(f"   ✓ ASENeighborList (cutoff={cutoff})")
        logger.info(f"   ✓ CastTo32")
        logger.info(f"   - Removed RemoveOffsets for simplified training")
        
        logger.info(f"🏗️  Creating AtomsDataModule...")
        try:
            newdata = spk.data.AtomsDataModule(
                db_path,
                batch_size=effective_batch_size,
                num_train=num_train,
                num_val=0,
                distance_unit='Ang',
                property_units={'delE_CISD': 'eV'},
                transforms=transforms,
                num_workers=0,  # Disable multiprocessing for stability
                split_file=split_file_full_path,
                pin_memory=False,  # Disable pin_memory for stability
                load_properties=['delE_CISD', 's1_cis', 't1_cis', 'exchange_integral', 'dsp_scf', 'homo_lumo_overlap']
            )
            logger.info(f"   ✓ DataModule created successfully")
        except Exception as dm_error:
            logger.error(f"   ❌ DataModule creation failed: {dm_error}")
            raise

        print(f"\n🔄 STEP 7: PREPARING DATA")
        print(f"{'='*50}")
        
        print(f"📊 Preparing data...")
        try:
            newdata.prepare_data()
            print(f"   ✓ Data preparation completed")
        except Exception as prep_error:
            print(f"   ❌ Data preparation failed: {prep_error}")
            raise
        
        print(f"⚙️  Setting up data...")
        try:
            newdata.setup() 
            print(f"   ✓ Data setup completed")
        except Exception as setup_error:
            print(f"   ❌ Data setup failed: {setup_error}")
            raise
        
        # Test data loaders
        print(f"🔍 Testing data loaders...")
        try:
            train_loader = newdata.train_dataloader()
            print(f"   ✓ Train loader: {len(train_loader)} batches")
            
            test_loader = newdata.test_dataloader()
            print(f"   ✓ Test loader: {len(test_loader)} batches")
            
            # Test loading one batch
            first_batch = next(iter(train_loader))
            print(f"   ✓ First batch loaded successfully")
            print(f"      Batch keys: {list(first_batch.keys())}")
            print(f"      Batch size: {first_batch['_atomic_numbers'].shape[0]}")
            
        except Exception as loader_error:
            print(f"   ❌ Data loader test failed: {loader_error}")
            raise
        
        print(f"✅ STEP 6-7 COMPLETED: Data module ready")
        
        # Step 8: Create model
        print(f"\n🧠 STEP 8: CREATING MODEL")
        print(f"{'='*50}")
        
        print(f"🏗️  Creating SimpleMolecularModel...")
        print(f"   Model parameters:")
        print(f"      n_atom_basis: {n_atom_basis}")
        print(f"      n_interactions: {n_interactions}")
        print(f"      cutoff: {cutoff}")
        print(f"      n_rbf: {n_rbf}")
        print(f"      n_descriptors: 5")
        
        try:
            # Create the simple combined model
            model = SimpleMolecularModel(
                n_atom_basis=n_atom_basis,
                n_interactions=n_interactions,
                cutoff=float(cutoff),
                n_rbf=n_rbf,
                n_descriptors=5
            )
            print(f"   ✓ SimpleMolecularModel created")
            
            # Test model parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"   Model statistics:")
            print(f"      Total parameters: {total_params:,}")
            print(f"      Trainable parameters: {trainable_params:,}")
            
        except Exception as model_error:
            print(f"   ❌ Model creation failed: {model_error}")
            raise

        print(f"🔗 Creating neural network potential...")
        try:
            # Create a simple wrapper that works with SchNetPack
            pairwise_distance = spk.atomistic.PairwiseDistances()
            
            # Simple wrapper that satisfies SchNetPack requirements
            class SimpleNNPot(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                    self.pairwise_distance = pairwise_distance
                    # For now, remove problematic postprocessors to get basic training working
                    # We can add them back later once the core model works
                    self.postprocessors = torch.nn.ModuleList([])
                    
                    # Required SchNetPack attributes
                    self.required_derivatives = []
                    self.model_outputs = ['delE_CISD']
                    self.do_postprocessing = True  # Required by SchNetPack
                    self.input_modules = []  # Required for some SchNetPack operations
                
                def initialize_transforms(self, datamodule):
                    """Initialize transforms - required by SchNetPack"""
                    # This method is called by the trainer to initialize transforms
                    # For our simple model, we don't need to do anything special here
                    print(f"   🔧 Initializing transforms for SimpleNNPot...")
                    
                    # Initialize postprocessors if they have an initialize method
                    if len(self.postprocessors) == 0:
                        print(f"      - No postprocessors to initialize (simplified model)")
                    else:
                        for i, postprocessor in enumerate(self.postprocessors):
                            name = f"Postprocessor_{i}"
                            if hasattr(postprocessor, 'initialize') and callable(getattr(postprocessor, 'initialize')):
                                try:
                                    postprocessor.initialize(datamodule)
                                    print(f"      ✓ Initialized {name}: {type(postprocessor).__name__}")
                                except Exception as e:
                                    print(f"      ⚠️  Failed to initialize {name}: {e}")
                            else:
                                print(f"      - {name} ({type(postprocessor).__name__}) does not need initialization")
                    
                    print(f"   ✅ Transform initialization completed")
                
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
                    # Future: Add postprocessors here once basic training works
                    for postprocessor in self.postprocessors:
                        outputs = postprocessor(outputs)
                    
                    return outputs
            
            nnpot = SimpleNNPot(model)
            print(f"   ✓ Simple neural network potential created")
            
        except Exception as nnpot_error:
            print(f"   ❌ Neural network potential creation failed: {nnpot_error}")
            raise

        print(f"🎯 Creating model output configuration...")
        try:
            output_delE = spk.task.ModelOutput(
                name='delE_CISD',
                loss_fn=torch.nn.MSELoss(),
                loss_weight=1.,
                metrics={
                    'MAE': torchmetrics.MeanAbsoluteError(),
                    'RMSE': torchmetrics.MeanSquaredError()
                }
            )
            print(f"   ✓ Model output configuration created")
        except Exception as output_error:
            print(f"   ❌ Model output creation failed: {output_error}")
            raise
        
        print(f"⚙️  Creating atomistic task...")
        print(f"   Task configuration:")
        print(f"      Optimizer: AdamW")
        print(f"      Learning rate: {lr}")
        print(f"      Scheduler: ReduceLROnPlateau")
        print(f"      Scheduler patience: {pat2}")
        
        try:
            task = spk.task.AtomisticTask(
                model=nnpot,
                outputs=[output_delE],
                optimizer_cls=torch.optim.AdamW,
                optimizer_args={'lr': lr},
                scheduler_cls=spk.train.ReduceLROnPlateau,
                scheduler_monitor='train_loss',
                scheduler_args={'mode': 'min', 'factor': 0.5, 'patience': pat2, 'threshold_mode': 'rel', 'cooldown': 2},
            )
            print(f"   ✓ Atomistic task created")
        except Exception as task_error:
            print(f"   ❌ Atomistic task creation failed: {task_error}")
            raise
        
        print(f"✅ STEP 8 COMPLETED: Model ready for training")

        # Step 9: Create trainer
        print(f"\n🏃 STEP 9: CREATING TRAINER")
        print(f"{'='*50}")
        
        print(f"📊 Creating TensorBoard logger...")
        try:
            tb_logger = pl.loggers.TensorBoardLogger(save_dir=new_datatut)
            print(f"   ✓ Logger created: {new_datatut}")
        except Exception as logger_error:
            print(f"   ❌ Logger creation failed: {logger_error}")
            raise

        print(f"⚙️  Creating trainer callbacks...")
        model_checkpoint_path = os.path.join(new_datatut, f'best_inference_model_trial_{trial.number}')
        print(f"   Model checkpoint path: {model_checkpoint_path}")
        
        try:
            callbacks = [
                pl.callbacks.ModelCheckpoint(
                    dirpath=new_datatut,
                    filename=f'best_model_trial_{trial.number}',
                    save_top_k=1,
                    monitor='train_loss',
                    mode='min',
                    save_last=True
                ),
                pl.callbacks.EarlyStopping(monitor='train_loss', mode='min', patience=patience, min_delta=0.001),
                pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
            ]
            print(f"   ✓ PyTorch Lightning ModelCheckpoint callback created")
            print(f"   ✓ EarlyStopping callback created (patience={patience})")
            print(f"   ✓ LearningRateMonitor callback created")
        except Exception as callback_error:
            print(f"   ❌ Callback creation failed: {callback_error}")
            raise

        print(f"🏗️  Creating PyTorch Lightning trainer...")
        trainer_config = {
            'max_epochs': 20,
            'accelerator': 'cpu',
            'devices': 1,
            'num_sanity_val_steps': 0,
            'enable_checkpointing': True,
            'limit_train_batches': 1.0,
            'limit_test_batches': 1.0,
            'log_every_n_steps': 5,
            'enable_progress_bar': False
        }
        print(f"   Trainer configuration: {trainer_config}")
        
        try:
            trainer = pl.Trainer(
                callbacks=callbacks,
                logger=tb_logger,
                default_root_dir=new_datatut,
                **trainer_config
            )
            print(f"   ✓ Trainer created successfully")
        except Exception as trainer_error:
            print(f"   ❌ Trainer creation failed: {trainer_error}")
            raise

        # Step 10: Train model
        print(f"\n🚀 STEP 10: TRAINING MODEL")
        print(f"{'='*50}")
        print(f"⏰ Training start time: {pd.Timestamp.now()}")
        print(f"🎯 Maximum epochs: {trainer_config['max_epochs']}")
        print(f"⏹️  Early stopping patience: {patience}")
        
        try:
            print(f"🔥 Starting training process...")
            trainer.fit(task, datamodule=newdata)
            print(f"✅ Training completed successfully!")
            print(f"⏰ Training end time: {pd.Timestamp.now()}")
            
            # Log training statistics
            print(f"📊 Training statistics:")
            print(f"   Current epoch: {trainer.current_epoch}")
            print(f"   Global step: {trainer.global_step}")
            if hasattr(trainer, 'callback_metrics'):
                print(f"   Final metrics: {dict(trainer.callback_metrics)}")
                
        except Exception as training_error:
            print(f"❌ Training failed: {training_error}")
            import traceback
            print(f"Training error traceback:\n{traceback.format_exc()}")
            raise
        
        # Get train loss for optimization
        print(f"\n📈 EXTRACTING TRAINING METRICS")
        print(f"{'='*50}")
        
        try:
            train_loss = trainer.callback_metrics.get('train_loss', torch.tensor(1000.0)).item()
            print(f"📊 Final training loss: {train_loss:.6f}")
        except Exception as loss_error:
            print(f"❌ Error extracting training loss: {loss_error}")
            train_loss = 1000.0
        
        # Step 11: Evaluate model
        print(f"\n🔬 STEP 11: EVALUATING MODEL")
        print(f"{'='*50}")
        
        # Look for the checkpoint file with PyTorch Lightning naming convention
        checkpoint_name = f'best_model_trial_{trial.number}.ckpt'
        best_model_path = os.path.join(new_datatut, checkpoint_name)
        
        print(f"📁 Looking for model checkpoint: {best_model_path}")
        
        # If specific checkpoint not found, look for any .ckpt files
        if not os.path.exists(best_model_path):
            print(f"⚠️  Specific checkpoint not found, searching for any .ckpt files...")
            if os.path.exists(new_datatut):
                ckpt_files = [f for f in os.listdir(new_datatut) if f.endswith('.ckpt')]
                print(f"   Found checkpoint files: {ckpt_files}")
                if ckpt_files:
                    best_model_path = os.path.join(new_datatut, ckpt_files[0])
                    print(f"   Using: {best_model_path}")
                else:
                    print(f"   No checkpoint files found")
                    print(f"   Directory contents: {os.listdir(new_datatut)}")
                    print(f"   Fallback: Using training loss as final score")
                    return min(train_loss, 1000.0)
            else:
                print(f"   Directory not found: {new_datatut}")
                print(f"   Fallback: Using training loss as final score")
                return min(train_loss, 1000.0)
        
        print(f"✅ Model file found: {os.path.getsize(best_model_path) / 1024:.2f} KB")
        
        try:
            print(f"🔄 Loading best model...")
            # Load the checkpoint
            checkpoint = torch.load(best_model_path, map_location='cpu')
            
            # Try to recreate the model architecture
            eval_model = SimpleMolecularModel(
                n_atom_basis=n_atom_basis,
                n_interactions=n_interactions,
                cutoff=float(cutoff),
                n_rbf=n_rbf,
                n_descriptors=5
            )
            
            print(f"   📋 Checkpoint type: {type(checkpoint)}")
            print(f"   📋 Available keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
            
            # Try different ways to load the state dict
            loaded = False
            if isinstance(checkpoint, dict):
                # Method 1: Direct state_dict
                if 'state_dict' in checkpoint:
                    try:
                        # Remove 'model.' prefix if present
                        state_dict = {}
                        for k, v in checkpoint['state_dict'].items():
                            if k.startswith('model.model.'):
                                state_dict[k.replace('model.model.', '')] = v
                            elif k.startswith('model.'):
                                state_dict[k.replace('model.', '')] = v
                            else:
                                state_dict[k] = v
                        eval_model.load_state_dict(state_dict, strict=False)
                        loaded = True
                        print(f"   ✓ Loaded from 'state_dict' key")
                    except Exception as e:
                        print(f"   ⚠️  Failed to load from state_dict: {e}")
                
                # Method 2: Try other common keys
                if not loaded:
                    for key in ['model_state_dict', 'model', 'network']:
                        if key in checkpoint:
                            try:
                                eval_model.load_state_dict(checkpoint[key], strict=False)
                                loaded = True
                                print(f"   ✓ Loaded from '{key}' key")
                                break
                            except Exception as e:
                                print(f"   ⚠️  Failed to load from {key}: {e}")
                
                # Method 3: Try to extract from top-level
                if not loaded:
                    try:
                        # Filter out non-model keys
                        model_state = {}
                        for k, v in checkpoint.items():
                            if not k.startswith(('epoch', 'lr_schedulers', 'optimizer', 'global_step')):
                                if k.startswith('model.model.'):
                                    model_state[k.replace('model.model.', '')] = v
                                elif k.startswith('model.'):
                                    model_state[k.replace('model.', '')] = v
                                else:
                                    model_state[k] = v
                        
                        if model_state:
                            eval_model.load_state_dict(model_state, strict=False)
                            loaded = True
                            print(f"   ✓ Loaded from filtered checkpoint keys")
                    except Exception as e:
                        print(f"   ⚠️  Failed to load filtered state: {e}")
            
            if not loaded:
                print(f"   ⚠️  Could not load model weights - using random initialization")
                print(f"   ⚠️  This will result in poor performance!")
                
            best_model = eval_model
            best_model.eval()
            print(f"   ✓ Model set to evaluation mode")
            
            # Evaluate on test set
            print(f"🧪 Running test set evaluation...")
            test_predictions = []
            test_actuals = []
            batch_count = 0
            
            print(f"   Creating test dataloader...")
            test_loader = newdata.test_dataloader()
            total_batches = len(test_loader)
            print(f"   ✓ Test loader ready: {total_batches} batches")
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    print(f"      Processing batch {batch_idx+1}/{total_batches}")
                    batch_count += 1
                    
                    try:
                        # Extract actual values
                        actual_batch = batch['delE_CISD'].cpu().numpy().flatten().tolist()
                        test_actuals.extend(actual_batch)
                        print(f"         ✓ Extracted {len(actual_batch)} actual values")
                        
                        # Make predictions
                        with torch.no_grad():
                            # Check if we have the wrapped model or direct model
                            if hasattr(best_model, 'pairwise_distance') and hasattr(best_model, 'model'):
                                # Using SimpleNNPot wrapper - apply full forward pass
                                result = best_model(batch)
                            elif hasattr(best_model, 'model'):
                                # Using some other wrapper - get inner model
                                # Apply pairwise distances manually
                                pairwise_distance = spk.atomistic.PairwiseDistances()
                                batch = pairwise_distance(batch)
                                result = best_model.model(batch)
                            else:
                                # Direct SimpleMolecularModel
                                pairwise_distance = spk.atomistic.PairwiseDistances()
                                batch = pairwise_distance(batch)
                                result = best_model(batch)
                                
                        predicted_batch = result['delE_CISD'].cpu().detach().numpy().flatten().tolist()
                        test_predictions.extend(predicted_batch)
                        print(f"         ✓ Generated {len(predicted_batch)} predictions")
                        
                    except Exception as pred_error:
                        print(f"         ❌ Prediction error in batch {batch_idx}: {pred_error}")
                        print(f"   Fallback: Using training loss as final score")
                        return min(train_loss, 1000.0)
            
            print(f"📊 Evaluation results:")
            print(f"   Batches processed: {batch_count}")
            print(f"   Total predictions: {len(test_predictions)}")
            print(f"   Total actuals: {len(test_actuals)}")
            
            # Calculate test performance
            if len(test_predictions) > 0 and len(test_actuals) > 0:
                test_rmse = np.sqrt(mean_squared_error(test_actuals, test_predictions))
                test_mae = mean_absolute_error(test_actuals, test_predictions)
                test_r2 = r2_score(test_actuals, test_predictions)
                
                print(f"📈 Test Performance Metrics:")
                print(f"   RMSE: {test_rmse:.6f}")
                print(f"   MAE: {test_mae:.6f}")
                print(f"   R²: {test_r2:.6f}")
                print(f"   Prediction range: [{min(test_predictions):.4f}, {max(test_predictions):.4f}]")
                print(f"   Actual range: [{min(test_actuals):.4f}, {max(test_actuals):.4f}]")
                
                final_score = test_rmse
                print(f"🎯 Final optimization score: {final_score:.6f}")
            else:
                print(f"❌ No test predictions/actuals generated")
                print(f"   Predictions: {len(test_predictions)}, Actuals: {len(test_actuals)}")
                final_score = min(train_loss, 1000.0)
                print(f"🎯 Fallback score: {final_score:.6f}")
                
        except Exception as model_error:
            print(f"❌ Model evaluation error: {model_error}")
            import traceback
            print(f"Evaluation error traceback:\n{traceback.format_exc()}")
            final_score = min(train_loss, 1000.0)
            print(f"🎯 Error fallback score: {final_score:.6f}")
        
        # Clean up trial-specific files
        print(f"\n🧹 CLEANUP: REMOVING TRIAL FILES")
        print(f"{'='*50}")
        
        cleanup_items = [
            ("Database file", db_path),
            ("Split file", split_file_full_path),
            ("Working directory", new_datatut),
            ("Temp XYZ directory", 'temp_xyz')
        ]
        
        for item_name, item_path in cleanup_items:
            try:
                if os.path.exists(item_path):
                    if os.path.isfile(item_path):
                        file_size = os.path.getsize(item_path) / 1024
                        os.remove(item_path)
                        print(f"   ✓ Removed {item_name}: {item_path} ({file_size:.2f} KB)")
                    elif os.path.isdir(item_path):
                        import shutil
                        shutil.rmtree(item_path)
                        print(f"   ✓ Removed {item_name}: {item_path}")
                else:
                    print(f"   - {item_name} not found: {item_path}")
            except Exception as cleanup_error:
                print(f"   ⚠️  Cleanup warning for {item_name}: {cleanup_error}")
        
        print(f"📊 TRIAL {trial.number} SUMMARY:")
        print(f"   Final Score: {final_score:.6f}")
        print(f"   Training Loss: {train_loss:.6f}")
        print(f"   Trial Duration: {pd.Timestamp.now() - pd.Timestamp.now()}")  # Will be updated
        print(f"✅ TRIAL {trial.number} COMPLETED SUCCESSFULLY")
        
        return final_score

    except Exception as e:
        import traceback
        debug_logger.error(f"\n💥 CRITICAL ERROR IN TRIAL {trial.number}")
        debug_logger.error(f"{'='*60}")
        debug_logger.error(f"❌ Error Type: {type(e).__name__}")
        debug_logger.error(f"❌ Error Message: {str(e)}")
        debug_logger.error(f"\n📋 FULL TRACEBACK:")
        debug_logger.error(f"{'-'*60}")
        debug_logger.error(traceback.format_exc())
        debug_logger.error(f"{'-'*60}")
        
        # Try to identify which step failed
        if 'data' not in locals():
            debug_logger.error(f"🔍 FAILURE POINT: Step 1 - Dataset loading")
        elif 'atoms_list' not in locals():
            debug_logger.error(f"🔍 FAILURE POINT: Step 2 - SMILES processing")
        elif 'atoms_objects' not in locals():
            debug_logger.error(f"🔍 FAILURE POINT: Step 3 - Atom objects creation")
        elif 'new_dataset' not in locals():
            debug_logger.error(f"🔍 FAILURE POINT: Step 4 - SchNet database creation")
        elif 'newdata' not in locals():
            debug_logger.error(f"🔍 FAILURE POINT: Step 6-7 - Data module creation/setup")
        elif 'task' not in locals():
            debug_logger.error(f"🔍 FAILURE POINT: Step 8 - Model creation")
        elif 'trainer' not in locals():
            debug_logger.error(f"🔍 FAILURE POINT: Step 9 - Trainer creation")
        else:
            debug_logger.error(f"🔍 FAILURE POINT: Step 10+ - Training or evaluation")
        
        # Emergency cleanup
        debug_logger.info(f"\n🚨 EMERGENCY CLEANUP")
        debug_logger.info(f"{'='*40}")
        emergency_paths = [
            f'./optuna_dataset_trial_{trial.number}.db',
            f'./optuna_dataset_trial_{trial.number}',
            'temp_xyz'
        ]
        
        for path in emergency_paths:
            try:
                if os.path.exists(path):
                    if os.path.isfile(path):
                        os.remove(path)
                        debug_logger.info(f"   ✓ Emergency removed file: {path}")
                    elif os.path.isdir(path):
                        import shutil
                        shutil.rmtree(path)
                        debug_logger.info(f"   ✓ Emergency removed directory: {path}")
            except Exception as emergency_error:
                debug_logger.warning(f"   ⚠️  Emergency cleanup failed for {path}: {emergency_error}")
        
        debug_logger.error(f"\n💔 TRIAL {trial.number} FAILED - RETURNING ERROR VALUE")
        return 1000.0

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
    fold_str = f"Trial - " if fold is not None else ""
    print(f"\n{fold_str}{dataset_name} Metrics:")
    print("-" * 50)
    print(f"R² Score: {r2:.4f}")
    print(f"Pearson r: {pearson_r:.4f} (p-value: {pearson_p:.2e})")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Max Absolute Error: {max_ae:.4f}")
    print(f"Relative MAE: {relative_mae:.4f}")
    
    return metrics_dict

def extract_molecular_features(atoms_objects, smiles_list=None):
    """Extract molecular features for interpretability analysis"""
    
    # Atomic weights dictionary
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
    
    # Extended atomic number mapping
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
    
    atomic_number_map_reverse = {v: k for k, v in atomic_number_map.items()}
    
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
            element_symbol = atomic_number_map_reverse.get(an, 'H')
            weight = atomic_weights.get(element_symbol, 1.0)
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

def create_prediction_plots(y_true, y_pred, dataset_name, fold=None, save_dir="plots"):
    """Create comprehensive prediction plots in the specified format"""
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    fold_str = f"_trial_{fold}" if fold is not None else ""
    
    # Create DataFrame for comparison
    comparison_df = pd.DataFrame({
        'Experimental': y_true,
        'Predicted (SchNet)': y_pred,
        'Predicted_LR': y_pred  # Linear regression line (will be fitted below)
    })
    
    # Fit linear regression for the regression line
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
    
    return {'r': corr_coeff, 'rmse': rmse, 'mae': mae}

# Create the Optuna study and optimize
if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Optuna hyperparameter optimization for SchNet delE_CISD prediction')
    parser.add_argument('--dataset', type=str, default='dataset.csv',
                       help='Path to dataset CSV file (default: dataset.csv)')
    parser.add_argument('--n_trials', type=int, default=100,
                       help='Number of Optuna trials to run (default: 100)')
    parser.add_argument('--debug', action='store_true',
                       help='Run single trial for debugging')
    
    args = parser.parse_args()
    
    print("="*80)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Number of trials: {args.n_trials}")
    print("Target: delE_CISD prediction using SchNet + External Descriptors")
    print(f"Log file: {log_file}")
    print("="*80)
    
    debug_logger.info("="*80)
    debug_logger.info("OPTUNA HYPERPARAMETER OPTIMIZATION STARTED")
    debug_logger.info("="*80)
    debug_logger.info(f"Dataset: {args.dataset}")
    debug_logger.info(f"Number of trials: {args.n_trials}")
    debug_logger.info("Target: delE_CISD prediction using SchNet + External Descriptors")
    debug_logger.info("="*80)
    
    # Check if dataset exists
    if not os.path.exists(args.dataset):
        print(f"ERROR: Dataset file '{args.dataset}' not found!")
        print("Please make sure the dataset file exists and try again.")
        sys.exit(1)
    
    # Test dataset loading
    try:
        test_data = pd.read_csv(args.dataset)
        print(f"Dataset validation: Found {len(test_data)} samples")
        required_columns = ['smiles', 's1_cis', 't1_cis', 'exchange_integral', 'dsp_scf', 'homo_lumo_overlap', 'delE_CISD']
        missing_columns = [col for col in required_columns if col not in test_data.columns]
        if missing_columns:
            print(f"ERROR: Missing required columns: {missing_columns}")
            sys.exit(1)
        print("Dataset validation: All required columns found")
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        sys.exit(1)
    
    # Debug mode - run single trial
    if args.debug:
        print("\n" + "="*50)
        print("DEBUG MODE - Running single trial")
        print("="*50)
        
        class DebugTrial:
            def __init__(self):
                self.number = 0
            
            def suggest_categorical(self, name, choices):
                return choices[0]  # Use first choice
            
            def suggest_int(self, name, low, high):
                return low  # Use minimum value
        
        debug_trial = DebugTrial()
        result = objective(debug_trial, args.dataset)
        print(f"\nDebug trial result: {result}")
        
        if result < 1000.0:
            print("✅ Debug trial successful! Proceeding with full optimization...")
        else:
            print("❌ Debug trial failed. Please check the error messages above.")
            sys.exit(1)
    
    # Create the objective function with dataset file path
    def objective_with_dataset(trial):
        return objective(trial, args.dataset)
    
    # Create study with better settings for Windows
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=5),
        storage=None  # Use in-memory storage to avoid file locking issues
    )  # 'minimize' because we want to minimize the test RMSE
    
    try:
        study.optimize(objective_with_dataset, n_trials=args.n_trials, timeout=3600)  # 1-hour timeout
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    except Exception as e:
        print(f"\nOptimization failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("="*80)
    print("OPTUNA OPTIMIZATION COMPLETED")
    print("="*80)
    print(f"Best parameters: {study.best_params}")
    print(f"Best trial value (test RMSE): {study.best_value:.4f}")
    print(f"Best trial: {study.best_trial}")
    
    # Save best parameters to JSON and TXT
    import json
    from datetime import datetime
    
    best_params = study.best_params.copy()
    best_params['best_rmse'] = study.best_value
    best_params['optimization_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    best_params['total_trials'] = len(study.trials)
    
    # Save to JSON (for programmatic use)
    with open('best_delE_CISD_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # Save to TXT (human-readable format)
    with open('best_delE_CISD_params.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("OPTUNA HYPERPARAMETER OPTIMIZATION RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Target: delE_CISD prediction using SchNet + External Descriptors\n")
        f.write(f"Optimization completed: {best_params['optimization_date']}\n")
        f.write(f"Total trials: {best_params['total_trials']}\n")
        f.write(f"Best test RMSE: {study.best_value:.4f} eV\n")
        f.write("\n" + "="*60 + "\n")
        f.write("BEST HYPERPARAMETERS:\n")
        f.write("="*60 + "\n")
        for param, value in study.best_params.items():
            f.write(f"{param:20}: {value}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("HOW TO USE THESE PARAMETERS:\n")
        f.write("="*60 + "\n")
        f.write("1. Copy the best_delE_CISD_params.json file to your working directory\n")
        f.write("2. Run: python schnet.py --use_optuna_params\n")
        f.write("   OR modify schnet.py to load these parameters automatically\n")
        f.write("\n")
        f.write("Parameter descriptions:\n")
        f.write("- bs: Batch size for training\n")
        f.write("- lr: Learning rate for AdamW optimizer\n")
        f.write("- patience: Early stopping patience\n")
        f.write("- n_rbf: Number of radial basis functions\n")
        f.write("- n_interactions: Number of SchNet interaction layers\n")
        f.write("- n_atom_basis: Atomic feature dimension\n")
        f.write("- cutoff: Cutoff distance for neighbor list (Angstrom)\n")
        f.write("- pat2: Learning rate scheduler patience\n")
    
    print(f"\nBest parameters saved to:")
    print(f"  - best_delE_CISD_params.json (for programmatic use)")
    print(f"  - best_delE_CISD_params.txt (human-readable)")
    print(f"\nRecommended settings for final training:")
    print("-" * 40)
    for param, value in study.best_params.items():
        print(f"{param:20}: {value}")
    
    print(f"\nTo use these parameters in schnet.py:")
    print("1. Run: python schnet.py --use_optuna_params")
    print("2. Or modify schnet.py to load parameters automatically")
    
    optuna.logging.set_verbosity(optuna.logging.INFO)
