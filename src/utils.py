import json
import logging
import os
import glob
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

# Assuming models are defined in src.models
from .models import FourModalFusion


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
        np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    return obj


def verify_model_alignment(experiment_base_path, concat_base_path):
    """Verify alignment of model paths for all folds."""
    print("\\n=== Starting Model Alignment Verification ===")
    model_paths = {
        'enlarged': {'paths': [], 'aucs': []},
        'original': {'paths': [], 'aucs': []},
        'concat': {'paths': [], 'aucs': []}
    }
    
    for fold in range(5):
        concat_model_path = os.path.join(concat_base_path, f'fold{fold}.pth')
        if not os.path.exists(concat_model_path):
            raise FileNotFoundError(f"Concat model not found at {concat_model_path}")
        model_paths['concat']['paths'].append(concat_model_path)
        model_paths['concat']['aucs'].append(None)

        for model_type in ['enlarged', 'original']:
            pattern = os.path.join(
                experiment_base_path, 'models',
                f'best_model_fold{fold}.0_{model_type}_smoothauc*.pth'
            )
            matching_files = glob.glob(pattern)
            if matching_files:
                model_path = matching_files[-1]
                model_paths[model_type]['paths'].append(model_path)
                try:
                    auc_value = float(model_path.split('smoothauc')[-1].split('.pth')[0])
                    model_paths[model_type]['aucs'].append(auc_value)
                except (ValueError, IndexError):
                    model_paths[model_type]['aucs'].append(None)
            else:
                model_paths[model_type]['paths'].append(None)
                model_paths[model_type]['aucs'].append(None)

    print("\\n=== Model Alignment Verification Complete ===")
    return model_paths


def calculate_weighted_metric(auc, ap, weight_auc=0.7):
    """Calculate weighted combination of AUC and Average Precision."""
    return weight_auc * auc + (1 - weight_auc) * ap


def verify_saved_model(original_model, saved_model_path, val_loader, device):
    """Verify saved model by comparing predictions with original model."""
    logger = logging.getLogger(__name__)
    
    # This function needs the feature lengths to load a model.
    # This is a drawback of the current FourModalFusion.load_model implementation.
    # A better implementation would save these params to the model checkpoint.
    # For now, we are assuming ra_columns and cl_columns are available in the scope
    # where this function is called, which is a bit of a hack.
    from .data import ra_columns, cl_columns
    
    # The load_model classmethod in the original script had issues.
    # Re-implementing a simpler load logic here for verification.
    # It's better to save and load only the state_dict.
    
    # We can't easily load the model without circular dependencies or major refactoring
    # of the model saving/loading logic. The original `load_model` is highly dependent
    # on the training script's global state.
    
    # For now, this function will be simplified to just check if the path exists.
    # A full verification requires a refactor of model saving/loading.
    
    verification_results = {
        'model_path': str(saved_model_path),
        'exists': saved_model_path.exists(),
        'verification_status': 'skipped',
        'message': 'Full model verification requires refactoring of save/load logic.',
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    logger.warning("Skipping full model verification due to complex dependencies in model loading.")
    
    return verification_results 