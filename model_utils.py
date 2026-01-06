"""
Utility functions for RetNet model operations.
Handles checkpoint loading, saving, and model information extraction.
"""

import os
from typing import Tuple, Dict, Any, Optional
import torch
from retnet_model import RetNet, RetNetConfig


def load_checkpoint(checkpoint_path: str, device: str = 'cpu') -> Tuple[RetNet, RetNetConfig, Dict[str, Any]]:
    """
    Load a RetNet model from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        device: Device to load the model on ('cpu' or 'cuda')
    
    Returns:
        Tuple of (model, config, checkpoint_info)
        checkpoint_info contains: global_step, best_val_loss, etc.
    
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint loading fails
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    try:
        # Load checkpoint with weights_only=False to handle custom classes
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Extract configuration
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Fallback to default config if not saved in checkpoint
            config = RetNetConfig()
        
        # Initialize model
        model = RetNet(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # Ensure float32 precision for CPU execution
        if device == 'cpu':
            model.float()
            
        model.eval()
        
        # Extract checkpoint metadata
        checkpoint_info = {
            'global_step': checkpoint.get('global_step', 0),
            'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
            'file_size_mb': os.path.getsize(checkpoint_path) / (1024 * 1024),
            'checkpoint_path': checkpoint_path
        }
        
        return model, config, checkpoint_info
    
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {str(e)}")


def initialize_model(config: RetNetConfig, device: str = 'cpu') -> RetNet:
    """
    Initialize a new RetNet model with the given configuration.
    
    Args:
        config: RetNetConfig object with model hyperparameters
        device: Device to create the model on
    
    Returns:
        Initialized RetNet model
    """
    model = RetNet(config)
    model.to(device)
    return model


def get_model_info(model: RetNet, config: RetNetConfig) -> Dict[str, Any]:
    """
    Extract detailed information about the model.
    
    Args:
        model: RetNet model instance
        config: Model configuration
    
    Returns:
        Dictionary with model statistics and architecture details
    """
    num_params = model.get_num_params()
    
    # Calculate model size in MB (assuming float32)
    model_size_mb = (num_params * 4) / (1024 * 1024)
    
    info = {
        'total_parameters': num_params,
        'total_parameters_millions': num_params / 1e6,
        'model_size_mb': model_size_mb,
        'architecture': {
            'vocab_size': config.vocab_size,
            'd_model': config.d_model,
            'n_layers': config.n_layers,
            'n_heads': config.n_heads,
            'head_dim_qk': config.head_dim_qk,
            'head_dim_v': config.head_dim_v,
            'ffn_dim': config.ffn_dim,
            'max_seq_len': config.max_seq_len,
            'dropout': config.dropout
        }
    }
    
    return info


def save_checkpoint(model: RetNet, optimizer: Optional[torch.optim.Optimizer], 
                   scheduler: Optional[Any], global_step: int, 
                   best_val_loss: float, config: RetNetConfig, 
                   save_path: str) -> None:
    """
    Save model checkpoint to disk.
    
    Args:
        model: RetNet model to save
        optimizer: Optimizer state (optional)
        scheduler: Learning rate scheduler (optional)
        global_step: Current training step
        best_val_loss: Best validation loss achieved
        config: Model configuration
        save_path: Path to save the checkpoint
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'global_step': global_step,
        'best_val_loss': best_val_loss
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save(checkpoint, save_path)


def get_device() -> str:
    """
    Get the best available device (CUDA if available, else CPU).
    
    Returns:
        Device string ('cuda' or 'cpu')
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available compute devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device': get_device()
    }
    
    if torch.cuda.is_available():
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        info['cuda_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return info
