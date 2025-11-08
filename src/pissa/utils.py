"""
Utility functions for PiSSA.
"""
import torch
import torch.nn as nn
from typing import Dict, List


def get_pissa_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract PiSSA parameters from a model.
    
    Args:
        model: Model with PiSSA layers
        
    Returns:
        Dictionary of PiSSA parameters
    """
    from .pissa_layer import PiSSALayer
    
    state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, PiSSALayer):
            state_dict[f"{name}.pissa_A"] = module.pissa_A.data
            state_dict[f"{name}.pissa_B"] = module.pissa_B.data
            if module.bias is not None:
                state_dict[f"{name}.bias"] = module.bias.data
    
    return state_dict


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with 'trainable' and 'total' counts
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    return {
        'trainable': trainable,
        'total': total,
        'percentage': 100 * trainable / total if total > 0 else 0,
    }


def freeze_non_pissa_parameters(model: nn.Module):
    """
    Freeze all parameters except PiSSA parameters.
    
    Args:
        model: Model with PiSSA layers
    """
    from .pissa_layer import PiSSALayer
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze PiSSA parameters
    for module in model.modules():
        if isinstance(module, PiSSALayer):
            module.pissa_A.requires_grad = True
            module.pissa_B.requires_grad = True
            if module.bias is not None:
                module.bias.requires_grad = True


def get_target_modules_for_model(model_name: str) -> List[str]:
    """
    Get recommended target modules for common model architectures.
    
    Args:
        model_name: Name of the model (e.g., 'gpt2', 'bert', 'llama')
        
    Returns:
        List of target module names
    """
    model_name = model_name.lower()
    
    if 'gpt2' in model_name or 'gpt-2' in model_name:
        return ['c_attn', 'c_proj']
    elif 'bert' in model_name or 'distilbert' in model_name:
        return ['query', 'value']
    elif 'llama' in model_name:
        return ['q_proj', 'v_proj']
    elif 't5' in model_name:
        return ['q', 'v']
    else:
        # Default for most transformer models
        return ['q_proj', 'v_proj']
