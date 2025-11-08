"""
PiSSA Model wrapper for applying PiSSA to transformer models.
"""
import torch
import torch.nn as nn
from typing import List, Optional, Dict
import re

from .pissa_layer import PiSSALayer
from .config import PiSSAConfig


class PiSSAModel(nn.Module):
    """
    Wrapper class for applying PiSSA to a pretrained model.
    
    Args:
        model: Base pretrained model
        config: PiSSA configuration
    """
    
    def __init__(self, model: nn.Module, config: PiSSAConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.pissa_layers = {}
        
        # Apply PiSSA to target modules
        self._apply_pissa()
        
        # Mark PiSSA parameters as trainable
        self._mark_only_pissa_as_trainable()
    
    def _apply_pissa(self):
        """Apply PiSSA to target modules in the model."""
        for name, module in self.model.named_modules():
            if self._is_target_module(name):
                self._replace_with_pissa(name, module)
    
    def _is_target_module(self, name: str) -> bool:
        """Check if a module name matches target modules."""
        if not isinstance(self.config.target_modules, list):
            return False
        
        for target in self.config.target_modules:
            # Support both exact match and pattern matching
            if target in name or re.search(target, name):
                return True
        return False
    
    def _replace_with_pissa(self, name: str, module: nn.Module):
        """Replace a linear layer with PiSSA layer."""
        if not isinstance(module, nn.Linear):
            return
        
        # Get parent module
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        
        if parent_name:
            parent = self.model.get_submodule(parent_name)
        else:
            parent = self.model
        
        # Create PiSSA layer
        pissa_layer = PiSSALayer(
            in_features=module.in_features,
            out_features=module.out_features,
            r=self.config.r,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        
        # Initialize from pretrained weights
        if self.config.init_weights:
            pissa_layer.initialize_from_weight(module.weight.data)
            if module.bias is not None:
                pissa_layer.bias.data = module.bias.data.clone()
        
        # Replace module
        setattr(parent, child_name, pissa_layer)
        self.pissa_layers[name] = pissa_layer
    
    def _mark_only_pissa_as_trainable(self):
        """Mark only PiSSA parameters as trainable."""
        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze PiSSA parameters
        for name, module in self.model.named_modules():
            if isinstance(module, PiSSALayer):
                module.pissa_A.requires_grad = True
                module.pissa_B.requires_grad = True
                if module.bias is not None and self.config.bias != "none":
                    module.bias.requires_grad = True
    
    def forward(self, *args, **kwargs):
        """Forward pass through the model."""
        return self.model(*args, **kwargs)
    
    def print_trainable_parameters(self):
        """Print the number of trainable parameters."""
        trainable_params = 0
        all_params = 0
        
        for param in self.model.parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_params:,} || "
            f"trainable%: {100 * trainable_params / all_params:.2f}%"
        )
    
    def save_pretrained(self, path: str):
        """Save PiSSA parameters."""
        pissa_state_dict = {}
        for name, module in self.model.named_modules():
            if isinstance(module, PiSSALayer):
                pissa_state_dict[f"{name}.pissa_A"] = module.pissa_A.data
                pissa_state_dict[f"{name}.pissa_B"] = module.pissa_B.data
                if module.bias is not None:
                    pissa_state_dict[f"{name}.bias"] = module.bias.data
        
        torch.save(pissa_state_dict, path)
        print(f"PiSSA parameters saved to {path}")
    
    def load_pretrained(self, path: str):
        """Load PiSSA parameters."""
        pissa_state_dict = torch.load(path)
        
        for name, param in pissa_state_dict.items():
            module_name = '.'.join(name.split('.')[:-1])
            param_name = name.split('.')[-1]
            
            module = self.model.get_submodule(module_name)
            if isinstance(module, PiSSALayer):
                if param_name == "pissa_A":
                    module.pissa_A.data = param
                elif param_name == "pissa_B":
                    module.pissa_B.data = param
                elif param_name == "bias":
                    module.bias.data = param
        
        print(f"PiSSA parameters loaded from {path}")


def apply_pissa(
    model: nn.Module,
    r: int = 8,
    target_modules: Optional[List[str]] = None,
    **kwargs
) -> PiSSAModel:
    """
    Apply PiSSA to a pretrained model.
    
    Args:
        model: Base pretrained model
        r: Rank of adaptation
        target_modules: List of module names to apply PiSSA to
        **kwargs: Additional arguments for PiSSAConfig
        
    Returns:
        PiSSAModel wrapper
        
    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> pissa_model = apply_pissa(
        ...     model,
        ...     r=8,
        ...     target_modules=["c_attn", "c_proj"]
        ... )
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]
    
    config = PiSSAConfig(r=r, target_modules=target_modules, **kwargs)
    return PiSSAModel(model, config)
