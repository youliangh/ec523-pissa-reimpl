"""
Configuration class for PiSSA.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PiSSAConfig:
    """
    Configuration class for PiSSA (Principal Singular values and Singular vectors Adaptation).
    
    Args:
        r (int): Rank of the adaptation. Number of principal components to train.
        target_modules (List[str]): List of module names to apply PiSSA to.
        modules_to_save (Optional[List[str]]): List of module names to save.
        init_weights (bool): Whether to initialize weights from pretrained model.
        fan_in_fan_out (bool): Set to True for Conv1D layers.
        bias (str): Bias type. Can be 'none', 'all' or 'pissa_only'.
    """
    r: int = 8
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    modules_to_save: Optional[List[str]] = None
    init_weights: bool = True
    fan_in_fan_out: bool = False
    bias: str = "none"
    
    def __post_init__(self):
        if self.bias not in ["none", "all", "pissa_only"]:
            raise ValueError(f"bias must be 'none', 'all', or 'pissa_only', got {self.bias}")
