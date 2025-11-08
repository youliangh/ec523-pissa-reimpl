"""
PiSSA Layer implementation.
"""
import torch
import torch.nn as nn
import math


class PiSSALayer(nn.Module):
    """
    PiSSA (Principal Singular values and Singular vectors Adaptation) Layer.
    
    This layer performs SVD on a pretrained weight matrix W and decomposes it into:
    W = U @ S @ V^T
    
    Then splits into principal and residual components:
    W = (U_r @ S_r @ V_r^T) + (U_res @ S_res @ V_res^T)
    
    During fine-tuning, only the principal component (U_r @ S_r @ V_r^T) is trainable.
    
    Args:
        in_features (int): Input dimension
        out_features (int): Output dimension
        r (int): Rank (number of principal components)
        bias (bool): Whether to include bias
        device: Device to place the layer on
        dtype: Data type for the layer
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        
        # Principal components (trainable)
        self.pissa_A = nn.Parameter(torch.zeros((r, in_features), device=device, dtype=dtype))
        self.pissa_B = nn.Parameter(torch.zeros((out_features, r), device=device, dtype=dtype))
        
        # Residual components (frozen)
        self.register_buffer(
            'residual_W',
            torch.zeros((out_features, in_features), device=device, dtype=dtype)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        # Initialize principal components with small random values
        nn.init.kaiming_uniform_(self.pissa_A, a=math.sqrt(5))
        nn.init.zeros_(self.pissa_B)
        
    def initialize_from_weight(self, weight: torch.Tensor):
        """
        Initialize PiSSA layer from a pretrained weight matrix using SVD.
        
        Args:
            weight: Pretrained weight matrix of shape (out_features, in_features)
        """
        with torch.no_grad():
            # Perform SVD on the pretrained weight
            U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
            
            # Extract principal components
            U_r = U[:, :self.r]
            S_r = S[:self.r]
            Vh_r = Vh[:self.r, :]
            
            # Extract residual components
            U_res = U[:, self.r:]
            S_res = S[self.r:]
            Vh_res = Vh[self.r:, :]
            
            # Initialize trainable parameters (principal components)
            # pissa_B @ pissa_A = U_r @ S_r @ Vh_r
            # We can distribute S_r in different ways. Here we use:
            # pissa_B = U_r @ sqrt(S_r), pissa_A = sqrt(S_r) @ Vh_r
            sqrt_S_r = torch.sqrt(S_r)
            self.pissa_B.data = (U_r @ torch.diag(sqrt_S_r)).to(weight.dtype)
            self.pissa_A.data = (torch.diag(sqrt_S_r) @ Vh_r).to(weight.dtype)
            
            # Store residual (frozen) components
            residual = U_res @ torch.diag(S_res) @ Vh_res
            self.residual_W.data = residual.to(weight.dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        # Compute output from trainable principal components
        principal_out = torch.matmul(x, self.pissa_A.t())
        principal_out = torch.matmul(principal_out, self.pissa_B.t())
        
        # Add residual (frozen) component
        residual_out = torch.matmul(x, self.residual_W.t())
        
        # Combine principal and residual
        result = principal_out + residual_out
        
        if self.bias is not None:
            result = result + self.bias
            
        return result
    
    def extra_repr(self) -> str:
        """String representation."""
        return f'in_features={self.in_features}, out_features={self.out_features}, r={self.r}, bias={self.bias is not None}'
