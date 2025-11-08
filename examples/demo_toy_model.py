"""
Simple demonstration of PiSSA on a toy model.
"""
import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pissa import apply_pissa, PiSSAConfig


class ToyTransformer(nn.Module):
    """A simple transformer-like model for demonstration."""
    
    def __init__(self, vocab_size=1000, hidden_size=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Attention layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'q_proj': nn.Linear(hidden_size, hidden_size),
                'k_proj': nn.Linear(hidden_size, hidden_size),
                'v_proj': nn.Linear(hidden_size, hidden_size),
                'o_proj': nn.Linear(hidden_size, hidden_size),
            })
            for _ in range(num_layers)
        ])
        
        self.output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            # Simple attention (not real attention, just for demo)
            q = layer['q_proj'](x)
            k = layer['k_proj'](x)
            v = layer['v_proj'](x)
            x = layer['o_proj'](v)
        
        return self.output(x)


def count_parameters(model):
    """Count trainable parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def main():
    print("=" * 60)
    print("PiSSA Demonstration on Toy Transformer Model")
    print("=" * 60)
    
    # Create a toy model
    print("\n1. Creating toy transformer model...")
    model = ToyTransformer(vocab_size=1000, hidden_size=256, num_layers=2)
    
    # Count parameters before PiSSA
    train_before, total_before = count_parameters(model)
    print(f"   Total parameters: {total_before:,}")
    print(f"   Trainable parameters: {train_before:,}")
    
    # Apply PiSSA
    print("\n2. Applying PiSSA...")
    pissa_model = apply_pissa(
        model,
        r=8,
        target_modules=["q_proj", "v_proj"],  # Only adapt query and value projections
    )
    
    # Count parameters after PiSSA
    train_after, total_after = count_parameters(pissa_model.model)
    print(f"   Total parameters: {total_after:,}")
    print(f"   Trainable parameters: {train_after:,}")
    print(f"   Reduction: {100 * (1 - train_after / train_before):.2f}%")
    
    # Show detailed breakdown
    print("\n3. Trainable parameters breakdown:")
    pissa_model.print_trainable_parameters()
    
    # Test forward pass
    print("\n4. Testing forward pass...")
    batch_size = 4
    seq_length = 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    
    with torch.no_grad():
        output = pissa_model(input_ids)
    
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Output shape: {output.shape}")
    print("   ✓ Forward pass successful!")
    
    # Test saving and loading
    print("\n5. Testing save/load...")
    save_path = "/tmp/pissa_demo.pt"
    pissa_model.save_pretrained(save_path)
    
    # Load into the same model (just the PiSSA parameters)
    print("   Loading PiSSA parameters back into same model...")
    pissa_model.load_pretrained(save_path)
    print("   ✓ Save/load successful!")
    
    # Verify outputs match
    with torch.no_grad():
        output_after_load = pissa_model(input_ids)
    
    max_diff = torch.max(torch.abs(output - output_after_load)).item()
    print(f"   Max difference after load: {max_diff:.2e}")
    assert max_diff < 1e-5, "Outputs should match exactly after loading!"
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
    
    # Clean up
    os.remove(save_path)


if __name__ == "__main__":
    main()
