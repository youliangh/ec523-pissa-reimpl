# PiSSA Re-implementation

**BU EC523 Deep Learning Final Project**

A PyTorch re-implementation of PiSSA (Principal Singular values and Singular vectors Adaptation), a parameter-efficient fine-tuning method for large language models.

## Overview

PiSSA is a novel parameter-efficient fine-tuning technique that leverages Singular Value Decomposition (SVD) to decompose pretrained weight matrices into principal and residual components. During fine-tuning, only the principal components (containing the most significant singular values and vectors) are trained, while the residual components remain frozen.

### Key Features

- **Parameter Efficient**: Significantly reduces the number of trainable parameters compared to full fine-tuning
- **SVD-based Initialization**: Intelligently initializes trainable parameters using SVD of pretrained weights
- **Easy Integration**: Simple API compatible with HuggingFace transformers
- **Flexible Configuration**: Customizable rank and target modules

## Installation

```bash
# Clone the repository
git clone https://github.com/youliangh/ec523-pissa-reimpl.git
cd ec523-pissa-reimpl

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from transformers import AutoModelForCausalLM
from pissa import apply_pissa

# Load a pretrained model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Apply PiSSA
pissa_model = apply_pissa(
    model,
    r=8,  # Rank of adaptation
    target_modules=["c_attn", "c_proj"]  # Modules to adapt
)

# Check trainable parameters
pissa_model.print_trainable_parameters()

# Use the model for training
# ... your training code ...
```

### Configuration Options

```python
from pissa import PiSSAConfig, PiSSAModel

config = PiSSAConfig(
    r=8,  # Rank (number of principal components)
    target_modules=["q_proj", "v_proj"],  # Target modules
    init_weights=True,  # Initialize from pretrained weights
    bias="none",  # Bias handling: 'none', 'all', or 'pissa_only'
)

pissa_model = PiSSAModel(model, config)
```

## How PiSSA Works

1. **SVD Decomposition**: For each target weight matrix W, perform SVD:
   ```
   W = U @ S @ V^T
   ```

2. **Split into Principal and Residual**:
   ```
   W = (U_r @ S_r @ V_r^T) + (U_res @ S_res @ V_res^T)
   ```
   where `r` is the rank parameter.

3. **Fine-tuning**: Only the principal component is trainable:
   ```
   W_adapted = (U_r @ S_r @ V_r^T)_trainable + (U_res @ S_res @ V_res^T)_frozen
   ```

## Examples

### Text Classification

See `examples/train_text_classification.py` for a complete example of fine-tuning a model on IMDB sentiment classification:

```bash
python examples/train_text_classification.py
```

### Custom Training Loop

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pissa import apply_pissa

# Setup
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
pissa_model = apply_pissa(model, r=8, target_modules=["c_attn", "c_proj"])

# Training
optimizer = torch.optim.AdamW(pissa_model.parameters(), lr=3e-4)

for epoch in range(num_epochs):
    for batch in dataloader:
        outputs = pissa_model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Save adapted parameters
pissa_model.save_pretrained("path/to/save")

# Load adapted parameters
pissa_model.load_pretrained("path/to/save")
```

## Testing

Run the test suite:

```bash
python tests/test_pissa_layer.py
```

## Project Structure

```
ec523-pissa-reimpl/
├── src/
│   └── pissa/
│       ├── __init__.py          # Package initialization
│       ├── config.py            # Configuration classes
│       ├── pissa_layer.py       # Core PiSSA layer implementation
│       └── pissa_model.py       # Model wrapper and utilities
├── tests/
│   └── test_pissa_layer.py      # Unit tests
├── examples/
│   └── train_text_classification.py  # Example training script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Advantages over Standard Fine-tuning

- **Reduced Parameters**: Typically 0.1-1% of total parameters are trainable
- **Faster Training**: Fewer parameters to update
- **Lower Memory**: Reduced memory footprint during training
- **Better Initialization**: SVD provides principled initialization

## Comparison with LoRA

While LoRA adds low-rank adapters to weight matrices, PiSSA decomposes existing weights:

- **LoRA**: `W_adapted = W_frozen + A @ B`
- **PiSSA**: `W_adapted = (U_r @ S_r @ V_r^T)_trainable + W_residual_frozen`

PiSSA leverages the information in pretrained weights more directly through SVD decomposition.

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- See `requirements.txt` for full list

## Citation

This is a re-implementation for educational purposes (BU EC523 Deep Learning course). For the original PiSSA paper, please refer to the academic literature.

## License

MIT License

## Authors

- BU EC523 Deep Learning Student Project

## Acknowledgments

- HuggingFace Transformers library
- Original PiSSA research paper
- BU EC523 Deep Learning course staff