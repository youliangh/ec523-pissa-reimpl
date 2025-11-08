# PiSSA Quick Start Guide

This guide demonstrates how to use PiSSA for parameter-efficient fine-tuning.

## Installation

```bash
pip install -r requirements.txt
```

## Basic Example

### 1. Import Libraries

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pissa import apply_pissa
```

### 2. Load Pretrained Model

```python
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

### 3. Apply PiSSA

```python
# Apply PiSSA with rank 8 to attention layers
pissa_model = apply_pissa(
    model,
    r=8,  # rank of adaptation
    target_modules=["c_attn", "c_proj"]  # GPT-2 attention modules
)

# Check parameter reduction
pissa_model.print_trainable_parameters()
# Output: trainable params: ~300K || all params: ~124M || trainable%: 0.24%
```

### 4. Fine-tune the Model

```python
# Standard PyTorch training loop
optimizer = torch.optim.AdamW(pissa_model.parameters(), lr=3e-4)

for epoch in range(num_epochs):
    for batch in dataloader:
        outputs = pissa_model(**batch)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### 5. Save and Load

```python
# Save only the PiSSA parameters (very small file)
pissa_model.save_pretrained("./pissa_weights.pt")

# Load PiSSA parameters
pissa_model.load_pretrained("./pissa_weights.pt")
```

## Configuration Options

### Rank Selection

The rank `r` determines the number of principal components to train:
- Lower rank (r=4-8): Fewer parameters, faster training, might lose capacity
- Higher rank (r=16-32): More parameters, better capacity, slower training

```python
# Low rank - very parameter efficient
pissa_model = apply_pissa(model, r=4, target_modules=["c_attn"])

# Higher rank - better capacity
pissa_model = apply_pissa(model, r=32, target_modules=["c_attn"])
```

### Target Modules

Choose which layers to adapt:

```python
# Only query and value projections (common choice)
pissa_model = apply_pissa(model, r=8, target_modules=["q_proj", "v_proj"])

# All attention projections
pissa_model = apply_pissa(model, r=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

# Attention and MLP layers
pissa_model = apply_pissa(model, r=8, target_modules=["q_proj", "v_proj", "mlp"])
```

### Model-Specific Configurations

```python
from pissa import get_target_modules_for_model

# Get recommended modules for your model
target_modules = get_target_modules_for_model("gpt2")  # Returns ["c_attn", "c_proj"]
target_modules = get_target_modules_for_model("llama")  # Returns ["q_proj", "v_proj"]
target_modules = get_target_modules_for_model("bert")  # Returns ["query", "value"]

pissa_model = apply_pissa(model, r=8, target_modules=target_modules)
```

## Advanced Usage

### Custom Configuration

```python
from pissa import PiSSAConfig, PiSSAModel

config = PiSSAConfig(
    r=16,
    target_modules=["q_proj", "v_proj", "k_proj"],
    init_weights=True,  # Initialize from pretrained weights (recommended)
    bias="none",  # Options: "none", "all", "pissa_only"
)

pissa_model = PiSSAModel(model, config)
```

### Parameter Statistics

```python
from pissa import count_parameters

stats = count_parameters(pissa_model.model)
print(f"Trainable: {stats['trainable']:,}")
print(f"Total: {stats['total']:,}")
print(f"Percentage: {stats['percentage']:.2f}%")
```

### Manual Parameter Freezing

```python
from pissa import freeze_non_pissa_parameters

# If you need to manually freeze parameters
freeze_non_pissa_parameters(model)
```

## Complete Example: Text Classification

See `examples/train_text_classification.py` for a complete working example.

## Tips and Best Practices

1. **Start with r=8**: This is a good default that works for most tasks
2. **Target q_proj and v_proj**: These typically give the best results
3. **Use learning rate 3e-4**: Higher than full fine-tuning (typically 1e-5)
4. **Initialize from pretrained**: Always set `init_weights=True` for best results
5. **Monitor performance**: If results are poor, try increasing r or adding more target modules

## Comparison with Full Fine-tuning

| Method | Trainable Params | Memory | Training Speed |
|--------|------------------|--------|----------------|
| Full Fine-tuning | 100% | High | Slow |
| PiSSA (r=8) | ~0.1-1% | Low | Fast |
| PiSSA (r=16) | ~0.2-2% | Medium | Medium |

## Troubleshooting

### Issue: Poor performance after fine-tuning

**Solutions:**
- Increase rank `r`
- Add more target modules
- Train for more epochs
- Increase learning rate

### Issue: Out of memory

**Solutions:**
- Decrease rank `r`
- Reduce batch size
- Use gradient accumulation

### Issue: Model not learning

**Solutions:**
- Check that parameters are trainable: `pissa_model.print_trainable_parameters()`
- Verify learning rate is appropriate (try 1e-4 to 5e-4)
- Ensure target modules match your model architecture

## References

For more details on PiSSA, see the academic literature on parameter-efficient fine-tuning methods.
