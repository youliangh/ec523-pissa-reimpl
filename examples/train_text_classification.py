"""
Example: Fine-tuning GPT-2 with PiSSA on a text classification task.
"""
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
import sys
import os
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pissa import apply_pissa


def prepare_data(tokenizer, batch_size=16, max_length=128):
    """Prepare IMDB dataset for training."""
    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb", split="train[:1000]")  # Use small subset for demo
    test_dataset = load_dataset("imdb", split="test[:200]")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    train_loader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(tokenized_test, batch_size=batch_size)
    
    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
    
    return total_loss / len(train_loader)


def evaluate(model, test_loader, device):
    """Evaluate the model."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    return accuracy


def main():
    # Configuration
    model_name = "distilbert-base-uncased"  # Using smaller model for demo
    r = 8
    target_modules = ["q_lin", "v_lin"]  # DistilBERT attention modules
    num_epochs = 3
    learning_rate = 3e-4
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )
    
    # Apply PiSSA
    print(f"\nApplying PiSSA with r={r}")
    pissa_model = apply_pissa(
        model,
        r=r,
        target_modules=target_modules,
    )
    pissa_model.print_trainable_parameters()
    pissa_model = pissa_model.to(device)
    
    # Prepare data
    train_loader, test_loader = prepare_data(tokenizer, batch_size=batch_size)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(pissa_model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )
    
    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        avg_loss = train_epoch(pissa_model, train_loader, optimizer, scheduler, device)
        print(f"Average loss: {avg_loss:.4f}")
        
        accuracy = evaluate(pissa_model, test_loader, device)
        print(f"Validation accuracy: {accuracy:.4f}")
    
    # Save model
    output_path = "pissa_distilbert_imdb.pt"
    pissa_model.save_pretrained(output_path)
    print(f"\nModel saved to {output_path}")


if __name__ == "__main__":
    main()
