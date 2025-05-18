import torch
import torch.nn as nn
import wandb

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, 
                device=torch.device("cpu"), wandb_logging=False, teacher_forcing_ratio=None):
    """
    Train the Seq2Seq model
    
    Args:
        model: The Seq2Seq model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to use for training
        wandb_logging: Whether to log metrics to wandb
        teacher_forcing_ratio: Override the model's default teacher forcing ratio
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is padding index
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct_predictions = 0
        train_total_predictions = 0
        
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            
            # Prepare target for input (everything but last token) and output (everything but first token)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass with specified teacher forcing ratio
            output = model(src, tgt_input, teacher_forcing_ratio)
            
            # Calculate accuracy metrics before reshaping
            predictions = output.argmax(dim=2)
            non_pad_mask = tgt_output != 0  # Assuming 0 is padding index
            correct = (predictions == tgt_output) & non_pad_mask
            train_correct_predictions += correct.sum().item()
            train_total_predictions += non_pad_mask.sum().item()
            
            # Reshape for loss calculation
            output_flat = output.reshape(-1, output.shape[-1])
            tgt_output_flat = tgt_output.reshape(-1)
            
            # Calculate loss
            loss = criterion(output_flat, tgt_output_flat)
            
            # Backpropagation
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update parameters
            optimizer.step()
            
            # Accumulate loss
            train_loss += loss.item()
        
        # Calculate average loss and accuracy for the epoch
        train_loss /= len(train_loader)
        train_accuracy = train_correct_predictions / train_total_predictions if train_total_predictions > 0 else 0
        
        # Validation
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        
        # Print epoch results
        print(f'Epoch: {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        # Log to wandb
        if wandb_logging:
            wandb.log({
                "epoch": epoch+1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            })
    
    return model


def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate the model on the validation set
    """
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for src, tgt in data_loader:
            src, tgt = src.to(device), tgt.to(device)
            
            # Prepare target for input and output
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Forward pass without teacher forcing
            output = model(src, tgt_input, teacher_forcing_ratio=0.0)
            
            # Reshape for loss calculation
            output_flat = output.reshape(-1, output.shape[-1])
            tgt_output_flat = tgt_output.reshape(-1)
            
            # Calculate loss
            loss = criterion(output_flat, tgt_output_flat)
            total_loss += loss.item()
            
            # Calculate accuracy (ignoring padding tokens)
            predictions = output.argmax(dim=2)
            non_pad_mask = tgt_output != 0  # Assuming 0 is padding index
            correct = (predictions == tgt_output) & non_pad_mask
            correct_predictions += correct.sum().item()
            total_predictions += non_pad_mask.sum().item()
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    return avg_loss, accuracy

