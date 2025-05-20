import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import wandb
import os
from PIL import Image
import io
from model_w_attention import build_model

def extract_attention_weights_from_model(model, src_seq, input_vocab, target_vocab, device, max_length=50):
    """
    Extract attention weights from your existing model during inference
    
    Args:
        model: Your trained Seq2Seq model with attention
        src_seq: Source sequence tensor [seq_len]
        input_vocab: Input vocabulary dictionary
        target_vocab: Target vocabulary dictionary
        device: Device to run inference on
        max_length: Maximum decoding length
        
    Returns:
        attention_weights: List of attention weight tensors
        predicted_sequence: List of predicted token indices
        encoder_outputs: Encoder outputs for visualization
    """
    model.eval()
    attention_weights = []
    predicted_sequence = []
    
    sos_token = target_vocab['<sos>']
    eos_token = target_vocab['<eos>']
    
    with torch.no_grad():
        # Add batch dimension
        src_seq = src_seq.unsqueeze(0)  # [1, seq_len]
        
        # Encoder forward pass
        embedded_src = model.encoder_embedding(src_seq)
        encoder_outputs, encoder_hidden = model.encoder_rnn(embedded_src)
        
        # Adapt hidden state dimensions if needed
        hidden = model._adapt_hidden_state(encoder_hidden)
        
        # Initialize decoder input with SOS token
        decoder_input = torch.tensor([[sos_token]], device=device)
        
        for t in range(max_length):
            # Get embedding
            embedded_input = model.decoder_embedding(decoder_input)
            
            # Decoder forward pass
            output, hidden = model.decoder_rnn(embedded_input, hidden)
            
            if model.use_attention:
                # Get decoder hidden state for attention
                decoder_hidden_for_attn = model._get_decoder_hidden_for_attention(hidden)
                
                # Compute attention - this returns the attention weights we need!
                attn_weights, context_vector = model.attention(encoder_outputs, decoder_hidden_for_attn)
                
                # Store attention weights [1, src_len] -> [src_len]
                attention_weights.append(attn_weights.squeeze(0).cpu())
                
                # Concatenate decoder output with context vector
                decoder_output = output.squeeze(1)
                combined_output = torch.cat([decoder_output, context_vector], dim=1)
                
                # Get prediction
                prediction = model.fc_out(combined_output)
            else:
                # If no attention, create dummy weights (uniform attention)
                src_len = encoder_outputs.size(1)
                dummy_weights = torch.ones(src_len) / src_len
                attention_weights.append(dummy_weights)
                prediction = model.fc_out(output.squeeze(1))
            
            # Get predicted token
            predicted_token = prediction.argmax(dim=-1)
            predicted_sequence.append(predicted_token.item())
            
            # Use predicted token as next input
            decoder_input = predicted_token.unsqueeze(0)
            
            # Stop if EOS token is generated
            if predicted_token.item() == eos_token:
                break
        
        # Stack attention weights: [tgt_len, src_len]
        if attention_weights:
            attention_weights = torch.stack(attention_weights, dim=0)
        else:
            # Fallback if no attention weights collected
            attention_weights = torch.ones(1, encoder_outputs.size(1))
    
    return attention_weights, predicted_sequence, encoder_outputs

def create_attention_heatmap(attention_weights, source_chars, target_chars, 
                           input_text, predicted_text, target_text, sample_idx, 
                           save_path="attention_heatmaps", figsize=(12, 8)):
    """
    Create and save a single attention heatmap
    """
    os.makedirs(save_path, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        attention_weights.numpy(),
        xticklabels=source_chars,
        yticklabels=target_chars,
        cmap='Blues',
        cbar=True,
        ax=ax,
        cbar_kws={'label': 'Attention Weight'},
        annot=False,  # Set to True if you want to show values
        fmt='.2f'
    )
    
    # Customize plot
    ax.set_xlabel('Source Sequence (Latin)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Target Sequence (Devanagari)', fontsize=12, fontweight='bold')
    ax.set_title(f'Attention Heatmap - Sample {sample_idx+1}\n'
                f'Input: {input_text}\n'
                f'Predicted: {predicted_text}\n'
                f'Target: {target_text}', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save the plot
    filename = f"{save_path}/attention_heatmap_sample_{sample_idx+1}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename

def create_attention_grid(attention_data_list, save_path="attention_heatmaps", figsize=(20, 24)):
    """
    Create a 3x3 grid (or adjust based on number of samples) of attention heatmaps
    """
    num_samples = len(attention_data_list)
    if num_samples == 0:
        return None
        
    # Determine grid size - for 10 samples, use 4x3 grid
    if num_samples <= 9:
        rows, cols = 3, 3
    else:
        rows, cols = 4, 3
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if num_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, data in enumerate(attention_data_list[:min(num_samples, rows*cols)]):
        ax = axes[i]
        
        # Create heatmap
        sns.heatmap(
            data['attention_weights'].numpy(),
            xticklabels=data['source_chars'],
            yticklabels=data['target_chars'],
            cmap='Blues',
            cbar=True,
            ax=ax,
            cbar_kws={'label': 'Attention'},
        )
        
        ax.set_title(f'Sample {i+1}\n'
                    f'Input: {data["input_text"][:20]}...\n'
                    f'Pred: {data["predicted_text"][:20]}...\n'
                    f'True: {data["target_text"][:20]}...', 
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('Source', fontsize=8)
        ax.set_ylabel('Target', fontsize=8)
        
        # Make labels smaller for grid view
        ax.tick_params(axis='both', which='major', labelsize=6)
    
    # Hide unused subplots
    for j in range(len(attention_data_list), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    # Save grid
    os.makedirs(save_path, exist_ok=True)
    grid_filename = f"{save_path}/attention_heatmaps_grid.png"
    plt.savefig(grid_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return grid_filename

def save_attention_heatmaps_for_test_samples(model, test_loader, input_vocab, target_vocab, 
                                           device, num_samples=10, save_path="attention_heatmaps"):
    """
    Generate and save attention heatmaps for test samples
    
    Args:
        model: Your trained Seq2Seq model
        test_loader: Test data loader
        input_vocab: Input vocabulary dictionary
        target_vocab: Target vocabulary dictionary
        device: Device for inference
        num_samples: Number of samples to visualize (default 10)
        save_path: Directory to save heatmaps
    """
    if not model.use_attention:
        print("Model does not use attention mechanism. Cannot generate attention heatmaps.")
        return
    
    model.eval()
    
    # Create reverse vocabulary mappings
    idx2input = {i: c for c, i in input_vocab.items()}
    idx2target = {i: c for c, i in target_vocab.items()}
    
    sos_token = target_vocab['<sos>']
    eos_token = target_vocab['<eos>']
    
    attention_data_list = []
    sample_count = 0
    
    with torch.no_grad():
        for batch in test_loader:
            if sample_count >= num_samples:
                break
                
            src, tgt = batch
            src, tgt = src.to(device), tgt.to(device)
            
            for i in range(src.size(0)):  # Iterate through batch
                if sample_count >= num_samples:
                    break
                
                src_seq = src[i]
                tgt_seq = tgt[i]
                
                # Extract attention weights
                attention_weights, predicted_sequence, encoder_outputs = extract_attention_weights_from_model(
                    model, src_seq, input_vocab, target_vocab, device
                )
                
                # Convert sequences to characters (removing padding, SOS, EOS)
                source_chars = [idx2input[idx.item()] for idx in src_seq if idx.item() != 0]
                target_chars = [idx2target[idx] for idx in predicted_sequence 
                               if idx not in (0, sos_token, eos_token)]
                true_target_chars = [idx2target[idx.item()] for idx in tgt_seq 
                                   if idx.item() not in (0, sos_token, eos_token)]
                
                # Create text representations
                input_text = ''.join(source_chars)
                predicted_text = ''.join(target_chars)
                target_text = ''.join(true_target_chars)
                
                # Truncate attention weights to match actual sequence lengths
                actual_tgt_len = len(target_chars)
                actual_src_len = len(source_chars)
                
                if actual_tgt_len > 0 and actual_src_len > 0:
                    attention_weights_truncated = attention_weights[:actual_tgt_len, :actual_src_len]
                    
                    # Create individual heatmap
                    heatmap_filename = create_attention_heatmap(
                        attention_weights_truncated, source_chars, target_chars,
                        input_text, predicted_text, target_text, sample_count, save_path
                    )
                    
                    # Store data for grid creation
                    attention_data_list.append({
                        'sample_id': sample_count,
                        'attention_weights': attention_weights_truncated,
                        'source_chars': source_chars,
                        'target_chars': target_chars,
                        'input_text': input_text,
                        'predicted_text': predicted_text,
                        'target_text': target_text,
                        'heatmap_file': heatmap_filename
                    })
                    
                    print(f"Sample {sample_count + 1}: {input_text} -> {predicted_text} (True: {target_text})")
                    sample_count += 1
    
    # Create grid of all heatmaps
    if attention_data_list:
        grid_filename = create_attention_grid(attention_data_list, save_path)
        print(f"\nSaved {len(attention_data_list)} individual heatmaps in: {save_path}")
        print(f"Saved grid heatmap as: {grid_filename}")
        
        return attention_data_list, grid_filename
    else:
        print("No valid attention heatmaps could be generated.")
        return [], None

def log_attention_to_wandb(attention_data_list, grid_filename, step=None):
    """
    Log attention heatmaps to Weights & Biases
    """
    if not attention_data_list:
        return
    
    # Log grid image
    if grid_filename and os.path.exists(grid_filename):
        wandb.log({
            "attention/heatmaps_grid": wandb.Image(grid_filename, caption="Attention Heatmaps Grid")
        }, step=step)
    
    # Log individual heatmaps
    for data in attention_data_list:
        if os.path.exists(data['heatmap_file']):
            wandb.log({
                f"attention/sample_{data['sample_id']+1}": wandb.Image(
                    data['heatmap_file'], 
                    caption=f"Input: {data['input_text']} | Pred: {data['predicted_text']} | True: {data['target_text']}"
                )
            }, step=step)

# Modified version of your evaluate_and_log function
def evaluate_and_log_with_attention(config: dict):
    wandb.init(project=config["wandb_project"], entity=config["wandb_entity"], config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    from data import get_data_loaders
    train_loader, val_loader, test_loader, input_vocab, target_vocab = get_data_loaders(
        "./data/dakshina_dataset_v1.0/hi/lexicons/", 
        batch_size=config["batch_size"]
    )
                                                                                           
    if "use_attention" in config and config["use_attention"]:
        # Build model with attention
        model = build_model(
            input_dim=len(input_vocab),
            output_dim=len(target_vocab),
            embedding_dim=config["embedding_dim"],
            hidden_size=config["hidden_size"],
            encoder_layers=config["encoder_layers"],
            decoder_layers=config["decoder_layers"],
            cell_type=config["cell_type"],
            dropout=config["dropout"],
            beam_size=config["beam_size"],
            teacher_forcing_ratio=config["teacher_forcing_ratio"],
            use_attention=config["use_attention"],
            device=device
        )
        os.makedirs("predictions_attention", exist_ok=True)
        save_path = "predictions_attention/predictions.csv"
        attention_save_path = "predictions_attention/attention_heatmaps"
    else:
        print("Warning: Model not configured to use attention. Attention heatmaps will not be generated.")
        return

    # Train model
    from train import train_model
    model = train_model(
        model, train_loader, val_loader, 
        lr=config["lr"], 
        device=device, 
        wandb_logging=True, 
        teacher_forcing_ratio=config["teacher_forcing_ratio"], 
        epochs=5
    )

    # Evaluate on test set
    from evaluate_and_log import evaluate_model  # Import your existing evaluate_model function
    predictions, word_acc, char_acc = evaluate_model(
        model, test_loader, input_vocab, target_vocab, device=device
    )

    # Generate and save attention heatmaps
    print("\nGenerating attention heatmaps...")
    attention_data_list, grid_filename = save_attention_heatmaps_for_test_samples(
        model, test_loader, input_vocab, target_vocab, 
        device, num_samples=10, save_path=attention_save_path
    )
    
    # Log attention heatmaps to wandb
    if attention_data_list:
        log_attention_to_wandb(attention_data_list, grid_filename)

    # Save predictions
    import pandas as pd
    df = pd.DataFrame(predictions, columns=["Input", "Target", "Prediction"])
    df.to_csv(save_path, index=False)

    # Log to wandb
    wandb.log({
        "test_exact_match_accuracy": word_acc,
        "test_character_level_accuracy": char_acc
    })

    # Show markdown grid
    grid_markdown = df.head(10).to_markdown(index=False)
    print(grid_markdown)
    wandb.log({"sample_predictions_markdown": wandb.Html(f"<pre>{grid_markdown}</pre>")})

    # Log predictions table
    prediction_table = wandb.Table(columns=["Input", "Target", "Prediction"])
    for row in df.head(20).itertuples(index=False):
        prediction_table.add_data(row.Input, row.Target, row.Prediction)
    wandb.log({"Sample Predictions": prediction_table})