import os
import torch
import wandb
import pandas as pd
from tqdm import tqdm
from model import build_model  # Your model builder
from train import train_model  # Your train function
from data import get_data_loaders     # Function to return train/val/test loaders and tokenizers

def evaluate_and_log(config: dict):
    wandb.init(project=config["wandb_project"], entity=config["wandb_entity"], config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader, test_loader, input_vocab, target_vocab = get_data_loaders("./data/dakshina_dataset_v1.0/hi/lexicons/", batch_size = config["batch_size"])
                                                                                           
    if "use_attention" in config:
        from model_w_attention import build_model
        
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

    else:
        from model import build_model
        
        # Build model without attention
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
            device=device
        )
        os.makedirs("predictions_attention", exist_ok=True)
        save_path = "predictions_attention/predictions.csv"


    # Train model
    model = train_model(model, train_loader, val_loader, lr = config["lr"], 
                        device= device, wandb_logging= True, teacher_forcing_ratio= config["teacher_forcing_ratio"], epochs= 2)

    # Evaluate on test set
    predictions, word_acc, char_acc = evaluate_model(
        model, test_loader, input_vocab, target_vocab,
        device=device
    )

    # Save predictions
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
    
def evaluate_model(model, test_loader, input_vocab, target_vocab, device):
    model.eval()
    predictions = []
    total_words = 0
    correct_words = 0
    total_chars = 0
    correct_chars = 0

    sos_token = target_vocab['<sos>']
    eos_token = target_vocab['<eos>']
    idx2input = {i: c for c, i in input_vocab.items()}
    idx2target = {i: c for c, i in target_vocab.items()}

    with torch.no_grad():
        for batch in test_loader:
            src, tgt = batch
            src, tgt = src.to(device), tgt.to(device)

            # Call the full Seq2Seq model — no teacher forcing
            outputs = model(src, tgt, teacher_forcing_ratio=0.0)  # shape: (batch_size, tgt_len, vocab_size)

            # Get predicted tokens
            predicted_idxs = outputs.argmax(dim=2)  # shape: (batch_size, tgt_len)

            for src_seq, pred_seq, tgt_seq in zip(src, predicted_idxs, tgt):
                src_chars = [idx2input[idx.item()] for idx in src_seq if idx.item() != 0]
                pred_chars = [
                    idx2target[idx.item()] for idx in pred_seq
                    if idx.item() not in (0, sos_token, eos_token)
                ]
                tgt_chars = [
                    idx2target[idx.item()] for idx in tgt_seq
                    if idx.item() not in (0, sos_token, eos_token)
                ]

                src_str = "".join(src_chars)
                pred_str = "".join(pred_chars)
                tgt_str = "".join(tgt_chars)

                predictions.append((src_str, tgt_str, pred_str))

                total_words += 1
                correct_words += (pred_str == tgt_str)
                char_matches = sum(pc == tc for pc, tc in zip(pred_str, tgt_str))
                correct_chars += char_matches
                total_chars += max(len(tgt_str), len(pred_str))

    word_accuracy = correct_words / total_words if total_words > 0 else 0
    char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
    return predictions, word_accuracy, char_accuracy
