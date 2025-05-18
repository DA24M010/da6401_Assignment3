import argparse
import torch
import wandb
from data import get_data_loaders 
from model import build_model      
from train import train_model    

# Sweep Configuration
sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "val_accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "embedding_dim": {"values": [32, 64, 128, 256]},
        "hidden_size": {"values": [64, 128, 256]},
        "encoder_layers": {"values": [1, 2, 3]},
        "decoder_layers": {"values": [1, 2, 3]},
        "cell_type": {"values": ["rnn", "gru", "lstm"]},
        "dropout": {"values": [0.2, 0.3, 0.5]},
        "beam_size": {"values": [1, 3, 5]},
        "lr": {"values": [0.1, 0.01, 0.001, 0.0001]},
        "teacher_forcing_ratio": {"values": [0.5, 0.7, 0.9]},
        "batch_size": {"values": [16, 32, 64, 128]},
    }
}

def make_run_name(config):
    return (
        f"emb_{config['embedding_dim']}_"
        f"hid_{config['hidden_size']}_"
        f"enc_{config['encoder_layers']}_"
        f"dec_{config['decoder_layers']}_"
        f"cell_{config['cell_type']}_"
        f"drop_{config['dropout']}_"
        f"beam_{config['beam_size']}_"
        f"lr_{config['lr']}"
        f"tfr{config['teacher_forcing_ratio']}"
        f"bs{config['batch_size']}"
    )

def wandb_train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        wandb.run.name = make_run_name(config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader, val_loader, test_loader, input_vocab, target_vocab = get_data_loaders("./data/dakshina_dataset_v1.0/hi/lexicons/", batch_size = config.batch_size)

        model = build_model(            
            input_dim=len(input_vocab),
            output_dim=len(target_vocab),
            embedding_dim=config.embedding_dim,
            hidden_size=config.hidden_size,
            encoder_layers=config.encoder_layers,
            decoder_layers=config.decoder_layers,
            cell_type=config.cell_type,
            dropout=config.dropout,
            beam_size=config.beam_size,
            teacher_forcing_ratio=config.teacher_forcing_ratio,
            device=device
        )
    
        # Train the model
        train_model(
            model, 
            train_loader, 
            val_loader, 
            epochs=10, 
            lr=config.lr, 
            device=device, 
            wandb_logging=True,
            teacher_forcing_ratio=config.teacher_forcing_ratio
        )
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="DA6401_Assignments", help="WandB project name")
    parser.add_argument("--entity", default="da24m010-indian-institute-of-technology-madras", help="WandB entity name")
    args = parser.parse_args()

    sweep_id = wandb.sweep(sweep_config, project=args.project, entity=args.entity)
    wandb.agent(sweep_id, function=wandb_train)
