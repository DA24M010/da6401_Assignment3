import argparse
import yaml
from evaluate_and_log import evaluate_and_log

def load_config_from_yaml(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="Evaluate best Seq2Seq model on test set")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file containing best model hyperparameters"
    )
    parser.add_argument(
        "--project", default="DA6401_Assignments",
        help="WandB project name"
    )
    parser.add_argument(
        "--entity", default="da24m010-indian-institute-of-technology-madras",
        help="WandB entity name"
    )

    args = parser.parse_args()

    # Load configuration from YAML
    config = load_config_from_yaml(args.config)

    # Inject WandB settings into config
    config["wandb_project"] = args.project
    config["wandb_entity"] = args.entity

    # Run full evaluation pipeline
    evaluate_and_log(config)

if __name__ == "__main__":
    main()
