import argparse
import wandb
from data import get_data_loaders
from model import CNNModel
import torch
from train import train_model
import matplotlib.pyplot as plt
import torchvision.transforms as T
import numpy as np

def main(project, entity):
    best_config = {
        "num_filters": 64,
        "filter_multiplier": 2,
        "kernel_size": 5,
        "activation": "gelu",
        "dropout_rate": 0.2,
        "use_batchnorm": False,
        "dense_units": 1024,
        "data_augmentation": True,
        "lr": 0.0001
    }

    train_loader, val_loader, test_loader, label_map = get_data_loaders(
        './partA/train.csv', './partA/val.csv',
        data_augmentation=best_config["data_augmentation"], seed=7
    )

    best_model = CNNModel(
        input_shape=(3, 224, 224),
        num_filters=best_config["num_filters"],
        kernel_size=best_config["kernel_size"],
        filter_multiplier=best_config["filter_multiplier"],
        activation=best_config["activation"],
        dropout_rate=best_config["dropout_rate"],
        dense_units=best_config["dense_units"],
        use_batchnorm=best_config["use_batchnorm"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model.to(device)
    print(f"Training model on {device}")
    train_model(best_model, train_loader, val_loader, epochs=1, lr=best_config["lr"], device=device, wandb_logging=False)

    best_model.eval()
    correct = 0
    total = 0
    all_preds, all_imgs, all_labels = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = best_model(inputs)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu())
            all_imgs.extend(inputs.cpu())
            all_labels.extend(labels.cpu())

    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")

    # Denormalize for visualization
    inv_transform = T.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
    idx_to_class = {v: k for k, v in label_map.items()}

    fig, axes = plt.subplots(10, 3, figsize=(9, 30))
    for i, ax in enumerate(axes.flat):
        img = inv_transform(all_imgs[i]).clamp(0, 1)
        img_np = img.permute(1, 2, 0).numpy()

        label = idx_to_class[all_labels[i].item()]
        pred = idx_to_class[all_preds[i].item()]

        ax.imshow(img_np)
        ax.axis("off")
        ax.set_title(f"Pred: {pred}", color="green" if pred == label else "red", fontsize=10)

    plt.tight_layout()
    plt.savefig("test_preds.png")

    # WandB logging
    wandb.init(project=project, entity=entity, name="Best Model")
    wandb.log({
        "final_test_accuracy": test_acc,
        "test_predictions_grid": wandb.Image("test_preds.png")
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="DA6401_Assignments", help="WandB project name")
    parser.add_argument("--entity", default="da24m010-indian-institute-of-technology-madras", help="WandB entity name")
    args = parser.parse_args()

    main(args.project, args.entity)
