## Deep Learning(DA6401) Assignment 3
Repo for assignment3 submission in DA6401

#### Roll no: DA24M010
#### Name: Mohit Singh

## Wandb Report link : 
https://wandb.ai/da24m010-indian-institute-of-technology-madras/DA6401%20Assignments/reports/DA6401-Assignment-3--VmlldzoxMjM4NTc4MQ

## Github repo link :
https://github.com/DA24M010/da6401_Assignment3.git

# Assignment Overview
The goal of this assignment is fourfold: 
1. learn how to model sequence to sequence learning problems using Recurrent Neural Networks 
2. compare different cells such as vanilla RNN, LSTM and GRU
3. understand how attention networks overcome the limitations of vanilla seq2seq models 
4. visualise the interactions between different components in a RNN based model.

# Structure
```
├── scripts/
│   ├── data.py
│   ├── extract_dataset.py
│   ├── evaluate.py
│   ├── evaluate_and_log.py
│   ├── model.py
│   ├── model_w_attention.py
│   ├── train.py
│   ├── train_best_model.py
│   ├── generate_heatmaps.py
│   ├── run_sweeps.py
│   ├── run_att_sweeps.py
│   └── run_sweeps.py
├── .gitignore
├── readme.md
├── best_att_config.yaml
└── best_config.yaml

```

# Script files
- **`extract_data.py`**: Download the Dakshina Dataset.
- **`data.py`**: Defines the Dakshina dataset class and a utility function to return data loaders for train, validation, and test sets.
- **`model.py`**: Implements the vanilla RNN architecture.
- **`model_w_attention.py`**: Implements the vanilla RNN architecture.
- **`train.py`**: Contains the training loop logic for the RNN model, using specified hyperparameters.
- **`run_sweeps.py`**: Executes WandB hyperparameter sweeps to tune the RNN model and logs metrics.
- **`run_att_sweeps.py`**: Executes WandB hyperparameter sweeps to tune the RNN model with attention and logs metrics.
- **`train_best_model.py`**: Trains the RNN models using the best hyperparameters from WandB sweeps, evaluates on the test set, and logs final test accuracy and predictions.
- **`evaluate.py`**: Test the RNN models on the hyperparameter specified in config files. 

# Installation and Setup
### 1. Clone the repository:
```sh
git clone https://github.com/DA24M010/da6401_Assignment3.git
cd da6401_Assignment3
```

### 2. Install dependencies:
```sh
pip install -r requirements.txt
```

### 3. Setup Weights & Biases (W&B)
Create an account on [W&B](https://wandb.ai/) and log in:
```sh
wandb login
```

# Running Scripts
### Running hypereparameter tuning (vanilla RNN)
Running hyperparameter tuning on vanilla RNN model and logging to wandb
```bash
python ./scripts/run_sweeps.py --project your_project_name --entity your_wandb_username
```
*Change the hyperparameters for tuning inside the script.*

### Running hypereparameter tuning (RNN with Attention)
Running hyperparameter tuning on RNN model with RNN and logging to wandb
```bash
python ./scripts/run_att_sweeps.py --project your_project_name --entity your_wandb_username
```
*Change the hyperparameters for tuning inside the script.*

### Evaluating the RNN Model on the Test Set
If you need to evaluate model for a specific set of hyperparameters on the test set:
```bash
python ./scripts/evaluate.py --project your_project_name --entity your_wandb_username --config best_config.yaml
```
*Change the hyperparameters for running inference inside the script. Generates test accuracy and prediction on test dataset logs in WandB*

### Evaluating the RNN Model with Attention on the Test Set
If you need to evaluate RNN model with attention for a specific set of hyperparameters on the test set:
```bash
python ./scripts/evaluate.py --project your_project_name --entity your_wandb_username --config best_att_config.yaml
```
*Change the hyperparameters for running inference inside the script. Generates test accuracy and prediction on test dataset logs in WandB*
