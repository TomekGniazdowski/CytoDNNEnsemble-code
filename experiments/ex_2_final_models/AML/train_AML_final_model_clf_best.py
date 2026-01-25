from os import environ, system
from time import time

import pandas as pd
import wandb

wandb.login()

environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
environ["CUDA_VISIBLE_DEVICES"] = "0"


LATENT_SPACE_EXTRACTOR_LOG_DIR = "../../../logs/AML/ex1/latent_space_extractor/1720785026.1583338"
CNN_1D_EXTRACTOR_LOG_DIR = "../../../logs/AML/ex1/cnn_1d_extractor/1720113197.7665977"
LATENT_SPACE_EXTRACTOR_SIZE = "large"
CNN_1D_EXTRACTOR_SIZE = "small"

DATASET_PATH = "/ssd2/datasets/users/tgniazdowski/CytoDNNEnsemble_data/AML"
TRAIN_FCS_INFO = f"{DATASET_PATH}/train/train_labels.csv"
TEST_FCS_INFO = f"{DATASET_PATH}/test/test_labels.csv"
MARKERFILE = f"{DATASET_PATH}/marker.csv"

DATASET_NAME = "AML"
LOGGER_NAME = f'CytoDNNEnsemble@{DATASET_NAME}@final_model@ex2'
NSUBSET = 20_000
CO_FACTOR = 5
NMARKER = pd.read_csv(MARKERFILE).shape[1]

NUM_CELLS_LIST = ["256", "512", "1024"]
SEED_LIST = ["1", "2", "3", "4", "5"]
LOG_DIR = f"../../../logs/{DATASET_NAME}/ex2/final_model"
LOG_INTERVAL = 1
SAVE_INTERVAL = 1

BIN_FILE = "../../../fit_final_model.py"

BETA_1 = 0.9
BETA_2 = 0.999
config_hyperparameters = {
    'method': 'grid',
    'parameters': {
        'batch_size': {'values': []},
        'final_clf_type': {'values': ['final_model_clf']},
        'final_model_clf_hidden_dim': {'values': []},
        'learning_rate_ae': {'values': []},
        'learning_rate': {'values': []},
        'num_epochs': {'values': []},
        'weight_decay': {'values': []},
        'latent_dim_ae': {'values': []},
        'train_backbone_clf': {'values': []}, 
        'patience': {'values': []}
    }
}

def hyperparameters_tuning_with_wandb(config=None):
    with wandb.init(config=config):
        config = wandb.config
        config_folder = f"{LOG_DIR}/{time()}"
        
        for ncell in NUM_CELLS_LIST:
            run_folder = f"{config_folder}/{ncell}"        
            for seed in SEED_LIST:
                system(
                    f"python {BIN_FILE} "
                    f"--dataset_name {DATASET_NAME} "
                    f"--batch_size {config.batch_size} "
                    f"--logger_name {LOGGER_NAME} "
                    f"--final_clf_type {config.final_clf_type} "
                    f"--final_model_clf_hidden_dim {config.final_model_clf_hidden_dim} "
                    f"--learning_rate_ae {config.learning_rate_ae} "
                    f"--learning_rate {config.learning_rate} "
                    f"--num_epochs {config.num_epochs} "
                    f"--weight_decay {config.weight_decay} "
                    f"--latent_dim_ae {config.latent_dim_ae} "
                    f"--train_backbone_clf {config.train_backbone_clf} "
                    f"--latent_space_extractor_size {LATENT_SPACE_EXTRACTOR_SIZE} "
                    f"--cnn_1d_extractor_size {CNN_1D_EXTRACTOR_SIZE} "
                    f"--latent_space_extractor_log_dir {LATENT_SPACE_EXTRACTOR_LOG_DIR}/{LATENT_SPACE_EXTRACTOR_SIZE}/{ncell}/seed_{seed} "
                    f"--cnn_1d_extractor_log_dir {CNN_1D_EXTRACTOR_LOG_DIR}/{CNN_1D_EXTRACTOR_SIZE}/{ncell}/seed_{seed} "
                    f"--beta1 {BETA_1} "
                    f"--beta2 {BETA_2} "
                    f"--patience {config.patience} "
                    f"--nmarker {NMARKER} "
                    f"--ncell {ncell} "
                    f"--nsubset {NSUBSET} "
                    f"--co_factor {CO_FACTOR} "
                    f"--train_fcs_info {TRAIN_FCS_INFO} "
                    f"--test_fcs_info {TEST_FCS_INFO} "
                    f"--markerfile {MARKERFILE} "
                    "--generate_valid "
                    "--test_rsampling "
                    "--shuffle "
                    f"--log_dir {run_folder}/seed_{seed} "
                    f"--log_interval {LOG_INTERVAL} "
                    f"--save_interval {SAVE_INTERVAL} "
                    f"--seed {seed} "
                    "--device cuda"
                )

sweep_id = wandb.sweep(config_hyperparameters, project=LOGGER_NAME)
wandb.agent(sweep_id, function=hyperparameters_tuning_with_wandb)