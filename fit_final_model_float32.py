import argparse
import os
import sys
from json import dump as json_dump

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

from cyto_dnn_ensemble.autoencoder_utils.dataset import Dataset as AutoencoderDataset
from cyto_dnn_ensemble.large_model.cnn_autoencoder import CNNAutoencoderLarge
from cyto_dnn_ensemble.large_model.model import (
    CNN1DExtractorLarge,
    LatentSpaceExtractorLarge,
)
from cyto_dnn_ensemble.medium_model.cnn_autoencoder import CNNAutoencoderMedium
from cyto_dnn_ensemble.medium_model.model import (
    CNN1DExtractorMedium,
    LatentSpaceExtractorMedium,
)
from cyto_dnn_ensemble.model_float32 import CytoDNNEnsembleModel
from cyto_dnn_ensemble.small_model.cnn_autoencoder import CNNAutoencoderSmall
from cyto_dnn_ensemble.small_model.model import (
    CNN1DExtractorSmall,
    LatentSpaceExtractorSmall,
)
from data import generete_train_val_test_data
from logger import WandbLogger
from train_float32 import test_save_model_results, test_single_models, train_model
from utils import EarlyStopping, down_rsampling, parse_true_value, set_seed


def fit_final_model(args):
    set_seed(args.seed)
    os.makedirs(f'{args.log_dir}/{args.final_clf_type}', exist_ok=False)
    os.makedirs(f'{args.log_dir}/attention_maps', exist_ok=False)
    
    train_loader, valid_loader, train_data, _, test_samples, test_phenotypes = generete_train_val_test_data(args)
    
    dataset_for_autoencoder = AutoencoderDataset(train_data, args.batch_size, num_bins=100, device=args.device, seed=args.seed)
    dataset_for_autoencoder.setup() 
    
    if args.latent_space_extractor_size == 'small':
        cnn_autoencoder = CNNAutoencoderSmall.load_from_checkpoint(
            f'{args.latent_space_extractor_log_dir}/cnn_autoencoder/model.pt',
            learning_rate=args.learning_rate_ae,
            latent_dim=args.latent_dim_ae,
            num_bins=dataset_for_autoencoder.num_bins
        ).to(args.device).to(torch.float32)
        latent_space_extractor = LatentSpaceExtractorSmall(
                cnn_encoder=cnn_autoencoder._encoder,
                latent_dim=args.latent_dim_ae,
                num_markers=args.nmarker,
                dataset_to_histograms_of_markers=dataset_for_autoencoder._dataset_to_histograms_of_markers
                ).to(args.device).to(torch.float32)
        latent_space_extractor.load_state_dict(torch.load(f'{args.latent_space_extractor_log_dir}/latent_space_extractor/best_model.pt'))

    elif args.latent_space_extractor_size == 'medium':
        cnn_autoencoder = CNNAutoencoderMedium.load_from_checkpoint(
            f'{args.latent_space_extractor_log_dir}/cnn_autoencoder/model.pt',
            learning_rate=args.learning_rate_ae,
            latent_dim=args.latent_dim_ae,
            num_bins=dataset_for_autoencoder.num_bins
        ).to(args.device).to(torch.float32)
        latent_space_extractor = LatentSpaceExtractorMedium(
                cnn_encoder=cnn_autoencoder._encoder, 
                latent_dim=args.latent_dim_ae,
                num_markers=args.nmarker,
                dataset_to_histograms_of_markers=dataset_for_autoencoder._dataset_to_histograms_of_markers
                ).to(args.device).to(torch.float32)
        latent_space_extractor.load_state_dict(torch.load(f'{args.latent_space_extractor_log_dir}/latent_space_extractor/best_model.pt'))
    
    elif args.latent_space_extractor_size == 'large':
        cnn_autoencoder = CNNAutoencoderLarge.load_from_checkpoint(
            f'{args.latent_space_extractor_log_dir}/cnn_autoencoder/model.pt',
            learning_rate=args.learning_rate_ae,
            latent_dim=args.latent_dim_ae,
            num_bins=dataset_for_autoencoder.num_bins
        ).to(args.device).to(torch.float32)
        latent_space_extractor = LatentSpaceExtractorLarge(
                cnn_encoder=cnn_autoencoder._encoder, 
                latent_dim=args.latent_dim_ae,
                num_markers=args.nmarker,
                dataset_to_histograms_of_markers=dataset_for_autoencoder._dataset_to_histograms_of_markers
                ).to(args.device).to(torch.float32)
        latent_space_extractor.load_state_dict(torch.load(f'{args.latent_space_extractor_log_dir}/latent_space_extractor/best_model.pt'))

    if args.cnn_1d_extractor_size == 'small':
        cnn_1d_extractor = CNN1DExtractorSmall(
            num_cells=args.ncell,
            num_markers=args.nmarker,
        ).to(args.device).to(torch.float32)
        cnn_1d_extractor.load_state_dict(torch.load(f'{args.cnn_1d_extractor_log_dir}/cnn_1d_extractor/best_model.pt'))
    
    elif args.cnn_1d_extractor_size == 'medium':
        cnn_1d_extractor = CNN1DExtractorMedium(
            num_cells=args.ncell,
            num_markers=args.nmarker,
        ).to(args.device).to(torch.float32)
        cnn_1d_extractor.load_state_dict(torch.load(f'{args.cnn_1d_extractor_log_dir}/cnn_1d_extractor/best_model.pt'))
    
    elif args.cnn_1d_extractor_size == 'large':
        cnn_1d_extractor = CNN1DExtractorLarge(
            num_cells=args.ncell,
            num_markers=args.nmarker,
        ).to(args.device).to(torch.float32)
        cnn_1d_extractor.load_state_dict(torch.load(f'{args.cnn_1d_extractor_log_dir}/cnn_1d_extractor/best_model.pt'))
    
    test_samples = [np.expand_dims(sample, 0).astype(np.float32) for sample in test_samples]
    if args.test_rsampling:
        test_samples = [down_rsampling(sample, args.ncell, axis=1) for sample in test_samples]
    test_single_models([latent_space_extractor, cnn_1d_extractor], test_samples, test_phenotypes, device=args.device)
    
    cyto_dnn_ensemble_model = CytoDNNEnsembleModel(
        latent_space_extractor=latent_space_extractor,
        cnn_1d_extractor=cnn_1d_extractor,
        train_backbone=args.train_backbone_clf,
        final_classifier_type=args.final_clf_type,
        final_model_clf_hidden_dim=args.final_model_clf_hidden_dim
    ).to(args.device).to(torch.float32)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    optimizer = optim.Adam(
        cyto_dnn_ensemble_model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )
    optimizer_scheduler = OneCycleLR(optimizer, max_lr=args.learning_rate, steps_per_epoch=len(train_loader), epochs=args.num_epochs)
    logger = WandbLogger(
        logger_name=args.logger_name,
        log_dir=args.log_dir,
        stream=sys.stdout,
        args=args,
        wandb_project=args.logger_name
    )
    train_model(
        model=cyto_dnn_ensemble_model,
        num_epochs=args.num_epochs,
        optimizer=optimizer,
        optimizer_scheduler=optimizer_scheduler,
        train_loader=train_loader,
        valid_loader=valid_loader,
        early_stopping=early_stopping,
        args=args,
        logger=logger,
        model_name=args.final_clf_type
    )
    cyto_dnn_ensemble_model.load_state_dict(torch.load(f'{args.log_dir}/{args.final_clf_type}/best_model.pt'))
    
    test_save_model_results(cyto_dnn_ensemble_model, test_samples, test_phenotypes, args.final_clf_type, args)    
    logger.info("Testing finished, saving training configurations....")
    with open(f"{args.log_dir}/config.json", 'w') as json_file:
        json_dump(vars(args), json_file)
    logger.info("Done")
    

def main():
    parser = argparse.ArgumentParser("Cyto DNN Ensemble Model Training")
    parser.add_argument('--dataset_name', type=str, help='name of the dataset')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--logger_name', type=str)
    
    parser.add_argument('--final_clf_type', type=str)
    parser.add_argument('--final_model_clf_hidden_dim', type=int)
    parser.add_argument('--learning_rate_ae', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--latent_dim_ae', type=int)
    parser.add_argument('--train_backbone_clf', type=parse_true_value)
    
    parser.add_argument('--latent_space_extractor_size', type=str)
    parser.add_argument('--cnn_1d_extractor_size', type=str)
    
    parser.add_argument('--latent_space_extractor_log_dir', type=str)
    parser.add_argument('--cnn_1d_extractor_log_dir', type=str)

    parser.add_argument('--beta1', type=float, help='beta_1 params in the optimizer')
    parser.add_argument('--beta2', type=float, help='beta_2 params in the optimizer')
    parser.add_argument('--patience', type=int, help='the patience param for early stopping')

    parser.add_argument('--train_fcs_info', type=str, default=None, help='path to train fcs info file')
    parser.add_argument('--valid_fcs_info', default=None, type=str, help='path to valid fcs info file')
    parser.add_argument('--test_fcs_info', type=str, default=None, help='path to test fcs info file')
    parser.add_argument('--train_pkl', type=str, default=None, help='path to the training pickle file')
    parser.add_argument('--valid_pkl', type=str, default=None, help='path to the valid pickle file')
    parser.add_argument('--test_pkl', type=str, default=None, help='path to the testing pickle file')

    parser.add_argument('--markerfile', type=str, help='path to marker indication file')
    parser.add_argument('--generate_valid', action='store_true', help='whether to generate valid data from train data')
    parser.add_argument('--test_rsampling', action='store_true', help='whether to test model using sampled data')
    parser.add_argument('--pkl', action='store_true', help='load data directly from pickled data')

    parser.add_argument('--nsubset', type=int, help='total number of multi-cell inputs that will be generated per class')
    parser.add_argument('--ncell', type=int, help='number of cells per multi-cell input')
    parser.add_argument('--nmarker', type=int, help='number of markers per cell')
    parser.add_argument('--co_factor', default=5, type=float, help='arcsinh normalization factor')
    parser.add_argument('--per_sample', action='store_true', help='whether the nsubset argument refers to each class or each input')

    parser.add_argument('--shuffle', action='store_true', help='whether to shuffle the data')
    parser.add_argument('--log_dir', type=str, help='path to log dir')
    parser.add_argument('--log_interval', default=1, type=int, help='logging interval')
    parser.add_argument('--save_interval', default=5, type=int, help='save model interval')

    parser.add_argument('--seed', type=int, help='random seed to use')
    parser.add_argument('--device', type=str, help='specify the training device')
    parser.add_argument('--ckpt', default=None, type=str, help='path to the checkpoint file')

    args = parser.parse_args()
    
    fit_final_model(args)


if __name__ == "__main__":
    main()
