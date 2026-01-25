import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from meters import AverageMeter
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm


def train_model(
    model,
    optimizer,
    optimizer_scheduler,
    num_epochs,
    train_loader,
    valid_loader,
    early_stopping,
    args,
    logger,
    model_name
):
    logger.info('**** Start Training ****')

    losses = AverageMeter(round=3)
    data_time = AverageMeter(round=3)
    step_time = AverageMeter(round=3)
    best_auc = 0
    pbar = tqdm(range(num_epochs), initial=0, dynamic_ncols=True, smoothing=0.01)

    for epoch in pbar:
        model.train()

        for x, y in train_loader:
            start_time = time.time()
            x, y = x.to(args.device).double(), y.to(args.device).double()
            data_time.update(time.time() - start_time)
            
            prob = model(x)
            loss = F.binary_cross_entropy(prob, y, reduction='mean')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_scheduler.step()

            step_time.update(time.time() - start_time)
            losses.update(loss.item())

        if (epoch + 1) % args.log_interval == 0:
            val_acc, val_loss, val_auc = validate_model(valid_loader, model, args)
            pbar.set_description(
                "Epoch: {}/{}, data: {:.3f}, step: {:.3f}, loss: {:.3f}, val_loss: {:.3f}, val_acc: {:.3f}, val_auc: {:.3f}".format(
                    str(epoch + 1).zfill(4), num_epochs, data_time.avg,
                    step_time.avg, losses.avg, val_loss, val_acc, val_auc
                )
            )
            stats = {
                'epoch': epoch + 1,
                'loss': losses.avg,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_auc': val_auc
            }
            logger._log_to_wandb(stats=stats, epoch=epoch + 1)
            early_stopping(val_loss=val_loss)
            if early_stopping.early_stop:
                logger.info(f"Training early stops at epoch: {epoch+1}")
                break

        losses.reset()
        data_time.reset()
        step_time.reset()

        if (epoch + 1) % args.save_interval == 0:
            val_acc, val_loss, val_auc = validate_model(valid_loader, model, args)
            is_best = val_auc >= best_auc
            best_auc = max(best_auc, val_auc)

            if is_best:
                torch.save(model.state_dict(), f'{args.log_dir}/{model_name}/best_model.pt')
                torch.save({
                    'val_acc': val_acc,
                    'val_auc': val_auc,
                    'epoch': epoch + 1
                    }, f'{args.log_dir}/{model_name}/best.ckpt')

        pbar.update()

    pbar.close()
    logger.info("**** Training Finished ****")


def validate_model(valid_loader, model, args):
    model.eval()
    losses = AverageMeter(round=3)
    correct_num, total_num = 0, 0
    y_pred, y_true = [], []

    for x, y in valid_loader:
        x, y = x.to(args.device).double(), y.to(args.device).double()
        with torch.no_grad():
            prob = model(x)
            loss = F.binary_cross_entropy(prob, y, reduction='mean')
            pred_label = torch.ge(prob, 0.5)
        losses.update(loss.item(), n=x.size(0))
        v = (pred_label == y).sum()

        y_true.append(y.detach().cpu().numpy())
        y_pred.append(prob.detach().cpu().numpy())

        correct_num += v.item()
        total_num += x.size(0)

    acc = float(correct_num) / total_num

    y_true, y_pred = np.hstack(y_true), np.hstack(y_pred)
    auc = roc_auc_score(y_true, y_pred)

    return acc, losses.avg, auc


def test_model(test_samples, test_phenotypes, model, device, model_name, log_dir):
    model.eval()
    correct_num, total_num = 0, 0
    y_pred, y_true = [], []
    losses = []

    for i, (sample, label) in enumerate(zip(test_samples, test_phenotypes)):
        with torch.no_grad():
            sample = torch.from_numpy(sample).to(device).double()
            true_label = torch.tensor([label]).to(device).double()
            if model_name == 'final_model_attention':
                model._attention_maps_path = f'{log_dir}/attention_maps/batch_{i}.pt'
            elif model_name in ['latent_space_extractor', 'cnn_1d_extractor']:
                model._representations_path = f'{log_dir}/representations/batch_{i}.pt'
            elif model_name != 'final_model_clf':
                raise ValueError(f'Invalid model name: {model_name}')
            prob = model(sample)
            loss = F.binary_cross_entropy(prob, true_label, reduction='mean')
            pred_label = torch.ge(prob, 0.5)

        losses.append(loss.item())
        v = (pred_label == label).sum()

        y_true.append(label)
        y_pred.append(prob.detach().cpu().numpy())

        correct_num += v.item()
        total_num += 1

    acc = float(correct_num) / total_num

    y_true, y_pred = np.array(y_true), np.hstack(y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    auc = roc_auc_score(y_true, y_pred)
    eval_loss = np.mean(np.array(losses))

    return eval_loss, acc, auc, fpr, tpr, (y_true, y_pred)


def test_save_model_results(model, test_samples, test_phenotypes, model_name, args):
    _, test_acc, test_auc, test_fpr, test_tpr, (y_true, y_pred) = test_model(test_samples, test_phenotypes, model, args.device, model_name, args.log_dir)
    pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}).to_csv(f'{args.log_dir}/model_predictions_test_{model_name}.csv', index=False)

    with open(f'{args.log_dir}/test_result_{model_name}.pkl', 'wb') as f:
        test_stat = {
            'test_samples': test_samples,
            'test_phenotypes': test_phenotypes,
            'fpr': test_fpr,
            'tpr': test_tpr,
            'test_acc': test_acc,
            'test_auc': test_auc
        }
        pickle.dump(test_stat, f)
        

def test_single_models(models, test_samples, test_phenotypes, device):
    for i, model in enumerate(models):
        model.eval()
        correct_num, total_num = 0, 0
        y_pred, y_true = [], []

        for sample, label in zip(test_samples, test_phenotypes):
            with torch.no_grad():
                sample = torch.from_numpy(sample).to(device).double()
                prob = model(sample)
                pred_label = torch.ge(prob, 0.5)

            v = (pred_label == label).sum()

            y_true.append(label)
            y_pred.append(prob.detach().cpu().numpy())

            correct_num += v.item()
            total_num += 1

        acc = float(correct_num) / total_num
        auc = roc_auc_score(y_true, y_pred)
        
        print(f'Model {i}:', acc, auc)