import copy
import torch
import numpy as np
from tqdm import trange

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

def compute_loss_reconstruction(model, loader):
    model.eval()
    with torch.no_grad():
        loss = 0
        for batch_index, (X) in enumerate(loader):
            X = X.to(device)
            model.update_mask(type='deterministic')
            X_pred = model(X)
            if batch_index == 0:
                X_pred_all = X_pred.reshape(-1).to("cpu")
                X_all = X.reshape(-1).to("cpu")
            else:
                X_pred_all = torch.cat([X_pred_all, X_pred.reshape(-1).to("cpu")], dim=0)
                X_all = torch.cat([X_all, X.reshape(-1).to("cpu")], dim=0)
            loss += X.size(0) * model.grad_loss.item()
        loss = loss / X_pred_all.size(0)
        reconstruction = ((X_all - X_pred_all)**2).mean().item()
    model.train()
    return loss, reconstruction

def train_autoencoder(model, train_loader, val_loader, max_epochs=200, frequency=2, patience=50, verbose=False, **kwargs):
    model.to(device)
    model.train()
    train_losses, val_losses, train_mses, val_mses = [], [], [], []
    best_val_loss = float("Inf")
    best_model = None
    pbar = trange(max_epochs, desc='Training', disable=not verbose)
    for epoch in pbar:
        for batch_index, (X_train) in enumerate(train_loader):
            X_train = X_train.to(device)
            model.update_mask()
            X_pred = model(X_train)
            model.step()

        if epoch % frequency == 0:
            # Stats on data sets
            train_loss, train_mse = compute_loss_reconstruction(model, train_loader)
            train_losses.append(round(train_loss, 3))
            train_mses.append(round(train_mse, 3))

            val_loss, val_mse = compute_loss_reconstruction(model, val_loader)
            val_losses.append(val_loss)
            val_mses.append(val_mse)

            pbar.set_postfix(train_mse=f'{train_mse:.2f}', val_mse=f'{val_mse:.2f}')

            if best_val_loss > val_losses[-1]:
                best_val_loss = val_losses[-1]
                best_model = copy.deepcopy(model)

            if np.isnan(val_losses[-1]):
                print('Detected NaN Loss')
                break

            if int(epoch / frequency) > patience and val_losses[-patience] <= min(val_losses[-patience:]):
                # print('Early Stopping.')
                break

    model = best_model
    return model, train_losses, val_losses, train_mses, val_mses