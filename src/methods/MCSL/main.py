from castle.algorithms.gradient import mcsl
import torch
from src.utils import Timer
import numpy as np
import os 
os.environ['CASTLE_BACKEND'] = 'pytorch'

def MCSL(X, args_model, args_data, **kwargs):
    (X_train, X_test) = X
    model = mcsl.MCSL(random_seed=args_model.pop('seed_model', 1230), **args_model)
    with Timer(name='Training time', verbose=True) as timer:
        model.learn(X_train)
    training_time = timer.elapsed
    pred_adj = np.array(model.causal_matrix)
    pred_adj = np.expand_dims(pred_adj, axis=0)
    pred_score = np.array(model.causal_matrix_weight)
    test_mse = -1
    if X_test is not None:
        w_prime = torch.as_tensor(pred_adj[0], device=model.device)
        X_test_tensor = torch.as_tensor(X_test, device=model.device)
        test_mse = model.masked_model._get_mse_loss(X_test_tensor, w_prime) / np.prod(X_test.shape)
    return pred_adj, test_mse, pred_score, training_time