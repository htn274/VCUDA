from castle.algorithms.gradient import gran_dag
import torch
from src.utils import Timer
import numpy as np
import os 
os.environ['CASTLE_BACKEND'] = 'pytorch'

def GraNDAG(X, args_model, args_data, **kwargs):
    (X_train, X_test) = X
    _, input_dim = X_train.shape
    random_seed = args_model.pop('seed_model', 42)
    model = gran_dag.GraNDAG(input_dim=input_dim, random_seed=random_seed, **args_model)
    with Timer(name='Training time', verbose=True) as timer:
        model.learn(X_train)
    training_time = timer.elapsed

    pred_adj = np.array(model.causal_matrix)
    pred_adj = np.expand_dims(pred_adj, axis=0)
    pred_score = model.model.get_w_adj().detach().cpu().numpy()
    test_mse = -1
    if X_test is not None:
        X_test_tensor = torch.as_tensor(X_test).type(torch.Tensor)
        weights, biases, extra_params = model.model.get_parameters(mode="wbx")
        test_mse = -torch.mean(model.model.compute_log_likelihood(X_test_tensor, weights, biases, extra_params)).item()
    # print(f'{pred_adj=}\n{pred_score=}')
    return pred_adj, test_mse, pred_score, training_time