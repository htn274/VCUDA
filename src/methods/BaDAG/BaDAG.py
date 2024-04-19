from datetime import datetime
from pathlib import Path
import sys
sys.path.append('')

import os
from uuid import uuid4
from .causica.datasets.dataset import Dataset
from .causica.datasets.variables import Variables
import torch
from .causica.models.torch_model import TorchModel
from .causica.utils.torch_utils import get_torch_device
from .causica.utils.causality_utils import intervention_to_tensor, intervene_graph
from functorch import vmap
from src.utils import Timer

from .causica.models.bayesdag.bayesdag_nonlinear import BayesDAGNonLinear
from .causica.models.bayesdag.bayesdag_linear import BayesDAGLinear
import logging
import numpy as np

logger = logging.getLogger('BaDAG')

def mse(model: BayesDAGNonLinear, X, W_adj_samples, params, buffers, batch_size):
    X = torch.as_tensor(X, device=model.device, dtype=torch.float)
    (intervention_idxs, intervention_mask, intervention_values,) = intervention_to_tensor(
        None,
        None,
        model.variables.group_mask,
        device=model.device,
    )

    if intervention_mask is not None and intervention_values is not None:
        X[:, intervention_mask] = intervention_values

    # This sets certain elements of W_adj to 0, to respect the intervention
    W_adj = intervene_graph(W_adj_samples, intervention_idxs, copy_graph=False)
    mse_samples = []
    for curr in range(0, X.shape[0], batch_size):
        x = X[curr : curr + batch_size]
        with torch.no_grad():
            if params is None:
                predict = model.ICGNN.predict(x, W_adj).transpose(0, 1).unsqueeze(-2)
            else:
                predict = vmap(model.ICGNN.predict, in_dims=(0, 0, None, 0))(
                    params, buffers, x, W_adj
                )
                predict_shape = predict.shape
                predict = predict.reshape(-1, model.num_particles, *predict_shape[1:])
            mse_samples.append(torch.square(x - predict).mean().item())

    return np.mean(mse_samples)

def BaDAG(X, args_model, args_data, **kwargs): 
         # n_posteriors, model_name, model_params, train_params, eval_batchsize, device, random_state, verbose, **kwargs):
    X_train, X_test = X
    _, d = X_train.shape
    # device: cpu, gpu, or number (i.e., 0 = cuda:0)
    device = get_torch_device(args_model['device'])
    model_id = str(uuid4())
    save_dir = os.path.join('exps/ckpts/BaDAG', model_id)
    # os.makedirs(save_dir)

    model_name = args_data['method']
    model_params = args_model['model_params']
    model_params['random_seed'] = args_model['random_state']
    train_params = args_model['train_params']
    model = {'linear': BayesDAGLinear, 'nonlinear': BayesDAGNonLinear}.get(model_name)

    variables = Variables.create_from_data_and_dict(X_train, np.ones_like(X_train))
    model: BayesDAGNonLinear = model.create(model_id, save_dir, variables, model_params, device=device)

    if isinstance(model, TorchModel):
        num_trainable_parameters = sum(p.numel() for p in model.parameters())
        logger.info(f'{num_trainable_parameters = }')
    
    logger.info(f"Created model with ID {model.model_id}.")

    dataset = Dataset(
        train_data=X_train,
        train_mask=np.ones_like(X_train),
        test_data=X_test,
        test_mask=np.ones_like(X_test),
        variables=variables,
        graph_args={'num_variables': d, 'seed': args_model['random_state']}
    )
    causal_dataset = dataset.to_causal(adjacency_data=None, subgraph_data=None, intervention_data=None)
    with Timer(f'BaDAG training', verbose=args_model['verbose']) as t:
        model.run_train(dataset=causal_dataset, train_config_dict=train_params)  # type: ignore

    training_time = t.elapsed
    W_adj_samples, params, buffers, A_samples = model.get_weighted_adj_matrix(samples=args_model['n_posteriors'], return_adj=True)
    test_mse = mse(model=model, X=X_test, W_adj_samples=W_adj_samples, params=params, buffers=buffers, batch_size=args_model['eval_batchsize'])
    post_dags = A_samples.cpu().numpy()
    W_adj_samples = W_adj_samples.cpu().numpy()
    pred_prob = np.mean(W_adj_samples.reshape(W_adj_samples.shape[0], -1), axis=0)

    # save_dir = Path(f'exps/results/{Path(__file__).with_suffix("").name}/{datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")}')
    # save_dir.mkdir(parents=True, exist_ok=True)
    # np.save(save_dir / 'post_dags.npy', post_dags)
    # np.save(save_dir / 'thetas.npy', W_adj_samples.cpu().numpy())

    # ret = dict(save_dir=str(save_dir.relative_to('.')), test_mse=test_mse, training_time=t.elapsed)
    return post_dags, test_mse, pred_prob, training_time