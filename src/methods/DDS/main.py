import torch 
import time
import sys
sys.path.append('')
from src.data.dataset import get_data_loader
from src.methods.DDS.probabilistic_dag_autoencoder import ProbabilisticDAGAutoencoder
from src.methods.DDS.train_probabilistic_dag_autoencoder import train_autoencoder, compute_loss_reconstruction
from src.globals import SAVE_DIR
from src.utils import Timer

def full_config_dict_to_str(# Dataset parameters
        seed_dataset,  # Seed to shuffle dataset. int
        dataname,  # Dataset name. string
        split,  # Split for train/val/test sets. list of floats

        # Architecture parameters
        seed_model,  # Seed to init model. int
        ma_hidden_dims,  # Hidden dimensions. list of ints
        ma_architecture,    # Encoder architecture name. string
        ma_fast,  # Use fast masked autoencoder implementation. Boolean
        pd_initial_adj,  # If 'Learned', the adjacency matrix is learned. Otherwise 'GT' (Ground Truth) or 'RGT' (Reverse Ground Truth). string
        pd_temperature,  # Temperature for differentiable sorting. int
        pd_hard,  # Hard or soft sorting. boolean
        pd_order_type,  # Type of differentiable sorting. string
        pd_noise_factor,  # Noise factor for Sinkhorn sorting. int

        # Training parameters
        max_epochs,  # Maximum number of epochs for training. int
        patience,  # Patience for early stopping. int
        frequency,  # Frequency for early stopping test. int
        batch_size,  # Batch size. int
        ma_lr,  # Learning rate for mask encoder. float
        pd_lr,  # Learning rate proabilistic model. float
        loss,  # Loss name. string
        regr,  # Regularization factor in Bayesian loss. float
        prior_p, **kwargs):
    cfg_dict = {
        'seed_dataset': seed_dataset,
        'dataset_name': dataname,
        'split': list(split),
        'seed_model': seed_model,
        'ma_hidden_dims': list(ma_hidden_dims),
        'ma_architecture': ma_architecture,
        'ma_fast': ma_fast,
        'pd_initial_adj': pd_initial_adj,
        'pd_temperature': pd_temperature,
        'pd_hard': pd_hard,
        'pd_order_type': pd_order_type,
        'pd_noise_factor': pd_noise_factor,
        'max_epochs': max_epochs,
        'patience': patience,
        'frequency': frequency,
        'batch_size': batch_size,
        'ma_lr': ma_lr,
        'pd_lr': pd_lr,
        'loss': loss,
        'regr': regr,
        'prior_p': prior_p
    }
    full_config_name = ''
    for k, v in cfg_dict.items():
        full_config_name += str(v) + '-'
    full_config_name = full_config_name[:-1]
    full_config_name = full_config_name.replace(" ", "")
    return full_config_name

def DDS(X, args_model, args_data, **kwargs):
    (X_train, X_test) = X
    with torch.random.fork_rng():
        torch.random.manual_seed(args_model['seed_model'])
        # Load data
        X_tensor = torch.as_tensor(X_train).type(torch.Tensor)
        train_loader, val_loader = get_data_loader(X = X_tensor, scaled=False, **args_model)
        if X_test is not None:
            X_test_tensor = torch.as_tensor(X_test).type(torch.Tensor)
            test_loader = torch.utils.data.DataLoader(X_test_tensor, batch_size=1024, shuffle=False, num_workers=0)
        # Train model
        _, input_dim = X_train.shape 
        model = ProbabilisticDAGAutoencoder(input_dim=input_dim,
                                            output_dim=1, 
                                            seed=args_model['seed_model'],
                                            **args_model)

        # full_config_dict = {**args_data, **args_model}
        # print(full_config_dict)
        # full_config_name = full_config_dict_to_str(**full_config_dict)
        # model_path = SAVE_DIR / args_model['directory_model'] 
        # model_path.mkdir(parents=True, exist_ok=True)
        # model_path = model_path / f'model-{full_config_name}'

        with Timer(name='Training time', verbose=True) as timer:
            model, train_losses, val_losses, train_mse, val_mse = train_autoencoder(model,
                                                                    train_loader,
                                                                    val_loader,
                                                                    **args_model,)
        training_time = timer.elapsed

        test_mse = -1
        if X_test is not None: test_loss, test_mse = compute_loss_reconstruction(model, test_loader)
        # return samples probabilistic adjacency
        posterior_adj = model.sample(n_samples=args_model['n_posterior_samples'])
        # pred_prob = model.prob_adj()
        # print(f'{posterior_adj=}')
        # print(f'{pred_prob=}')
        pred_prob = model.prob_adj()
        return posterior_adj, test_mse, pred_prob, training_time