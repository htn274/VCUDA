import warnings
import logging
import numpy as np
from src.data.dataset import get_data_loader
from src.methods.VCUDA.DAGAutoencoderModel import DAGAutoencoderModel
from src.globals import PROJECT_NAME, SAVE_DIR
from src.utils import Timer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer
import torch

def cfg_to_str(
        seed_dataset,  # Seed to shuffle dataset. int
        dataname,  # Dataset name. string
        i_dataset,  # Dataset name. string
        split,  # Split for train/val/test sets. list of floats
        seed_model,  # Seed to init model. int
        **kwargs
):
    param_dicts = locals()
    full_config_name = ''
    for k, v in param_dicts.items():
        if k == 'kwargs': continue
        full_config_name += str(v) + '-'
    full_config_name = full_config_name[:-1]
    full_config_name = full_config_name.replace(" ", "")
    return full_config_name

def train(model, train_loader, val_loader, test_loader, max_epochs, val_freq, log_freq, device="cpu", verbose=False, logger=None, **kwargs):
    earlystop_callback = EarlyStopping(mode='min', 
                                           monitor='val_loss', 
                                           patience=10, 
                                           verbose=verbose) 
    # if torch.cuda.device_count():
    #     device = "gpu" 
    # else:
    #     device = "cpu"
    logging.info(f'Using {device=}')
    with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            trainer = Trainer(
                accelerator="gpu" if torch.cuda.device_count() else "cpu",
                devices="auto",
                max_epochs=max_epochs,
                logger=logger,
                log_every_n_steps=log_freq, 
                enable_checkpointing=False,
                enable_progress_bar=verbose,
                enable_model_summary=False,
                deterministic=False,
                num_sanity_val_steps=0,
                # detect_anomaly=True,
                callbacks=[earlystop_callback],
                # gradient_clip_val=1,
                # gradient_clip_algorithm="value",
                check_val_every_n_epoch=val_freq,
            )
            with Timer(name='Training time', verbose=verbose) as timer:
                trainer.fit(model=model, 
                        train_dataloaders=train_loader,
                        val_dataloaders=val_loader)
            training_time = timer.elapsed
            # test_mse = -1
            # if test_loader is not None:
            #     test_mse = trainer.test(model, test_loader, verbose=False)[0]['loss']
    return model, trainer, training_time

def VCUDA(X, args_model, args_data, **kwargs):
    random_state = args_model['seed_model']
    if random_state is not None:
        torch.random.fork_rng(enabled=True)
        torch.random.manual_seed(random_state)
    
    gt_adj = kwargs['gt_adj']
    # print(f'{gt_adj=}')
    # Load data
    (X_train, X_test) = X
    _, input_dim = X_train.shape
    # X_train = (X_train - X_train.mean()) / X_train.std()
    X_tensor = torch.as_tensor(X_train).type(torch.Tensor)
    train_loader, val_loader = get_data_loader(X = X_tensor, **args_model)
    test_loader = None
    if X_test is not None:
        X_test_tensor = torch.as_tensor(X_test).type(torch.Tensor)
        test_loader = torch.utils.data.DataLoader(X_test_tensor, batch_size=1024, shuffle=False, num_workers=0)
    # Train model
    logger = False
    if args_model['verbose']: 
        logger = WandbLogger(project=PROJECT_NAME, name=f'Ours-{args_data["dataname"]}')
        logger.experiment.config.update(args_data)

    # print(f'{args_model=}')
    model = DAGAutoencoderModel(input_dim=input_dim,
                                output_dim=1, gt_adj=gt_adj,
                                **args_model)
    # model.init_f_model(X_tensor, max_iters=100)
    model, trainer, training_time = train(model, train_loader, val_loader, test_loader, 
                                           logger= logger, gt_adj=gt_adj, **args_model)
    # print(f'{model.dag_sampler.edge_log_prob=}')
    # Sample posterior_adj
    n_samples = args_model.get('n_posterior_samples', 1)
    posterior_adj = model.sample(n_samples=n_samples)
    posterior_adj_np = posterior_adj.detach().numpy()
    pred_prob = np.mean(posterior_adj_np.reshape(posterior_adj.shape[0], -1), axis=0)
    test_mse = -1
    if test_loader is not None:
        test_mse = []
        for adj in posterior_adj:
            model.update_mask(new_mask=adj)
            test_mse.append(trainer.test(model, test_loader, verbose=False)[0]['loss'])
        test_mse = np.mean(test_mse)
    return posterior_adj_np, test_mse, pred_prob, training_time
    