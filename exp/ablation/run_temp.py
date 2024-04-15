from functools import partial
from pathlib import Path
import copy
import numpy as np
# import optuna
import logging
import pandas as pd
import wandb
from datetime import datetime
import os
# wandb might cause an error without this.
os.environ["WANDB_START_METHOD"] = "thread"

import sys
sys.path.append('')
from src.utils import gen_data_grid, pmap, read_config, save_file
from src.data.gen_data import simulate_data
from src.utils import load_data
from src.metrics import edge_apr, edge_auroc, exp_shd
from src.methods.VCUDA import VCUDA
from src.globals import PROJECT_NAME

CUR_DIR = Path(__file__).parent

MAX_CPU = 10

def gen_train_datasets(data_cfg):
    datasets = []
    if 'e' not in data_cfg:
        data_cfg['e'] = data_cfg['e_coef'] * data_cfg['d']
    for i in range(data_cfg['n_dataset_train']):
        data_cfg['seed'] = i + 1
        data = simulate_data(**data_cfg)
        datasets.append((i, *data))
    return datasets

def process(task: dict):
    task_id, idataset, true_adj, X_train, X_test, model, model_cfg, data_cfg = task
    try:
        posterior_adj, MSE, pred_prob, training_time = model(X=(X_train, X_test), args_model=model_cfg, args_data=data_cfg, gt_adj=true_adj)
        # print(f'{pred_prob=}')
        # print(f'{true_adj=}')
    except Exception as e:
        logging.error(f'Error at task={task_id}', exc_info=True)
    output = dict(idataset=idataset, X=X_train, MSE=MSE, true_adj=true_adj, posterior_adj=posterior_adj, pred_prob=pred_prob, model_cfg=model_cfg, data_cfg=data_cfg)
    output['Training time'] = training_time
    return output

def do_trial(model, model_cfg, data_cfg, datasets):
    # model, model_cfg, data_cfg, datasets = trial_args
    wandb.init(
        project= f"{PROJECT_NAME}_HYPERPARAMS_TUNING",
        config=model_cfg,
        reinit=True,
        group='TEMP-CHANGE'
    )
    tasks = [(i, *data_point, model, model_cfg, data_cfg) for i, data_point in enumerate(datasets)]
    res = pmap(process, tasks, n_jobs=min(len(datasets), MAX_CPU), verbose=False)
    res = pd.DataFrame(res)
    scores = eval_res(method_name='', res=res)
    metrics = ["Dir-AUC-ROC", "Dir-AUC-PR", "SHD", "extra", "missing", "reverse"]
    print(f'SHD Summary\n{scores[metrics]}')
    stats = scores[metrics].mean()
    for k, v in stats.items():
        wandb.run.summary[k] = v
    wandb.finish(quiet=True)
    return scores

def eval_res(method_name:str, res: pd.DataFrame) -> pd.DataFrame:
    res[['SHD', 'extra', 'missing', 'reverse']] = res.apply(lambda x: exp_shd(x['posterior_adj'], x['true_adj']), axis=1, result_type='expand')
    res['Dir-AUC-ROC'] = res.apply(lambda x: edge_auroc(x['pred_prob'], x['true_adj']), axis=1)
    res['Dir-AUC-PR'] = res.apply(lambda x: edge_apr(x['pred_prob'], x['true_adj']), axis=1)
    res['Method'] = method_name
    return res

        
if __name__ == '__main__':
    logging.getLogger().setLevel(level=logging.ERROR)
    # read_cfg 
    cfg_path = CUR_DIR / 'cfg.yml'
    args = read_config(cfg_path) 
    print(f'{args=}')
    train_datasets = gen_train_datasets(copy.deepcopy(args['data']))
    data_cfg = gen_data_grid(args['data'])[0]
    eval_datasets = load_data(data_cfg)
    model = VCUDA
    model_cfg = args['methods']['VCUDA']
    wandb.setup()
    temp_choices = [0.3, 0.5, 1.0]
    total_res = []
    for temp_p in temp_choices: 
        model_cfg['temp_p'] = temp_p
        res = do_trial(model=model, model_cfg=model_cfg, data_cfg=data_cfg, datasets=eval_datasets)
        res['temp_p'] = temp_p
        total_res.append(pd.DataFrame(res))
    total_res = pd.concat(total_res)
    outfile = f"{data_cfg['dataname']}({data_cfg['n_dataset']}).csv" 
    save_file(data=total_res, outdir=CUR_DIR, outfile=outfile)