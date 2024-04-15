import copy
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.append('')
from argparse import ArgumentParser
import logging
import wandb

from src.utils import Timer, gen_data_grid, load_data, load_methods, pmap, read_config, save_file, load_file 
from src.metrics import edge_apr, edge_auroc, exp_shd
from src.globals import PROJECT_NAME
from src.viz import plot_multiple_d

CUR_DIR = Path(__file__).parent
SAVE_DIR = CUR_DIR / 'results'
PLOT_DIR = CUR_DIR / 'plot'

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

def eval_res(method_name:str, res: pd.DataFrame) -> pd.DataFrame:
    res[['SHD', 'extra', 'missing', 'reverse']] = res.apply(lambda x: exp_shd(x['posterior_adj'], x['true_adj']), axis=1, result_type='expand')
    res['Dir-AUC-ROC'] = res.apply(lambda x: edge_auroc(x['pred_prob'], x['true_adj']), axis=1)
    res['Dir-AUC-PR'] = res.apply(lambda x: edge_apr(x['pred_prob'], x['true_adj']), axis=1)
    res['Method'] = method_name
    # print(res[['Dir-AUC-ROC', 'Dir-AUC-PR', 'MSE']])
    return res

def run(method_name:str, model:object, model_cfg:dict, data_cfg:dict, datasets:tuple, n_jobs:int=1) -> pd.DataFrame:
    if data_cfg['dataname'] == 'sachs':
        print('Prepare for Sachs')
        rng = np.random.RandomState(0)
        n_trials = 10
        seeds = [rng.randint(2**10) for _ in range(n_trials)] 
        _, true_adj, X_train, X_test = datasets[0]
        tasks = []
        for itrial in range(n_trials):
            model_cfg_i = copy.deepcopy(model_cfg)
            model_cfg_i['seed_model'] = seeds[itrial]
            task = (itrial, itrial, true_adj, X_train, X_test, model, model_cfg_i, data_cfg)
            tasks.append(task)
    else:
        tasks = [(i, *data_point, model, model_cfg, data_cfg) for i, data_point in enumerate(datasets)]
    with Timer(name=f'Running {method_name=}', verbose=True):
        res = pmap(process, tasks, n_jobs, verbose=True)
        res = pd.DataFrame(res)
    return res

def main(args_methods:dict, data_cfg:dict, datasets:tuple, methods_dict: dict, n_jobs:int=1, wandb_logger=None) -> pd.DataFrame:
    """
    Run several methods with a specific data config
    Args:
    Return:
    """
    total_res = []
    artifact_res = wandb.Artifact(name=f'Results-{data_cfg["dataname"]}', type='results')
    for method_name, model in methods_dict.items():
        model_cfg = args_methods.get(method_name, {}).copy()
        print(f'{method_name=}\n{model_cfg=}')
        outdir = SAVE_DIR / method_name
        outfile = f"{data_cfg['dataname']}({data_cfg['n_dataset']}).csv" 
        do_run = model_cfg.pop('run', True)
        try:
            if not do_run:
                # Read results if it exists
                print(f'Read results from {outdir / outfile}')
                res = load_file(outdir / outfile)
                # res = pd.read_pickle(outdir / outfile)
                print(f'{res["model_cfg"][0]}')
            else: 
                raise Exception
        except Exception as e:
            print(f'{e}. Runing method {method_name}...')
            res = run(method_name, model, model_cfg, data_cfg, datasets, n_jobs)
            save_file(data=res, outdir= outdir, outfile=outfile)
        artifact_res.add_file(local_path=outdir/outfile, name=f'{method_name}/{outfile}')
        method_sum = eval_res(method_name=method_name, res=res)
        method_sum['dataname'] = data_cfg['dataname']
        if 'd' in data_cfg:
            method_sum['Num_vars'] = data_cfg['d']
        total_res.append(method_sum)
    total_res = pd.concat(total_res)
    if wandb_logger: wandb_logger.log_artifact(artifact_res)
    return total_res


def print_final_result(df:pd.DataFrame, metrics:list):
    res = df.groupby(['dataname', 'Method'], axis=0)[metrics].agg(['mean', 'std'])
    print(res)
    return res
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cfg_file', type=str, default= 'linear.yml',
                            help='Config path for running experiment')
    parser.add_argument('--run_meds', type=str, nargs='+', default=['VCUDA', 'DDS', 'GraNDAG', 'MCSL', 'DiBS'])
    parser.add_argument('--log', type=str, default='WARN', choices='FATAL ERROR WARN INFO DEBUG NOTSET'.split())
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args() 
    logging.getLogger().setLevel(level=eval(f'logging.{args.log}'))

    cfg_path = CUR_DIR / f'configs/{args.cfg_file}'
    cfg = read_config(cfg_path) 
    wandb_logger = None
    if args.wandb: 
        wandb_logger = wandb.init(project=f'{PROJECT_NAME}-experiments', config=cfg)
        wandb_logger.config.update(args)
    methods_dict = load_methods(args.run_meds)
    data_cfg_list = gen_data_grid(cfg['data'])
    total_res = []
    for data_cfg in data_cfg_list:
        print("="*100)
        print(f'{data_cfg=}')
        datasets = load_data(data_cfg)
        res = main(cfg['methods'], data_cfg, datasets, methods_dict, n_jobs=cfg['n_jobs'], wandb_logger=wandb_logger)
        total_res.append(res)
    total_res = pd.concat(total_res)
    metrics = ['SHD', 'Dir-AUC-ROC', 'Dir-AUC-PR', 'MSE', 'Training time']
    res_summary = print_final_result(total_res, metrics=metrics)
    if args.plot: fig = plot_multiple_d(total_res=total_res, metrics=metrics, outdir=CUR_DIR)
    if wandb_logger:
        res_summary.reset_index(inplace=True)
        res_summary.columns = ["_".join(a) for a in res_summary.columns.to_flat_index()]
        table = wandb.Table(dataframe=res_summary)
        wandb_logger.log({"Result summary": table})
        if args.plot: wandb_logger.log({"Result plot": wandb.Image(fig)})