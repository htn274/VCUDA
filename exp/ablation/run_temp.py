import copy
from functools import partial
from pathlib import Path
import matplotlib
import numpy as np
# import optuna
import logging
import pandas as pd
import wandb
from datetime import datetime
import itertools
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from tueplots import axes, bundles, figsizes, fontsizes
# wandb might cause an error without this.
os.environ["WANDB_START_METHOD"] = "thread"

import sys
sys.path.append('')
from src.utils import gen_data_grid, load_file, pmap, read_config, save_file
from src.data.gen_data import simulate_data
from src.utils import load_data
from src.metrics import edge_apr, edge_auroc, exp_shd
from src.methods.VCUDA import VCUDA
from src.globals import PROJECT_NAME

CUR_DIR = Path(__file__).parent

OBJ_DIR_MAP = {
    'Dir-AUC-ROC': 'maximize',
    'Dir-AUC-PR': 'maximize',
    'SHD' : 'minimize'
}

higher_metrics = ['AUC-ROC', 'AUC-PR']

MAX_CPU = 5

matplotlib.rc('font', family='DejaVu Sans')
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['lines.linewidth'] = 1
matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'


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
    output = dict(idataset=idataset, X=X_train, MSE=MSE, true_adj=true_adj, posterior_adj=posterior_adj, pred_prob=pred_prob, model_cfg=model_cfg, data_cfg=data_cfg, temp_p=model_cfg['temp_p'])
    output['Training time'] = training_time
    return output

def do_trial(model, model_cfg, data_cfg, datasets):
    # model, model_cfg, data_cfg, datasets = trial_args
    tasks = [(i, *data_point, model, model_cfg, data_cfg) for i, data_point in enumerate(datasets)]
    res = pmap(process, tasks, n_jobs=min(len(datasets), MAX_CPU), verbose=False)
    res = pd.DataFrame(res)
    scores = eval_res(method_name='', res=res)
    metrics = ["Dir-AUC-ROC", "Dir-AUC-PR", "SHD", "extra", "missing", "reverse"]
    print(f'SHD Summary\n{scores[metrics]}')
    return scores

def eval_res(method_name:str, res: pd.DataFrame) -> pd.DataFrame:
    res[['SHD', 'extra', 'missing', 'reverse']] = res.apply(lambda x: exp_shd(x['posterior_adj'], x['true_adj']), axis=1, result_type='expand')
    res['Dir-AUC-ROC'] = res.apply(lambda x: edge_auroc(x['pred_prob'], x['true_adj']), axis=1)
    res['Dir-AUC-PR'] = res.apply(lambda x: edge_apr(x['pred_prob'], x['true_adj']), axis=1)
    if method_name != '':
        res['Method'] = method_name
    return res

def run(args):
    data_cfgs = gen_data_grid(args['data'])
    eval_datasets = [load_data(cfg) for cfg in data_cfgs]
    # eval_datasets = list(itertools.chain(eval_datasets))
    model = VCUDA
    model_cfg = args['model']
    temp_choices = [0.1, 0.3, 0.5, 1.0]
    tasks = []
    for temp_p, (data_cfg, dataset) in itertools.product(temp_choices, zip(data_cfgs, eval_datasets)): 
        model_cfg['temp_p'] = temp_p
        tasks = tasks + [(i, *data_point, model, copy.deepcopy(model_cfg), copy.deepcopy(data_cfg)) for i, data_point in enumerate(dataset)]
    res = pmap(process, tasks, n_jobs=args['n_jobs'], verbose=True)
    res = pd.DataFrame(res)
    res = eval_res(method_name='', res=res)
    print(res.groupby(['temp_p'])[['Dir-AUC-ROC', 'Dir-AUC-PR']].mean())
    save_file(data=res, outdir=CUR_DIR/ 'res_temp', outfile='linear.csv')

def format_metric(metric):
    if metric == 'Dir-AUC-ROC' or metric == 'Dir-AUC-PR':
        metric = metric[4:]
    if metric == 'D_order':
        texts = metric.split('_')
        return (r'${main}_\textrm{{{sub}}}$($\downarrow$)'.format(main=texts[0], sub=texts[1]))
    if metric in higher_metrics:
        return r'\textbf{{{metric}}} ($\uparrow$)'.format(metric=metric)
    return (r'\textbf{{{metric}}} ($\downarrow$)'.format(metric=metric))

def plot(args):
    df = pd.read_csv(f'{CUR_DIR}/res_temp/linear.csv')
    metrics = ['Dir-AUC-ROC', 'Dir-AUC-PR', 'MSE']
    df = pd.melt(df, id_vars=['temp_p'], value_vars=metrics, value_name='Value', var_name='Metric')
    df.Metric = df.Metric.map(format_metric)
    print(df.Metric.unique())
    df['Same'] = ''
    colors = ['red', 'orange', 'green', 'purple', 'steelblue']
    g = sns.FacetGrid(df, row='Same', col='Metric', sharey=False, aspect=1)
    g.map_dataframe(sns.lineplot, x='temp_p', y='Value', palette=colors,  errorbar='se', err_style='band', markers=True)
    g.set_xlabels(r'Temperature $t$', fontsize=12)
    g.set_ylabels('')
    g.set_yticklabels(fontsize=14)
    g.set_xticklabels(fontsize=12)
    g.set_titles(r'{col_name}', size=13)
    g.set(xlim=(0.1, 1.0))
    for i, ax in enumerate(g.axes.flat):
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.grid(axis='y', linestyle='--')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    g.tight_layout() 
    g.savefig(CUR_DIR / 'temps.pdf')
        
if __name__ == '__main__':
    logging.getLogger().setLevel(level=logging.ERROR)
    # read_cfg 
    cfg_path = CUR_DIR / 'linear.yml'
    args = read_config(cfg_path) 
    print(f'{args=}')
    # run(args)
    plot(args)
    