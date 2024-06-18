from argparse import ArgumentParser
import logging
from pathlib import Path

import sys
import numpy as np

import pandas as pd
sys.path.append('')
from src.utils import gen_data_grid, load_file, read_config
from run import eval_res
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from tueplots import axes, bundles, figsizes, fontsizes

matplotlib.rc('font', family='DejaVu Sans')
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['lines.linewidth'] = 1
matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'
CUR_DIR = Path(__file__).parent
SAVE_DIR = CUR_DIR / 'results'
PLOT_DIR = CUR_DIR / 'plot'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

colors = ['red', 'orange', 'green', 'purple', 'steelblue', 'gold']
higher_metrics = ['AUC-ROC', 'AUC-PR']

def format_metric(metric):
    if metric == 'Dir-AUC-ROC' or metric == 'Dir-AUC-PR':
        metric = metric[4:]
    if metric == 'D_order':
        texts = metric.split('_')
        return (r'${main}_\textrm{{{sub}}}$ ($\downarrow$)'.format(main=texts[0], sub=texts[1]))
    if metric in higher_metrics:
        return (r'\textbf{{{metric}}} ($\uparrow$)'.format(metric=metric))
    return (r'\textbf{{{metric}}} ($\downarrow$)'.format(metric=metric))


def get_res(data_cfg, methods):
    if data_cfg['dataname'] == 'sachs': 
        data_cfg['n_dataset'] = 1
    elif data_cfg['dataname'] == 'syntren':
        data_cfg['n_dataset'] = 10

    total_res = []
    for method_name in methods:
        outdir = SAVE_DIR / method_name
        outfile = f"{data_cfg['dataname']}({data_cfg['n_dataset']}).csv" 
        res = load_file(outdir / outfile)
        res = res.iloc[:5]
        method_sum = eval_res(method_name=method_name, res=res)
        if 'd' in data_cfg:
            method_sum['Num_vars'] = data_cfg['d']
        total_res.append(method_sum)
    total_res = pd.concat(total_res)
    return total_res

def plot_training_time(total_res, outdir=None):
    df = pd.melt(total_res, id_vars=['Method', 'Num_vars'], value_vars='Training time', value_name='Value', var_name='Metric')
    df['Same'] = ''
    df['Method'] = df['Method'].map(change_method_name)
    palette = dict(zip(df.Method.unique(), colors))
    plt.rcParams.update(figsizes.aaai2024_half())
    plt.rcParams.update(axes.lines())
    plt.rcParams.update(fontsizes._from_base(base=9))
    g = sns.FacetGrid(df, row='Same', col='Metric', sharey=False, aspect=2.2)
    g.map_dataframe(sns.lineplot, x='Num_vars', y='Value', hue='Method', 
                    marker='o', errorbar=('ci', 95), palette=palette)
    g.set_xlabels(r'\#Variables')
    g.set_ylabels(r'Training time (seconds)')
    g.set_titles('')
    g.add_legend(label_order=df.Method.unique())
    ax = g.axes.flat[0]
    ax.set_yscale('log')
    ax.grid(axis='y', linestyle='--')
    g.tight_layout()
    if outdir is not None:
        g.savefig(outdir, dpi=300)

def change_method_name(x: str):
    if x == 'Ours':
        return 'V-CUDA'
    if x == 'GraNDAG':
        return 'GraN-DAG'
    return x


def plot_structure_synthetic(total_res, outfile):
    metrics = ['Dir-AUC-ROC', 'Dir-AUC-PR', 'MSE',]
    df = pd.melt(total_res, id_vars=['Method', 'Num_vars', 'dag_type'], value_vars=metrics, value_name='Value', var_name='Metric')
    df.Metric = df.Metric.map(format_metric)
    df['Method'] = df['Method'].map(change_method_name)
    # print(df.Method.unique())
    palette = dict(zip(df.Method.unique(), colors))
    dag_type_dict = {
        'ER':  'Erdős-Rényi', 
        'SF':  'Scale-Free',
    }
    df['dag_type'] = df['dag_type'].map(lambda x: dag_type_dict[x])
    dag_types = df['dag_type'].unique()
    ncols = len(metrics)
    nrows = len(dag_types)
    plt.rcParams.update(figsizes.aaai2024_half(nrows=nrows, ncols=ncols))
    plt.rcParams.update(axes.lines())
    plt.rcParams.update(fontsizes._from_base(base=15))
    g = sns.FacetGrid(df, row='dag_type', col='Metric', sharey=False, aspect=1.1)
    g.map_dataframe(sns.boxplot, y='Value', x='Method', palette=palette)
    g.set_titles(r'{col_name}')
    N_cols = len(metrics)
    for i, ax in enumerate(g.axes.flat):
        ax.grid(axis='y', linestyle='--')
        ax.xaxis.set_tick_params(labelrotation=45)
        if i % N_cols == 0:
            ax.set_ylabel(rf'{dag_types[i // N_cols]}')
        if i // N_cols == 1: 
            ax.set_title('')
            ax.set_xlabel(rf'$(d = {df["Num_vars"][0]})$', )
    g.tight_layout()
    g.savefig(outfile)

def viz_structure_synthetic(cfg_file):
    cfg_path = CUR_DIR / f'configs/{cfg_file}'
    cfg = read_config(cfg_path) 
    d_list = cfg['data']['d']
    data_cfg_list = gen_data_grid(cfg['data'])
    total_res = []
    for data_cfg in data_cfg_list:
        res = get_res(data_cfg, args.methods)
        res['dag_type'] = data_cfg['dag_type']
        total_res.append(res)
    df = pd.concat(total_res)

    for d in d_list:
        outfile = PLOT_DIR / f'{data_cfg["method"]}-{d}.pdf'
        res = df[df['Num_vars'] == d]
        plot_structure_synthetic(res, outfile)
    

def viz_training_time(cfg_file, filename):
    cfg_path = CUR_DIR / f'configs/{cfg_file}'
    cfg = read_config(cfg_path) 
    data_cfg_list = gen_data_grid(cfg['data'])
    total_res = []
    for data_cfg in data_cfg_list:
        res = get_res(data_cfg, args.methods)
        total_res.append(res)
    total_res = pd.concat(total_res)
    plot_training_time(total_res, outdir=PLOT_DIR / filename) 

def viz_real(cfg_file):
    cfg_path = CUR_DIR / f'configs/{cfg_file}'
    cfg = read_config(cfg_path) 
    data_cfg_list = gen_data_grid(cfg['data'])
    total_res = []
    for data_cfg in data_cfg_list:
        res = get_res(data_cfg, args.methods)
        total_res.append(res)
    total_res = pd.concat(total_res)
    mean = total_res.groupby(['Method'])[['Dir-AUC-ROC', 'Dir-AUC-PR']].mean()
    std = total_res.groupby(['Method'])[['Dir-AUC-ROC', 'Dir-AUC-PR']].std()
    d = '$' + np.round(mean, 2).astype(str) + '\pm{\scriptstyle ' + np.round(std, 2).astype(str) + '}$'
    d = d.loc[['GraNDAG', 'MCSL', 'DiBS', 'DDS', 'Ours']]
    print(d)
    d.to_clipboard(excel=True, header=False, index=False)
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cfg_file', type=str, default= 'linear.yml',
                            help='Config path for running experiment')
    parser.add_argument('--methods', type=str, nargs='+', default=['VCUDA', 'GraNDAG', 'MCSL', 'DiBS', 'DDS', 'BaDAG'])
    args = parser.parse_args() 
    # viz_structure_synthetic(args.cfg_file)
    viz_training_time(args.cfg_file, filename='training_time_plot.pdf')
    # viz_training_time(args.cfg_file, filename='training_time_plot_ours_dds.pdf')
    # viz_real(args.cfg_file)