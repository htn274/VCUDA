from argparse import ArgumentParser
import sys

sys.path.append('')
from src.methods.VCUDA.dag_sampling import DAG_sampler
from src.methods.DDS.probabilistic_dag import ProbabilisticDAG
from src.utils import check_dag
import numpy as np
import pandas as pd
import time 
import tqdm
from pathlib import Path
from itertools import product
import logging
import seaborn as sns 
import matplotlib.pyplot as plt
from tueplots import axes, bundles, figsizes, fontsizes
logging.getLogger().setLevel(level=logging.ERROR)

CUR_DIR = Path(__file__).parent

def score_time(model, n_samples):
    sampling_times = np.zeros(n_samples)
    for i in range(n_samples + 1):
        t0 = time.time()
        A = model.sample().detach().cpu().numpy()
        if i == 0: continue
        sampling_times[i - 1] = time.time() - t0
        # assert check_dag(A.detach().cpu().numpy())
    res = pd.DataFrame({'Runid': np.arange(n_samples), 'Time': sampling_times})
    return res

def get_model_by_name(model_name, n_nodes):
    if model_name == 'DDS-Sinkhorn':
        return ProbabilisticDAG(n_nodes=n_nodes, hard=True, order_type='sinkhorn', initial_adj=None)
    elif model_name == 'DDS-Softsort':
        return ProbabilisticDAG(n_nodes=n_nodes, hard=True, order_type='topk', initial_adj=None)
    elif model_name == 'Ours':
        return DAG_sampler(n_nodes, hard=True, device="cuda")
    raise NotImplementedError

def main(args):
    results = []
    n_samples = 10
    n_nodes_list = [500, 1000, 3000, 5000,]
    model_list = args.methods #['DDS-Sinkhorn', 'DDS-Softsort', 'Ours']
    for n_nodes, model_name in tqdm.tqdm(list(product(n_nodes_list, model_list))):
        model = get_model_by_name(model_name, n_nodes)
        running_time = score_time(model, n_samples)
        running_time['Method'] = model_name
        running_time['Num_vars'] = n_nodes
        results.append(running_time)
    results = pd.concat(results)
    results.to_csv(CUR_DIR / 'times.csv')
    return results

def plot_time(total_res, outdir=None):
    colors = ['red', 'orange', 'green', 'purple', 'steelblue']
    df = pd.melt(total_res, id_vars=['Method', 'Num_vars'], value_vars=['Time'], value_name='Value', var_name='Metric')
    df['Same'] = ''
    palette = dict(zip(sorted(df.Method.unique(), reverse=True), colors))
    plt.rcParams.update(figsizes.aaai2024_half())
    plt.rcParams.update(axes.lines())
    plt.rcParams.update(fontsizes._from_base(base=9))
    g = sns.FacetGrid(df, row='Same', col='Metric', sharey=False, aspect=1.2)
    g.map_dataframe(sns.lineplot, x='Num_vars', y='Value', hue='Method', 
                    marker='o', errorbar=('ci', 95), palette=palette)
    g.set_xlabels(r'#Variables')
    g.set_ylabels(r'Sampling Time (seconds)')
    g.set_titles('')
    g.add_legend(label_order=df.Method.unique(), fontsize=12)
    ax = g.axes.flat[0]
    ax.set_yscale('log')
    ax.grid(axis='y', linestyle='--')
    g.tight_layout()
    if outdir: g.savefig(f'{outdir}/plot.pdf', dpi=300)
    return g.figure

def plot_cmp(results):
    methods_dict = {
        'DDS-Sinkhorn': 'Gumbel-Sinkhorn', 
        'DDS-Softsort': 'Gumbel-Top-k'
    }
    results['Method'] = results['Method'].map(lambda x: methods_dict.get(x, x))
    g = plot_time(results, CUR_DIR)
    g.savefig(f'{CUR_DIR}/plot.pdf')

def plot_ours(results):
    results_ours = results[results['Method'] == 'Ours']
    g = plot_time(results_ours, CUR_DIR)
    g.savefig(f'{CUR_DIR}/plot_ours.pdf')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--methods', type=str, nargs='+', default=['Ours', 'DDS-Sinkhorn', 'DDS-Softsort'])
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.run:
        results = main(args)
    else:
        results = pd.read_csv(CUR_DIR / 'times.csv')
    print(results.groupby(['Method', 'Num_vars'])['Time'].mean())
    plot_cmp(results)
    # plot_ours(results)
    
    