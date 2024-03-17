from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.rc('font', family='DejaVu Sans')
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['lines.linewidth'] = 1
matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'

colors = ['red', 'orange', 'green', 'purple', 'steelblue']
higher_metrics = ['Un-AUC-ROC', 'Un-AUC-PR', 'Dir-AUC-ROC', 'Dir-AUC-PR']

def format_metric(metric):
    if metric == 'D_order':
        texts = metric.split('_')
        return (r'${main}_\textrm{{{sub}}}$ ($\downarrow$)'.format(main=texts[0], sub=texts[1]))
    if metric in higher_metrics:
        return (r'\textbf{{{metric}}} ($\uparrow$)'.format(metric=metric))
    return (r'\textbf{{{metric}}} ($\downarrow$)'.format(metric=metric))

def plot_time(total_res, outdir=None):
    df = pd.melt(total_res, id_vars=['Method', 'Num_vars'], value_vars=['LogTime'], value_name='Value', var_name='Metric')
    df.Metric = df.Metric.map(format_metric)
    df['Same'] = ''
    palette = dict(zip(df.Method.unique(), colors))
    g = sns.FacetGrid(df, row='Same', col='Metric', sharey=False)
    g.map_dataframe(sns.lineplot, x='Num_vars', y='Value', hue='Method', 
                    marker='o', markersize=7, errorbar=None, palette=palette)
    g.set_xlabels(r'\textbf{\#Variables}', fontsize=10)
    g.set_ylabels(r'$\log$\textbf{(Time) (sec)}')
    g.set_titles('')
    # g.set_titles(r'{col_name}', size=12)
    g.add_legend(label_order=df.Method.unique())
    for ax in g.axes.flat:
        ax.grid(axis='y', linestyle='--')
    g.tight_layout()
    if outdir: g.savefig(f'{outdir}/plot.pdf', dpi=300)
    return g.figure

def plot_multiple_d(total_res, outdir, metrics=['shd', 'training_time']):
    df = pd.melt(total_res, id_vars=['Method', 'Num_vars'], value_vars=metrics, value_name='Value', var_name='Metric')
    df.Metric = df.Metric.map(format_metric)
    df['Same'] = ''
    palette = dict(zip(df.Method.unique(), colors))
    g = sns.FacetGrid(df, row='Same', col='Metric', sharey=False)
    g.map_dataframe(sns.lineplot, x='Num_vars', y='Value', hue='Method', 
                    marker='o', markersize=7, errorbar=('ci', 95), palette=palette)
    g.set_xlabels(r'\textbf{\#Variables}', fontsize=10)
    g.set_ylabels('')
    g.set_titles(r'{col_name}', size=12)
    g.add_legend(label_order=df.Method.unique())
    for ax in g.axes.flat:
        ax.grid(axis='y', linestyle='--')
    g.tight_layout()
    g.savefig(f'{outdir}/plot.pdf', dpi=300)
    return g.figure

def plot_single_d(total_res:pd.DataFrame, outdir, metrics=['shd', 'auc']):
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.melt(total_res, id_vars=['Method', 'Num_vars'], value_vars=metrics, value_name='Value', var_name='Metric')
    num_vars = df.loc[0, 'Num_vars']
    df.Metric = df.Metric.map(format_metric)
    df['Same'] = ''
    palette = dict(zip(df.Method.unique(), colors))
    g = sns.FacetGrid(df, row='Same', col='Metric', sharey=False)
    g.map_dataframe(sns.boxplot, y='Value', x='Method', palette=palette)
    g.set_ylabels('')
    g.set_xlabels('')
    g.set_titles(r'{col_name}', size=12)
    for ax in g.axes.flat:
        ax.grid(axis='y', linestyle='--')
    g.tight_layout()
    g.savefig(f'{outdir}/plot_{num_vars}.pdf')
    return g.figure

if __name__ == '__main__':
    from globals import EXP_DIR
    res = pd.read_pickle(EXP_DIR / 'vi-csl/summary.csv') 
    print(res)
    plot_single_d(total_res=res, outdir= EXP_DIR/'vi-csl', metrics=['shd', 'auc'])