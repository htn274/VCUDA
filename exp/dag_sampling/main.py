from argparse import ArgumentParser
from copy import deepcopy
import itertools
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import trange
sys.path.append('')
from src.methods.VCUDA.dag_sampling import DAG_sampler
from src.methods.DDS.probabilistic_dag import ProbabilisticDAG
from src.utils import pmap, read_config, save_file 
from src.data.gen_data import gen_dag
from src.metrics import edge_apr, edge_auroc

CUR_DIR = Path(__file__).parent

def gen_data_grid(data_cfg):
    num_nodes = data_cfg['d']
    if isinstance(num_nodes, int):
        num_nodes = [num_nodes]
    dag_types = data_cfg['dag_type']
    if isinstance(dag_types, str):
        dag_types = [dag_types]
    if 'e_coef' in data_cfg:
        edges = [node * data_cfg['e_coef'] for node in num_nodes]
    else:
        edges = data_cfg['e']
        if isinstance(edges, int): edges = [edges]

    cfg_list = []
    cfg = deepcopy(data_cfg)
    for dag in dag_types:
        for d, e in zip(num_nodes, edges):
            cfg['d'] = d
            cfg['e'] = e
            cfg['dag_type'] = dag
            cfg_list.append(deepcopy(cfg))
    return cfg_list

def gen_model_grid(model_cfg):
    lr_list = model_cfg['lr']
    epoch_list = model_cfg['epochs']
    cfg = deepcopy(model_cfg)
    cfg_list = []
    for lr, epochs in zip(lr_list, epoch_list):
        cfg['lr'] = lr
        cfg['epochs'] = epochs
        cfg_list.append(deepcopy(cfg))
    return cfg_list

def gen_data(data_cfg):
    data = []
    rng = np.random.RandomState(data_cfg['seed'])
    for _ in range(data_cfg['n_dataset']):
        data_cfg['seed'] = rng.randint(2**10)
        dag = gen_dag(**data_cfg)
        data.append(dag)
    return data
    

def train(model:DAG_sampler, gt: np.array, lr:int, epochs:int, device:str):
    model = model.to(device=device)
    gt = torch.Tensor(gt).to(device=device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    pbar = trange(epochs, disable=True)
    for i in pbar:
        dag_sample = model.sample()
        loss = ((dag_sample - gt)**2).sum()
        optim.zero_grad()
        loss.backward() 
        optim.step()
        pbar.set_postfix(loss=f'{loss.item():.2f}')
    return model

def sample(model, n_sample):
    model.eval()
    sample_dags = []
    for _ in range(n_sample):
        sample_dag = model.sample().detach().cpu().numpy()
        sample_dag = np.expand_dims(sample_dag, axis=0)
        sample_dags.append(sample_dag)
    sample_dags = np.concatenate(sample_dags, axis=0)
    pred_prob = np.mean(sample_dags.reshape(sample_dags.shape[0], -1), axis=0)
    return pred_prob 

def process(task):
    task_id, gt, n_nodes, dag_type, model_cfg = task
    try:
        model = DAG_sampler(n_nodes=n_nodes, device=model_cfg['device'])
        # model = ProbabilisticDAG(n_nodes=n_nodes, order_type='topk')
        model = train(model, gt, lr=model_cfg['lr'], epochs=model_cfg['epochs'], device=model_cfg['device'])
        pred_prob = sample(model, n_sample=100)
        pr = edge_apr(pred_prob, gt)
        auc = edge_auroc(pred_prob, gt)   
    except Exception as e:
        logging.error(f'Error at task={task_id}', exc_info=True)
    output = dict(iDataset=task_id, Num_vars=n_nodes, DAG_type=dag_type, PR=pr, AUC=auc, pred_prob=pred_prob)
    return output


def main(data_cfg, model_cfg):
    data = gen_data(data_cfg)
    n_nodes = data_cfg['d']
    tasks = [(i, data[i], n_nodes, data_cfg['dag_type'], model_cfg) for i in range(len(data))]
    res = pmap(process, tasks, n_jobs=5, verbose=True)
    return res 

def print_results(df, metrics):
    res = df.groupby(['DAG_type', 'Num_vars'], axis=0)[metrics].agg(['mean', 'std'])
    print(res)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cfg_file', type=str, default= 'cfg.yml',
                            help='Config path for running experiment')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    cfg_path = CUR_DIR / f'{args.cfg_file}'
    cfg = read_config(cfg_path)  
    if args.run: 
        total_res = []
        data_cfg_list = gen_data_grid(cfg['data'])
        model_cfg_list = gen_model_grid(cfg['model'])
        for data_cfg, model_cfg in itertools.product(data_cfg_list, model_cfg_list):
            print(f'{data_cfg=}-{model_cfg=}')
            res = main(data_cfg=data_cfg, model_cfg=model_cfg)
            total_res += res
        total_res = pd.DataFrame(total_res)
        save_file(data=total_res, outdir= CUR_DIR, outfile='res.csv')
    total_res = pd.read_csv(CUR_DIR / 'res.csv')
    print_results(total_res, metrics=['PR', 'AUC'])