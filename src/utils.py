import contextlib
from copy import deepcopy
import importlib
import logging
logging.getLogger(__name__).setLevel(logging.DEBUG)
import os
import time
from typing import NamedTuple
import torch
import numpy as np
import networkx as nx
import yaml
from tqdm import tqdm
from multiprocessing import Pool
from src.data.dataset import load_dataset, load_real_dataset
from src.data.gen_data import simulate_data
from src.globals import DATA_DIR, REAL_DATA
import pandas as pd
from pathlib import Path

@contextlib.contextmanager
def Timer(name=None, verbose=None):
    if verbose:
        logging.info(f'Starting {name}...')
    start = time.perf_counter()
    timer = NamedTuple('timer', elapsed=str)
    yield timer
    timer.elapsed: float = time.perf_counter() - start
    if verbose:
        logging.info(f'Finished {name} in {timer.elapsed:.3f}s\n')

def read_config(path):
    with open(path, 'r') as f:
        params = yaml.safe_load(f) or {}
    return params

def load_methods(method_names=None):
    if method_names: methods = method_names
    else:  methods = os.listdir('src/methods/')
    print(f'{methods =}')
    models = {}
    for method in methods:
        try:
            module = importlib.import_module(f'src.methods.{method}')
            models[method] = getattr(module, method)
        except Exception as e:
            logging.exception(f'Import method {method} error')
            models[method] = None
    return models

def load_data(args):
    """
    Load data following a specific config.
    Output: 
    data_list: a list of datasets, each consisting of (data_id, Adj, X)
    """
    data_dir_name = args['dataname']
    data_dir = DATA_DIR / data_dir_name
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f'Read data from {data_dir}')

    if data_dir_name in REAL_DATA: 
        data_list = list(load_real_dataset(data_dir_name))
        args['n_dataset'] = len(data_list)
        return data_list

    if 'n_dataset' in args:
        n_dataset = args['n_dataset']
        dataset_id_list = range(n_dataset)
    else:
        dataset_id_list = args['i_dataset'] 
    data_list = [] 
    for data_id in dataset_id_list:
        try: 
            load_test = args.get('load_test', True)
            Adj, X_train, X_test = load_dataset(dataname=data_dir_name, i_dataset=data_id, load_test=load_test)
        except:
            logging.info(f'Data {data_id} is not available.')
            with Timer('Generate data', verbose=True):
                args['seed'] += data_id
                Adj, X_train, X_test = simulate_data(**args)
                dag_file_name = data_dir / f'dag_{data_id}.npy'
                data_train_file_name = data_dir / f'data_train_{data_id}.npy'
                data_test_file_name = data_dir / f'data_test_{data_id}.npy'
                np.save(dag_file_name, Adj)
                np.save(data_train_file_name, X_train)
                np.save(data_test_file_name, X_test)

        data_list.append((data_id, Adj, X_train, X_test))
    return data_list

def gen_data_grid(args):
    """
    Generate a list of data config.
    """
    cfg_list = []
    # For real dataset
    datanames = args.pop('datanames', [])
    for dataname in datanames:
        args['dataname'] = dataname
        cfg_list.append(deepcopy(args))
    if len(cfg_list) > 0:
        return cfg_list

    # For synthetic dataset
    num_nodes = args['d']
    if isinstance(num_nodes, int):
        num_nodes = [num_nodes]
    dag_types = args['dag_type']
    if isinstance(dag_types, str):
        dag_types = [dag_types]
    if 'e_coef' in args:
        edges = [node * args['e_coef'] for node in num_nodes]
    else:
        edges = args['e']
        if isinstance(edges, int): edges = [edges]
    assert len(num_nodes) == len(edges)
    for dag in dag_types:
        for d, e in zip(num_nodes, edges):
            args['d'] = d
            args['e'] = e
            args['dag_type'] = dag
            args['dataname'] = f"{args['dag_type']}_d{args['d']}_e{args['e']}_N{args['N']}_{args['sem_type']}"
            cfg_list.append(deepcopy(args))
    return cfg_list

def pmap(func, tasks, n_jobs, verbose):
    if n_jobs == 1:
        res = list(map(func, tqdm(tasks, disable=not verbose)))
    else:
        with Pool(n_jobs) as p:
            res = list(tqdm(p.imap_unordered(func, tasks), total=len(tasks), disable=not verbose))
    return res

def sample_gumbel(shape, eps=1e-20, device=None):
    eps = torch.tensor(eps, device=device)
    u = torch.rand(shape, device=device)
    u = -torch.log(-torch.log(u + eps) + eps)
    u[np.arange(shape[0]), np.arange(shape[0])] = 0
    return u

def gumbel_sigmoid(logits, temperature, device):
    gumbel_softmax_sample = (logits
                 + sample_gumbel(logits.shape, device=device)
                 - sample_gumbel(logits.shape, device=device))
    y = torch.sigmoid(gumbel_softmax_sample / temperature)
    return y

def check_dag(A):
    g = nx.from_numpy_array(A, create_using=nx.DiGraph)
    return nx.is_directed_acyclic_graph(g)


def to_string(obj):
    if isinstance(obj, np.ndarray):
        return str(obj.tolist())
    elif isinstance(obj, dict):
        return {key: to_string(val) for key, val in obj.items()}
    return obj
    
def save_file(data: pd.DataFrame, outdir: Path, outfile: str):
    outdir.mkdir(parents=True, exist_ok=True)
    data = data.applymap(to_string)
    data.to_csv(outdir / outfile, index=False)

def load_file(path: Path):
    data = pd.read_csv(path, header=0)
    object_cols = data.columns[data.dtypes == 'object']
    data[object_cols] = data[object_cols].applymap(lambda x: np.array(eval(x)))
    # data[object_cols] = data[object_cols].applymap(lambda x: np.array(x))
    return data