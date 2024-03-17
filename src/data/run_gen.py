from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('')
from src.utils import read_config
from src.data.gen_data import simulate_data
from src.globals import DATA_DIR

import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.WARNING)

CUR_DIR = Path(__file__).absolute().parent

def main(args):
    data_dir_name = f"{args['dag_type']}_d{args['d']}_e{args['e']}_N{args['N']}_{args['sem_type']}"
    data_dir = DATA_DIR / data_dir_name
    data_dir.mkdir(parents=True, exist_ok=True)
    n_reps = args['n_reps']
    for i in tqdm(range(n_reps)):
        args['seed'] += i
        Adj_gt, X = simulate_data(**args)
        dag_file_name = data_dir / f'dag_{i}.npy'
        data_file_name = data_dir / f'data_{i}.npy'
        np.save(dag_file_name, Adj_gt)
        np.save(data_file_name, X)

if __name__ == '__main__':
    cfg_path = CUR_DIR / 'config.yml'
    args = read_config(cfg_path)
    main(args)
    