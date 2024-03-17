import sys 
sys.path.append('')
from src.globals import DATA_DIR
import numpy as np
import torch 
import glob

def load_dataset(dataname, i_dataset, load_test=True, **kwargs):
    data_dir = DATA_DIR / dataname
    adjacency = np.load(data_dir / f'dag_{i_dataset}.npy')
    X_train = np.load(data_dir / f'data_train_{i_dataset}.npy')
    if load_test:
        X_test = np.load(data_dir / f'data_test_{i_dataset}.npy')
    else:
        X_test = None
    return adjacency, X_train, X_test

def load_real_dataset(dataname):
    data_dir = DATA_DIR / dataname
    dag_files = glob.glob(f'{data_dir}/dag_*.npy')
    dag_list = map(np.load, dag_files)
    data_files = glob.glob(f'{data_dir}/data_*.npy')
    data_list = map(np.load, data_files)
    data_id_list = range(len(dag_files))
    data_test_list = [None] * len(dag_files)
    return zip(data_id_list, dag_list, data_list, data_test_list)

def get_data_loader(X: np.array, batch_size:int, num_worker:int, split: list, seed_dataset:int, scaled:bool=False, **kwargs):
    assert np.sum(split) == 1.0
    rng = np.random.RandomState(seed_dataset)
    n_data = X.shape[0]
    indices = list(range(n_data))
    # split0, split1 = int(n_data * split[0]), int(n_data * (split[0] + split[1]))
    split0 = int(n_data * split[0])
    rng.shuffle(indices)
    # Train split
    train_indices = indices[:split0]
    X_train = X[train_indices]
    mean, std = torch.mean(X_train, 0, keepdim=True), torch.std(X_train, 0, keepdim=True)
    if scaled: X_train = (X_train - mean)/std
    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    # Validation split
    val_indices = indices[split0:]
    if len(val_indices) == 0:
        val_indices = train_indices[:100]
    X_val = X[val_indices]
    if scaled: X_val = (X_val - mean)/std
    val_loader = torch.utils.data.DataLoader(X_val, batch_size=1024, shuffle=False, num_workers=num_worker)
    # Test split
    # test_indices = indices[split1:]
    # X_test = X[test_indices]
    # test_loader = torch.utils.data.DataLoader(X_test, batch_size=1024, shuffle=False, num_workers=num_worker)
    return train_loader, val_loader

if __name__ == '__main__':
    print(list(load_real_dataset(dataname='sachs')))
    # seed = 0
    # X, adj = load_dataset(
    #         dataname='ER_d10_e20_N1000_gp',
    #         i_dataset=1,
    # )
    # with torch.random.fork_rng():
    #     torch.random.manual_seed(seed)
    #     train_loader, val_loader, test_loader = get_data_loader(
    #         X = X,
    #         batch_size=100,
    #         num_worker=4,
    #         split=[0.8, 0.1, 0.1],
    #         seed_dataset=seed
    #     )
    #     X_train = next(iter(train_loader))
    #     print(X_train.shape)