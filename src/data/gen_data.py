from castle.datasets import IIDSimulation, DAG

def gen_dag(d, e, dag_type, seed=None, weight_range=(0.5, 2), **kwargs):
    dag_generators = {
        'ER': DAG.erdos_renyi,
        'SF': DAG.scale_free
    }
    dag_gen = dag_generators[dag_type]
    W = dag_gen(n_nodes=d, 
                n_edges=e, 
                seed=seed, 
                weight_range=weight_range)
    Adj = (W != 0).astype(int)
    return Adj

def simulate_data(N, d, e, 
                dag_type, method, sem_type, 
                noise_scale, weight_range=(0.5, 2), seed=None, **kwargs):
    """
    N: number of samples
    d: number of nodes
    e: number of expected edges
    dag_type: 'ER' or 'SF'
    method: 'linear' or 'nonlinear'
    sem_type: for linear, this is the noise type (gauss, exp, gumbel, uniform)
            for nonliear, this is the form of function (quaradtic, gp, gp-add, mlp, mim)
    noise_scale: scale of noise
    seed: for randomness
    ----
    return the adjacency matrix and the generated data
    """
    dag_generators = {
        'ER': DAG.erdos_renyi,
        'SF': DAG.scale_free
    }
    dag_gen = dag_generators[dag_type]
    Adj = dag_gen(n_nodes=d, 
                n_edges=e, 
                seed=seed, 
                weight_range=weight_range)
    # print(f'Adj = {Adj}')
    train = IIDSimulation(Adj, n=N, 
                      method=method,
                      sem_type=sem_type,
                      noise_scale=noise_scale)
    test = IIDSimulation(Adj, n=1000, method=method, sem_type=sem_type, noise_scale=noise_scale)
    return train.B, train.X, test.X 

if __name__ == '__main__':
    Adj, X_train, X_test = simulate_data(N=10, d=5, e=5, 
                dag_type='ER', method='linear', sem_type='gauss', 
                noise_scale=1.0, weight_range=(0.5, 2), seed=None)
    print(X_train == X_test)
    

