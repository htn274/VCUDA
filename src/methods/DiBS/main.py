from dibs.inference import MarginalDiBS, JointDiBS
from dibs.target import make_nonlinear_gaussian_model, make_graph_model, make_linear_gaussian_equivalent_model
from dibs.models import LinearGaussian, BGe, DenseNonlinearGaussian
from dibs.metrics import neg_ave_log_marginal_likelihood
import jax.random as random
import jax.numpy as jnp
import numpy as np

import sys 
sys.path.append('')
from src.utils import Timer

# simulate some data
# key, subk = random.split(key)
# data, graph_model, likelihood_model = make_linear_gaussian_equivalent_model(key=subk, n_vars=20)

# sample 10 DAG and parameter particles from the joint posterior
# dibs = JointDiBS(x=data.x, interv_mask=None, graph_model=graph_model, likelihood_model=likelihood_model)
# key, subk = random.split(key)
# gs, thetas = dibs.sample(key=subk, n_particles=10, steps=1000)
# negll = neg_ave_log_marginal_likelihood(dist=dist, x=data.x_ho,
#                 eltwise_log_marginal_likelihood=dibs.eltwise_log_marginal_likelihood_observ)

def DiBS(X, args_model, args_data, **kwargs):
    key = random.PRNGKey(args_model.get('seed_model', 0))
    X_train, X_test = X
    X_train = jnp.array(X_train)
    _, n_vars = X_train.shape
    graph_prior_str = args_data['dag_type'].lower()
    # default params
    graph_model = make_graph_model(n_vars=n_vars, graph_prior_str=graph_prior_str)
    likelihood_model = BGe(n_vars=n_vars)
    dibs = MarginalDiBS(x=X_train, interv_mask=None, graph_model=graph_model, likelihood_model=likelihood_model)
    key, subk = random.split(key)
    with Timer(name='Training time', verbose=False) as timer:
        posterior_adj = dibs.sample(key=subk, n_particles=args_model['n_posterior_samples'], steps=args_model['steps'])
    training_time = timer.elapsed
    posterior_adj = np.asarray(posterior_adj)
    pred_prob = np.mean(posterior_adj.reshape(posterior_adj.shape[0], -1), axis=0)
    dist = dibs.get_empirical(posterior_adj)
    test_mse = -1
    if X_test is not None:
        X_test = jnp.array(X_test)
        negll = neg_ave_log_marginal_likelihood(dist=dist, x=X_test,
                    eltwise_log_marginal_likelihood=dibs.eltwise_log_marginal_likelihood_observ) 
        test_mse = negll / np.prod(X_test.shape)
    return posterior_adj, test_mse, pred_prob, training_time