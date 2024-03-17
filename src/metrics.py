import numpy as np
import pandas as pd
from sklearn import metrics

def edge_auroc(pred_edges: np.ndarray, true_edges: np.ndarray):
    if true_edges.min() < 0 or true_edges.max() > 1:
        print("Groundtruth is CPDAG")
        true_edges = np.clip(true_edges, 0, 1)
    fpr, tpr, thresholds = metrics.roc_curve(true_edges.reshape(-1), pred_edges.reshape(-1))
    auc = metrics.auc(fpr, tpr)
    return auc

def edge_apr(pred_edges: np.ndarray, true_edges: np.ndarray):
    if true_edges.min() < 0 or true_edges.max() > 1:
        print("Groundtruth is CPDAG")
        true_edges = np.clip(true_edges, 0, 1)
    return metrics.average_precision_score(true_edges.reshape(-1), pred_edges.reshape(-1))

# https://github.com/xunzheng/notears/blob/master/notears/utils.py
def SHD(A_true: np.ndarray, A_pred: np.ndarray, **kwargs):
    # linear index of nonzeros
    pred = np.flatnonzero(A_pred == 1)
    cond = np.flatnonzero(A_true)
    cond_reversed = np.flatnonzero(A_true.T)
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(A_pred + A_pred.T))
    cond_lower = np.flatnonzero(np.tril(A_true + A_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return shd, len(extra_lower), len(missing_lower), len(reverse)

def exp_shd(posterior_samples: np.ndarray, true_adj: np.ndarray):
    """
    This function is used for debug purpose.
    posterior_samples: (B, N, N) batch of sampled adj.
    true_adj: ground truth
    Output: expected shd, along with additional information: extra, missing, and reverse edges
    """
    B = posterior_samples.shape[0]
    N = true_adj.shape[0]
    def shd_1d(pred_adj, true_adj):
        pred_adj = pred_adj.reshape(N, N)
        shd = SHD(true_adj, pred_adj)
        return shd
    posterior_samples = posterior_samples.reshape(B, -1)
    results = np.apply_along_axis(func1d=shd_1d, axis=1, arr=posterior_samples, true_adj=true_adj,)
    # print(f'{results=}')
    return results.mean(axis=0)

def cal_precision_recall(W_p, W_true):
    """
    Parameters
    ----------
    W_p: pd.DataDrame
        [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
    W_true: pd.DataDrame
        [d, d] ground truth graph, {0, 1}.
    
    Return
    ------
    precision: float
        TP/(TP + FP)
    recall: float
        TP/(TP + FN)
    F1: float
        2*(recall*precision)/(recall+precision)
    """

    assert(W_p.shape==W_true.shape and W_p.shape[0]==W_p.shape[1])
    TP = (W_p + W_true).applymap(lambda elem:1 if elem==2 else 0).sum(axis=1).sum()
    TP_FP = W_p.sum(axis=1).sum()
    TP_FN = W_true.sum(axis=1).sum()
    precision = TP/TP_FP
    recall = TP/TP_FN
    F1 = 2*(recall*precision)/(recall+precision)
    
    return precision, recall, F1

def exp_f1(posterior_samples: np.ndarray, true_adj: np.ndarray):
    """
    This function is used for debug purpose.
    posterior_samples: (B, N, N) batch of sampled adj.
    true_adj: ground truth
    Output: expected shd, along with additional information: extra, missing, and reverse edges
    """
    B = posterior_samples.shape[0]
    N = true_adj.shape[0]
    def f1_single(pred_adj, true_adj):
        *f1, _ = metrics.precision_recall_fscore_support(y_true=true_adj, y_pred=pred_adj, average='binary')
        return f1
    posterior_samples = posterior_samples.reshape(B, -1)
    true_adj = true_adj.reshape(1, -1).squeeze()
    results = np.apply_along_axis(func1d=f1_single, axis=1, arr=posterior_samples, true_adj=true_adj,)
    return results.mean(axis=0)

if __name__ == '__main__':
    rng = np.random.RandomState(0) # use this as random function
    B = 3
    N = 5
    posterior_samples = rng.randint(2, size=(B, N, N))
    true_adj = posterior_samples[0, :, :]
    # Test Exp shd
    print('Numpy function approach:')
    print(exp_shd(posterior_samples=posterior_samples, true_adj=true_adj))
    print(exp_f1(posterior_samples=posterior_samples, true_adj=true_adj))

    # print('For loop approach: ') 
    # for pred_adj in posterior_samples[:]:
    #     print(SHD(true_adj, pred_adj))

    # print('SHD calculation with matrix manipulation')
    # print(fast_exp_shd(posterior_samples, true_adj))