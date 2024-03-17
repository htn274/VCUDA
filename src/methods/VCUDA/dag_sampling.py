import sys
sys.path.append('')

from src.utils import Timer, gumbel_sigmoid, sample_gumbel
from src.architectures.linear_sequential import linear_sequential

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import logging
from abc import ABC, abstractmethod


# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class DAG_sampler_Abstract(ABC, nn.Module):
    def __init__(self, n_nodes, temp_w=1.0, temp_p=1.0, seed=0, hard=True, device="cuda"):
        super().__init__()
        set_seed(seed)
        self.n_nodes = n_nodes
        self.temp_w  = temp_w
        self.temp_p = temp_p
        self.seed = seed
        self.hard = hard
        if torch.cuda.is_available():
            self.device = device
        else:
            self.device = 'cpu'
    
    @abstractmethod
    def sample_W(self):
        raise NotImplementedError
    
    @abstractmethod
    def sample_P(self):
        raise NotImplementedError

    def sample(self):
        W = self.sample_W()
        P = self.sample_P()
        A = W * P
        if self.hard:  # Straight-through estimator
            A = torch.round(A).detach() + A - A.detach()
        return A

    

class DAG_sampler(DAG_sampler_Abstract):
    def __init__(self, n_nodes, temp_w=1.0, temp_p=1.0, seed=0, hard=True, device="cuda"):
        super().__init__(n_nodes=n_nodes, temp_w=temp_w, temp_p=temp_p, seed=seed, hard=hard, device=device)
        e = torch.zeros(self.n_nodes, self.n_nodes, device=self.device)
        nn.init.uniform_(e)
        torch.diagonal(e).fill_(-300)
        self.edge_log_prob = nn.Parameter(e)
        mean_p_score = torch.zeros(self.n_nodes, device=self.device)
        nn.init.normal_(mean_p_score)
        self.mean_p_score = nn.Parameter(mean_p_score)
        self.scale_p_score = nn.Parameter(torch.zeros(1, device=self.device))
        
    def sample_W(self):
        """
        Sample a binary mask based on edge_log_prob params
        """
        p_log = F.logsigmoid(torch.stack((self.edge_log_prob, -self.edge_log_prob)))
        # print(f'{p_log=}')
        W = F.gumbel_softmax(p_log, tau=self.temp_w, hard=True, dim=0)[0]
        return W

    def sample_P(self):
        """
        Sample a permutation matrix: 
        1. Sample a list of potential scores theta 
        2. Calculate the pairwise diff of theta 
        3. Apply tempered sigmoid
        ====
        Return a permutation matrix in {0,1}
        """
        norm = torch.distributions.Normal(self.mean_p_score, torch.exp(self.scale_p_score) * torch.ones_like(self.mean_p_score))
        p_score = norm.rsample((1, )).reshape(1, self.n_nodes) # (1, self.n_nodes)
        # logging.info(f'{p_score=}')
        p_score_T = torch.transpose(p_score, 0, 1)
        pairwise_diff = p_score - p_score_T
        # logging.info(f'{pairwise_diff=}')
        p = torch.sigmoid((pairwise_diff)/ self.temp_p)
        id_mat = torch.eye(self.n_nodes, device=p.device)
        p = p * (1 - id_mat)
        return p

class DAG_sampler_WP(DAG_sampler_Abstract):
    """
    q(P, W) = q(P|W)q(W)
    """
    def __init__(self, n_nodes, temp_w=1.0, temp_p=1.0, seed=0, hard=True, device="cpu"):
        super().__init__(n_nodes=n_nodes, temp_w=temp_w, temp_p=temp_p, seed=seed, hard=hard, device=device)
        e = torch.Tensor(self.n_nodes, self.n_nodes)
        nn.init.uniform_(e)
        torch.diagonal(e).fill_(-300)
        self.edge_log_prob = nn.Parameter(e)

        self.p_model = linear_sequential(input_dims= e.shape,
                                         output_dim= n_nodes + 1,
                                         hidden_dims=[16])

    def sample_W(self):
        p_log = F.logsigmoid(torch.stack((self.edge_log_prob, -self.edge_log_prob)))
        # print(f'{p_log=}')
        W = F.gumbel_softmax(p_log, tau=self.temp_w, hard=True, dim=0)[0]
        return W

    def sample_P(self, W):
        n_nodes = W.shape[0]
        W = torch.flatten(W)  # (n_nodes*n_nodes)
        W = torch.unsqueeze(W, dim=0)  # (1, n_nodes*n_nodes)
        p_params = self.p_model(W).squeeze(dim=0)
        self.mean_p_score = p_params[:n_nodes]
        self.scale_p_score = p_params[-1]
        norm = torch.distributions.Normal(self.mean_p_score, torch.exp(self.scale_p_score))
        p_score = norm.rsample((1, )).reshape(1, self.n_nodes) # (1, self.n_nodes)
        logging.info(f'{p_score=}')
        p_score_T = torch.transpose(p_score, 0, 1)
        pairwise_diff = p_score - p_score_T
        logging.info(f'{pairwise_diff=}')
        p = torch.sigmoid((pairwise_diff)/ self.temp_p)
        p = p * (1 - torch.eye(self.n_nodes))
        return p

    def sample(self):
        W = self.sample_W()
        P = self.sample_P(W)
        A = W * P
        if self.hard:  # Straight-through estimator
            A = torch.round(A).detach() + A - A.detach()
        return A


class DAG_sampler_PW(DAG_sampler_Abstract):
    def __init__(self, n_nodes, temp_w=1.0, temp_p=1.0, seed=0, hard=True, device="cpu"):
        super().__init__(n_nodes=n_nodes, temp_w=temp_w, temp_p=temp_p, seed=seed, hard=hard, device=device)
        # e = torch.Tensor(self.n_nodes, self.n_nodes, device=device)
        # nn.init.uniform_(e)
        # torch.diagonal(e).fill_(-300)
        # self.edge_log_prob = nn.Parameter(e)
        mean_p_score = torch.zeros(self.n_nodes, device=device)
        nn.init.normal_(mean_p_score)
        self.mean_p_score = nn.Parameter(mean_p_score)
        self.scale_p_score = nn.Parameter(torch.zeros(1, device=device))
        self.W_model = linear_sequential(input_dims= n_nodes,
                                         output_dim= n_nodes*n_nodes, 
                                         hidden_dims=[128])
        
    def sample_W(self, p):
        """
        Sample a binary mask based on edge_log_prob params
        """
        n_nodes = p.shape[1]
        # p = torch.squeeze(p, dim=0)
        edge_log_prob = self.W_model(p) - 1
        self.edge_log_prob = edge_log_prob.reshape(shape=(n_nodes, n_nodes))
        # torch.diagonal(self.edge_log_prob).fill_(-300)
        p_log = F.logsigmoid(torch.stack((self.edge_log_prob, -self.edge_log_prob)))
        # print(f'{p_log=}')
        W = F.gumbel_softmax(p_log, tau=self.temp_w, hard=True, dim=0)[0]
        return W

    def sample_P(self):
        """
        Sample a permutation matrix: 
        1. Sample a list of potential scores theta 
        2. Calculate the pairwise diff of theta 
        3. Apply tempered sigmoid
        ====
        Return a permutation matrix in {0,1}
        """
        norm = torch.distributions.Normal(self.mean_p_score, torch.exp(self.scale_p_score) * torch.ones_like(self.mean_p_score))
        p_score = norm.rsample((1, )).reshape(1, self.n_nodes) # (1, self.n_nodes)
        logging.info(f'{p_score=}')
        p_score_T = torch.transpose(p_score, 0, 1)
        pairwise_diff = p_score - p_score_T
        logging.info(f'{pairwise_diff=}')
        p = torch.sigmoid((pairwise_diff)/ self.temp_p)
        id_mat = torch.eye(self.n_nodes).to(device=p.device)
        p = p * (1 - id_mat)
        return p, p_score

    def sample(self):
        P, p_score = self.sample_P()
        W = self.sample_W(p_score)
        A = W * P
        if self.hard:  # Straight-through estimator
            A = torch.round(A).detach() + A - A.detach()
        return A

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dag_sampler = DAG_sampler_PW(n_nodes=5, hard=True, seed=13, temp_p=0.1, temp_w=0.5)
    print(dag_sampler.sample())
    # for _ in range(4):
    #     A = dag_sampler.sample().detach()
    #     print(f'{A=}')
        # Check DAG
        # G = nx.DiGraph(A.numpy())
        # assert nx.is_directed_acyclic_graph(G)
