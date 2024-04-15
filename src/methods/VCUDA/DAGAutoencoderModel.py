import sys

sys.path.append('')
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim.adam
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.grads import grad_norm

from src.methods.VCUDA.dag_sampling import DAG_sampler, DAG_sampler_PW, DAG_sampler_WP
from src.architectures.linear_sequential import linear_sequential
from src.metrics import exp_shd, edge_auroc

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.basicConfig()

class FunctionalModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 64, 64]):
        # The model is linear when hidden_dims = []
        super().__init__()
        self.mask = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.autoencoders = nn.ModuleList([linear_sequential(input_dims=self.input_dim,
                                                                 hidden_dims=self.hidden_dims,
                                                                 output_dim=self.output_dim,
                                                                 k_lipschitz=None) for i in range(self.input_dim)])
        # print(f'{self.autoencoders=}')

    def update_mask(self, new_mask):
        self.mask = new_mask

    def forward(self, X):
        masked_duplicate_X = X.unsqueeze(1).expand([-1, self.input_dim, -1]) * self.mask.T.unsqueeze(0) # [batch_size, input_dim, input_dim]
        X_pred = torch.zeros_like(X) # [batch_size, input_dim]
        for i in range(self.input_dim):
            X_pred[:, i] = self.autoencoders[i](masked_duplicate_X[:, i, :]).squeeze()
        return X_pred

DAG_sampler_dict = {
    'WP': DAG_sampler_WP,
    'PW': DAG_sampler_PW
}
class DAGAutoencoderModel(LightningModule):
    def __init__(self, input_dim, output_dim, hidden_dims,
                 temp_w, temp_p, hard,
                 lr_f, lr_dag, lambda_p_score,lambda_e, 
                 prior_edge_prob, prior_p_scale,
                 max_epochs, seed_model, verbose, gt_adj, 
                 l2_f=0.0, factorize_type=None, 
                 use_flow=False, flow_layers=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.f_model = FunctionalModel(input_dim=input_dim, 
                                       output_dim=output_dim, 
                                       hidden_dims=hidden_dims)
        if factorize_type is None:
            self.dag_sampler = DAG_sampler(n_nodes=input_dim, 
                                       temp_w=temp_w,
                                       temp_p=temp_p,
                                       seed=seed_model, hard=hard, device=self.device)
        else:
            self.dag_sampler = DAG_sampler_dict[factorize_type](
                n_nodes = input_dim, 
                temp_w=temp_w,
                temp_p=temp_p,
                seed=seed_model, 
                hard=hard,
            )
        self.automatic_optimization = False
        
    def forward(self, X):
        X_pred =  self.f_model(X)
        return X_pred

    def cal_kl_term(self):
        num_nodes = self.hparams['input_dim']
        edge_logits = torch.flatten(self.dag_sampler.edge_log_prob)
        # p_params = self.dag_sampler.mean_p_score
        prior_prob = torch.ones_like(edge_logits) * self.hparams['prior_edge_prob']
        prior_edge_dist = torch.distributions.Bernoulli(probs=prior_prob)
        prior_p_score_dist = torch.distributions.Normal(loc=torch.zeros(num_nodes, device=self.device), scale=self.hparams['prior_p_scale'] * torch.ones(num_nodes, device=self.device))
        # Approximate by variational families
        # Calculate KL edge
        edge_dist = torch.distributions.Bernoulli(logits=edge_logits)
        kl_edge = torch.distributions.kl_divergence(edge_dist, prior_edge_dist).sum()
        # calculate KL p_score
        p_score_dist = torch.distributions.Normal(loc=self.dag_sampler.mean_p_score, scale=torch.exp(self.dag_sampler.scale_p_score) * torch.ones(num_nodes, device=self.device))
        kl_p_score = torch.distributions.kl_divergence(p_score_dist, prior_p_score_dist).sum()
        kl_term = kl_edge + kl_p_score
        return kl_term

    def loss(self, X_pred, X):
        N, d = X.shape
        mse = nn.MSELoss(reduction='sum')
        reconstruction_loss = mse(X_pred, X)
        kl_term = self.cal_kl_term()
        ELBO_loss = (reconstruction_loss + kl_term)/d
        return ELBO_loss, reconstruction_loss, kl_term

    def update_mask(self, new_mask=None):
        if new_mask is None:
            new_mask = self.dag_sampler.sample()
        self.f_model.update_mask(new_mask)

    def logging(self):
        logger = self.logger.experiment
        for name, p in self.named_parameters():
            logger.add_histogram(name, p, self.global_step)

    def init_f_model(self, X, max_iters=100):
        # print('Init f_model')
        d = self.hparams['input_dim']
        full_mask = torch.ones(d, d) * (1 - torch.eye(d))
        optimizer = torch.optim.Adam(self.f_model.parameters(), lr=5.e-3)
        self.update_mask(full_mask)
        for _ in range(max_iters):
            # for X_batch in train_loader:
            X_hat = self.f_model(X)
            loss = ((X - X_hat)**2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def training_step(self, batch, batch_idx):
        opt_f, opt_dag = self.optimizers(use_pl_optimizer=False)
        X = batch
        self.update_mask()
        X_pred = self.f_model(X)
        opt_f.zero_grad()
        opt_dag.zero_grad()
        loss, mse, kl_term = self.loss(X_pred, X)
        self.manual_backward(loss)
        opt_f.step()
        opt_dag.step()
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_mse', mse, on_epoch=True)
        self.log('train_kl', kl_term, on_epoch=True)
        # self.log('train_kl_p', kl_p, on_epoch=True)
        # Logging
        # self.logging()
        return loss

    def on_train_epoch_start(self) -> None:
        sch_dag = self.lr_schedulers()
        # sch_f.step()
        sch_dag.step()
        return super().on_train_epoch_start()

    def on_train_end(self) -> None:
        if self.hparams['verbose']:
            print(f'Learned P-score params: mean={self.dag_sampler.mean_p_score.cpu().detach().numpy()}-sig={self.dag_sampler.scale_p_score.item()}')
            e_shd, extra, missing, reverse, auc = self.CSL_evaluation()
            print(f'\n{e_shd=}, {extra=}, {missing=}, {reverse=}, {auc=}')
            self.logger.log_metrics({
                'E_SHD':  e_shd,
                'E_Extra': extra,
                'E_Missing': missing, 
                'E_Reverse': reverse, 
                'AUC': auc
            })

    def on_before_optimizer_step(self, optimizer):
        """
        Log norm of gradients
        """
        if self.hparams['verbose']:
            norms = grad_norm(self, norm_type=2)
            # print(f'{norms=}')
            self.log_dict(norms, logger=True)

    def CSL_evaluation(self):
        adj_samples = self.sample(n_samples=100).cpu().detach().numpy()
        gt_adj = self.hparams['gt_adj']
        pred_prob = np.mean(adj_samples.reshape(adj_samples.shape[0], -1), axis=0)
        e_shd, extra, missing, reverse = exp_shd(adj_samples, gt_adj)
        auc = edge_auroc(pred_prob, gt_adj)
        return e_shd, extra, missing, reverse, auc

    def on_validation_epoch_start(self) -> None:
        # print('Validation epoch start')
        if self.hparams['verbose']:
            e_shd, extra, missing, reverse, auc = self.CSL_evaluation() 
            print(f'\n{e_shd=}, {extra=}, {missing=}, {reverse=}, {auc=}')
        self.update_mask()
        return super().on_validation_epoch_start()

    def validation_step(self, batch, batch_idx):
        X = batch
        X_pred = self.f_model(X)
        loss, mse, *_ = self.loss(X_pred, X)
        self.log('val_loss', loss, on_epoch=True, reduce_fx=torch.mean, prog_bar=True)
        self.log('val_mse', mse, on_epoch=True, reduce_fx=torch.mean)

    def on_test_epoch_start(self) -> None:
        self.update_mask()
        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx):
        X = batch
        X_pred = self.f_model(X)
        mse_loss = ((X_pred - X)**2).mean()
        # self.log('test_loss', mse_loss, on_epoch=True, reduce_fx=torch.mean, prog_bar=False)
        self.log_dict({'loss': mse_loss})

    def configure_optimizers(self):
        optimizer_f = torch.optim.Adam(self.f_model.parameters(), lr=self.hparams['lr_f'], weight_decay=self.hparams['l2_f'])
        optimizer_dag = torch.optim.Adam(self.dag_sampler.parameters(), lr=self.hparams['lr_dag'])
        scheduler_f = None
        # scheduler_f = torch.optim.lr_scheduler.MultiStepLR(optimizer_f, milestones=[500], gamma=0.1, verbose=False)
        scheduler_dag = torch.optim.lr_scheduler.MultiStepLR(optimizer_dag, milestones=[500], gamma=0.1, verbose=False)

        return ([optimizer_f, optimizer_dag], [scheduler_dag])

    def sample(self, n_samples=1):
        A = [1 * (self.dag_sampler.sample().unsqueeze(0) > 0.5) for _ in range(n_samples)]
        A = torch.concat(A) #(n_samples, N, N)
        return A

if __name__ == '__main__':
    d = 3
    N = 10
    X = torch.randn(N, d)
    enc = FunctionalModel(input_dim=d, output_dim=1, hidden_dims=[]) 
    mask = torch.zeros(d, d)
    enc.update_mask(new_mask=mask)
    X_pred = enc(X)
    print(f'{X_pred}')
    mse = ((X - X_pred)**2).mean()
    print(f'{mse=}')