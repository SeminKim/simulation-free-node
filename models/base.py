import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchdiffeq import odeint, odeint_adjoint
from utils import append_dims
from typing import Dict, List
from .dynamics import get_dynamics

MAX_NUM_STEPS = 1000  # Maximum number of steps for ODE solver


class AppendRepeat(nn.Module):
    '''
    append and apply repeat {rep_dims}
    e.g. rep_dims=(H,W) for (B, C) -> (B, C, H, W)
    '''

    def __init__(self, rep_dims):
        super(AppendRepeat, self).__init__()
        self.rep_dims = rep_dims

    def forward(self, x):
        ori_dim = x.ndim
        for _ in range(len(self.rep_dims)):
            x = x.unsqueeze(-1)
        return x.repeat(*[1 for _ in range(ori_dim)], *self.rep_dims)


class ODEBlock(nn.Module):
    """Solves ODE defined by odefunc.

    Parameters
    ----------
    device : torch.device

    odefunc : ODEFunc instance or anode.conv_models.ConvODEFunc instance
        Function defining dynamics of system.

    is_conv : bool
        If True, treats odefunc as a convolutional model.

    tol : float
        Error tolerance.

    adjoint : bool
        If True calculates gradient with adjoint method, otherwise
        backpropagates directly through operations of ODE solver.
    """

    def __init__(self, device, odefunc, is_conv=False, tol=1e-3, adjoint=False):
        super(ODEBlock, self).__init__()
        self.adjoint = adjoint
        self.device = device
        self.is_conv = is_conv
        self.odefunc = odefunc
        self.tol = tol

    def forward(self, x, eval_times=None, method='dopri5'):
        """Solves ODE starting from x.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, self.odefunc.data_dim)

        eval_times : None or torch.Tensor
            If None, returns solution of ODE at final time t=1. If torch.Tensor
            then returns full ODE trajectory evaluated at points in eval_times.
        """
        self.odefunc.nfe = 0

        if eval_times is None:
            integration_time = torch.tensor([0, 1.0]).float().type_as(x)
        else:
            integration_time = eval_times.type_as(x)

        x_aug = x

        options = None if method == 'euler' else {'max_num_steps': MAX_NUM_STEPS}
        odeint_fn = odeint_adjoint if self.adjoint else odeint

        out = odeint_fn(self.odefunc, x_aug, integration_time,
                        rtol=self.tol, atol=self.tol, method=method,
                        options=options)

        if eval_times is None:
            return out[1]  # Return only final time
        else:
            return out

    def trajectory(self, x, timesteps, method='dopri5'):
        """Returns ODE trajectory.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, self.odefunc.data_dim)

        timesteps : int
            Number of timesteps in trajectory.
        """
        if isinstance(timesteps, int):
            integration_time = torch.linspace(0., 1., timesteps)
        elif isinstance(timesteps, torch.Tensor):
            integration_time = timesteps
        else:
            raise ValueError('timesteps should be int or torch.Tensor')
        return self.forward(x, eval_times=integration_time, method=method)


class BaseModel(L.LightningModule):
    '''
    A base model for training and inference.
    '''

    def __init__(self, method='ours', force_zero_prob=0.1, metric_type='accuracy', label_scaler=None, scheduler='none',
                 lr=1e-4, wd=0., task_criterion='ce', dynamics=None, adjoint=False, label_ae_noise=0.0,
                 total_steps=None, resume_path=None, resume_keys=(), freeze_keys=(),
                 **kwargs):

        super().__init__()
        # unused param warning
        if kwargs:
            print(f'[!] unused kwargs: {kwargs}')

        self.method = method
        self.force_zero_prob = force_zero_prob
        self.metric_type = metric_type
        self.label_scaler = label_scaler
        self.scheduler = scheduler
        self.lr = lr
        self.wd = wd
        self.adjoint = adjoint
        self.label_ae_noise = label_ae_noise
        self.total_steps = total_steps
        self.resume_path = resume_path
        self.resume_keys = resume_keys
        self.freeze_keys = freeze_keys

        if task_criterion == 'mse':
            self.task_criterion = torch.nn.MSELoss()
        elif task_criterion == 'ce':
            self.task_criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f'Unknown task_criterion: {task_criterion}')

        self.label_ae_criterion = torch.nn.MSELoss()

        self.dynamics = dynamics
        if isinstance(dynamics, str):
            self.dynamics = get_dynamics(dynamics, return_v=False)

        if self.metric_type == 'accuracy':
            self.best_metric = {
                f'val/best_acc': 0.,
                f'test/best_acc': 0.,
            }
        elif self.metric_type == 'rmse':
            self.best_metric = {
                f'val/best_rmse': np.inf,
                f'test/best_rmse': np.inf,
            }
        else:
            raise ValueError(f'Unknown metric type: {self.metric_type}')

        # should be defined in the child class
        self.in_projection = None
        self.out_projection = None
        self.label_projection = None

    def setup(self, stage):
        '''
        Delete the label_projection if not needed.
        This is to avoid unused param error with DDP.
        Additionally load from ckpt and freeze it.
        '''
        if self.method != 'ours':
            self.label_projection = nn.Identity()
        self.load_ckpt()
        if stage == 'fit':
            self.freeze_weights()
        super().setup(stage)

    def load_ckpt(self):
        '''
        Load checkpoint at `self.resume_path`, filtered by key prefixes in `self.resume_keys`.
        '''
        if self.resume_path is not None:
            ckpt = torch.load(self.resume_path)
            ckpt = ckpt.get('state_dict', ckpt)  # unwrap if needed
            ckpt = {k: v for k, v in ckpt.items() if any(k.startswith(prefix) for prefix in self.resume_keys)}
            missing, unexpected = self.load_state_dict(ckpt, strict=False)
            assert len(unexpected) == 0, f'Unexpected keys: {unexpected}'
            print(f'[!] Loaded checkpoint from {self.resume_path}. Loaded keys: {len(ckpt)}')

    def freeze_weights(self):
        '''
        Freeze weights based on the keys in `self.freeze_keys`.
        '''
        assert all(k in ['in_projection', 'pos_embed', 'out_projection', 'label_projection', 'odeblock'] for k in self.freeze_keys), \
            f'Unknown keys in freeze_keys: {self.freeze_keys}'
        param_count = 0
        for key in self.freeze_keys:
            target = getattr(self, key)
            if isinstance(target, nn.Module):
                for param in target.parameters():
                    param.requires_grad = False
                    param_count += param.numel()
            elif isinstance(target, nn.Parameter):  # e.g. pos_embed
                target.requires_grad = False
                param_count += target.numel()
            else:
                raise ValueError(f'Unknown target type: {type(target)}')
        print(f'[!] {param_count} parameters are freezed.')

    def forward(self, x, return_features=False, method='dopri5'):
        x = self.in_projection(x)
        eval_t = torch.tensor([0., 1.], device=x.device)
        features = self.odeblock(x, method=method, eval_times=eval_t)[-1]

        pred = self.out_projection(features)

        if return_features:
            return features, pred

        return pred

    @torch.inference_mode()
    def inference(self, X, method='dopri5', num_timesteps=1+1, return_feat=False):
        '''
        Do inference and return the prediction (optionally also return the prediction in the latent space). 
        If one need a whole trajectory in the latent space, use get_traj instead.
        '''
        if self.method == 'onestep':
            z0 = self.in_projection(X)
            feat = self.odeblock.odefunc(0, z0)
            pred = self.out_projection(feat)
        elif method == 'dopri5':
            feat, pred = self(X, return_features=True, method='dopri5')
        else:
            traj, pred = self.get_traj(X, method=method, timesteps=num_timesteps)  # use get_traj to enforce n-step euler.
            feat = traj[-1]
        if return_feat:
            return feat, pred
        return pred

    def evaluate(self, batch, method='dopri5', num_timesteps=1+1, dataloader_type='val', save_mse=False):
        '''
        Do inference and save metric (with data count)
        Also save nfe for dopri.
        '''
        X, Y = batch
        bs = X.size(0)
        try:
            pred = self.inference(X, method=method, num_timesteps=num_timesteps, return_feat=False)
        except AssertionError:
            # prevent shutdown for dopri error
            print(f'Error in inference with method {method}')
            pred = Y * torch.nan
        if method == 'dopri5':
            self.metrics[f'{dataloader_type}/dopri_nfe'] += self.odeblock.odefunc.nfe * bs

        if self.metric_type == 'accuracy':
            if method == 'dopri5':
                self.metrics[f'{dataloader_type}/accuracy_dopri'] += (pred.argmax(dim=-1) == Y.argmax(dim=-1)).float().sum().item()
            else:
                self.metrics[f'{dataloader_type}/accuracy_{num_timesteps-1}'] += (pred.argmax(dim=-1) == Y.argmax(dim=-1)).float().sum().item()
        elif self.metric_type == 'rmse':
            if self.label_scaler is not None:
                Y_unnorm = self.label_scaler.inverse_transform(Y.cpu().numpy())
                pred_unnorm = self.label_scaler.inverse_transform(pred.cpu().numpy())
                rmse = np.mean((Y_unnorm - pred_unnorm)**2)
            else:
                rmse = F.mse_loss(pred, Y).item()
            if method == 'dopri5':
                self.metrics[f'{dataloader_type}/rmse_dopri'] += rmse * bs
            else:
                self.metrics[f'{dataloader_type}/rmse_{num_timesteps-1}'] += rmse * bs
        else:
            raise ValueError(f'Unknown metric type: {self.metric_type}')

    def get_traj(self, x, timesteps=100+1, method='dopri5'):
        '''
        note: should +1 to timesteps since it is both start & end inclusive.
        '''
        x = self.in_projection(x)
        out = self.odeblock.trajectory(x, timesteps, method=method)
        return out, self.out_projection(out[-1])

    def pred_v(self, z, t):
        self.odeblock.odefunc.nfe = 0
        return self.odeblock.odefunc(t, z)

    def sample_timestep(self, z0):
        t = torch.rand(z0.size(0), device=self.device)
        t = append_dims(t, z0.ndim)
        # make some portion of sampled t to zero
        if self.force_zero_prob > 0.:
            mask = (torch.rand_like(t) < self.force_zero_prob).float()
            t = t * (1. - mask)
        return t

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)

        # lr scheduler
        if self.scheduler == 'none':
            return optimizer
        elif self.scheduler == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.total_steps, eta_min=0)
        else:
            raise ValueError(f'Unknown scheduler: {self.scheduler}')

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }

    def on_after_backward(self):
        '''
        This function is called after the backward pass and before the optimizer step.
        '''
        backward_nfe = int(self.odeblock.odefunc.nfe)
        self.log('train/backward_nfe', backward_nfe)
        self.odeblock.odefunc.nfe = 0

    # Training
    def node_training_step(self, batch, batch_idx):
        X, Y = batch

        pred = self(X)
        loss = self.task_criterion(pred, Y)
        forward_nfe = int(self.odeblock.odefunc.nfe)
        self.log_dict({
            'train/loss': loss.item(),
            'train/nfe': forward_nfe,
        })
        self.odeblock.odefunc.nfe = 0
        return loss

    def ours_training_step(self, batch, batch_idx):
        X, Y = batch
        z0 = self.in_projection(X)
        z1 = self.label_projection(Y)

        # sample timestep and construct zt, vt
        t = self.sample_timestep(z0)
        zt = self.dynamics.get_zt(z0, z1, t)
        v_target = self.dynamics.get_vt(z0, z1, t).squeeze()

        # flow loss
        v_pred = self.pred_v(zt, t)
        flow_loss = F.mse_loss(v_pred, v_target)

        # label autoencoding loss
        z1_noised = z1
        if self.label_ae_noise > 0.:
            z1_noised = z1 + self.label_ae_noise * torch.randn_like(z1)
        y_pred = self.out_projection(z1_noised)
        label_ae_loss = self.label_ae_criterion(y_pred, Y)

        # total loss
        loss = flow_loss + label_ae_loss

        # logging
        self.log_dict({
            'train/loss': loss.item(),
            'train/flow_loss': flow_loss.item(),
            'train/label_ae_loss': label_ae_loss.item(),
        })

        return loss

    def training_step(self, batch, batch_idx):
        if self.method == 'node':
            return self.node_training_step(batch, batch_idx)
        elif self.method == 'ours':
            return self.ours_training_step(batch, batch_idx)
        else:
            raise ValueError(f'Method {self.method} not supported')

    def on_validation_epoch_start(self):
        self.metrics = {}
        for dataloader_type in ['val', 'test']:
            for num_euler_steps in [1, 2, 10, 20]:
                self.metrics[f'{dataloader_type}/{self.metric_type}_{num_euler_steps}'] = torch.tensor(0.)
            self.metrics[f'{dataloader_type}/{self.metric_type}_dopri'] = 0.
            self.metrics[f'{dataloader_type}/dopri_nfe'] = 0.

        self.data_count = {
            'val': 0.,
            'test': 0.,
        }
        # for label autoencoding accuracy
        self.num_classes = None

    def on_validation_model_zero_grad(self):
        '''
        Small hack to avoid validation step on resume. 
        This will NOT work if the gradient accumulation step should be performed at this point.
        We raise StopIteration Exception to make training_epoch_loop.run() stop, just before val_loop.run().
        See training_epoch_loop.run(), and ~.on_advance_end().
        '''
        super().on_validation_model_zero_grad()
        if self.trainer.ckpt_path is not None and getattr(self, '_restarting_skip_val_flag', True):
            self._restarting_skip_val_flag = False
            raise StopIteration

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0:
            self.num_classes = batch[1].size(-1)

        if dataloader_idx == 0:  # test set
            dataloader_type = 'test'
        elif dataloader_idx == 1:  # optional train-subset or held-out validation set.
            dataloader_type = 'val'
        else:
            raise ValueError(f'Unknown dataloader_idx: {dataloader_idx}')

        X, Y = batch
        bs = X.size(0)
        self.data_count[dataloader_type] += bs
        # final metric and data/latent mse
        self.evaluate(batch, method='dopri5', num_timesteps=1+1, dataloader_type=dataloader_type, save_mse=True)
        for num_euler_steps in [1, 2, 10, 20]:
            self.evaluate(batch, method='euler', num_timesteps=num_euler_steps+1, dataloader_type=dataloader_type, save_mse=False)

    def on_validation_epoch_end(self):
        gathered_metrics: Dict[str, List[torch.Tensor]] = self.all_gather(self.metrics)
        summed_metrics = {k: v.sum().item() for k, v in gathered_metrics.items()}
        gathered_data_count: Dict[str, List[torch.Tensor]] = self.all_gather(self.data_count)
        total_data_count = {k: v.sum().item() for k, v in gathered_data_count.items()}
        self.metrics = summed_metrics
        self.data_count = total_data_count

        # calculate average
        for k, v in self.metrics.items():
            dataloader_type = k.split('/')[0]
            self.metrics[k] /= max(self.data_count[dataloader_type], 1)

        if self.metric_type == 'rmse':
            for k, v in self.metrics.items():
                if 'rmse' in k:
                    self.metrics[k] = v ** 0.5

        # update best metric
        if self.metric_type == 'accuracy':
            if self.metrics['val/accuracy_dopri'] > self.best_metric['val/best_acc']:
                self.best_metric['val/best_acc'] = self.metrics['val/accuracy_dopri']

            if self.metrics['test/accuracy_dopri'] > self.best_metric['test/best_acc']:
                self.best_metric['test/best_acc'] = self.metrics['test/accuracy_dopri']
        elif self.metric_type == 'rmse':
            if self.metrics['val/rmse_dopri'] < self.best_metric['val/best_rmse']:
                self.best_metric['val/best_rmse'] = self.metrics['val/rmse_dopri']

            if self.metrics['test/rmse_dopri'] < self.best_metric['test/best_rmse']:
                self.best_metric['test/best_rmse'] = self.metrics['test/rmse_dopri']

        # log all metrics
        self.log_dict(self.metrics, sync_dist=True)
        self.log_dict(self.best_metric, sync_dist=True, prog_bar=True)

    def on_fit_end(self):
        save_path = os.path.join(self.logger.save_dir, f'last_step={self.trainer.global_step}.ckpt')
        self.trainer.save_checkpoint(save_path)
        print(f'[!] Saved last checkpoint at {save_path}')
