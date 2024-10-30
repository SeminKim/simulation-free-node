
import torch
import torch.nn as nn
from .base import *


class tMLPBlock(nn.Module):
    def __init__(self, t_dim, hidden_dim):
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim+t_dim, hidden_dim)

    def forward(self, x, t):
        out = self.fc1(x)
        return out


class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        return out


class MLPODEFunc(nn.Module):
    """MLP modeling the derivative of ODE system.

    Parameters
    ----------
    device : torch.device

    data_dim : int
        Dimension of data.

    hidden_dim : int
        Dimension of hidden layers.

    time_dependent : bool
        If True adds time as input, making ODE time dependent.
    """

    def __init__(self, device, data_dim, hidden_dim, time_dependent=True, h_add_blocks=0):
        super(MLPODEFunc, self).__init__()
        self.device = device
        self.data_dim = data_dim
        self.input_dim = data_dim
        self.hidden_dim = hidden_dim
        self.nfe = 0  # Number of function evaluations
        t_dim = 0

        self.time_dependent = time_dependent

        self.h_add_blocks = h_add_blocks
        if self.h_add_blocks == -1:
            print('[!] Using identity ODEFunc..')
            return

        if time_dependent:
            t_dim = 1

        self.fc1 = nn.Linear(self.input_dim + t_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim + t_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim + t_dim, self.input_dim)

        self.non_linearity = nn.ReLU(inplace=False)

        self.h_add_blocks = h_add_blocks
        if h_add_blocks > 0:
            tmlp_layers = [tMLPBlock(t_dim, hidden_dim) for _ in range(h_add_blocks)]
            self.tmlp_blocks = nn.ModuleList(tmlp_layers)

    def forward(self, t, x):
        """
        Parameters
        ----------
        t : torch.Tensor
            Current time. Shape (1,).

        x : torch.Tensor
            Shape (batch_size, input_dim)
        """
        # Forward pass of model corresponds to one function evaluation, so
        # increment counter
        self.nfe += 1
        if self.h_add_blocks == -1:
            return x  # identity

        if self.time_dependent:
            t_vec = torch.ones(x.shape[0], 1, device=x.device) * t
            out = self.fc1(torch.cat([t_vec, x], 1))
            out = self.non_linearity(out)
            out = self.fc2(torch.cat([t_vec, out], 1))
            out = self.non_linearity(out)

            if self.h_add_blocks > 0:
                for i in range(self.h_add_blocks):
                    out = self.tmlp_blocks[i](torch.cat([t_vec, out], 1), t_vec)
                    out = self.non_linearity(out)

            out = self.fc3(torch.cat([t_vec, out], 1))

        else:
            out = self.fc1(x)
            out = self.non_linearity(out)
            out = self.fc2(out)
            out = self.non_linearity(out)

            if self.h_add_blocks > 0:
                for i in range(self.h_add_blocks):
                    out = self.tmlp_blocks[i](out, t=None)
                    out = self.non_linearity(out)

            out = self.fc3(out)

        return out


class MLPModel(BaseModel):
    """An ODEBlock followed by a Linear layer.

    Parameters
    ----------
    device : torch.device

    data_dim : int
        Dimension of data.

    hidden_dim : int
        Dimension of hidden layers.

    output_dim : int
        Dimension of output after hidden layer. Should be 1 for regression or
        num_classes for classification.

    time_dependent : bool
        If True adds time as input, making ODE time dependent.

    tol : float
        Error tolerance.

    adjoint : bool
        If True calculates gradient with adjoint method, otherwise
        backpropagates directly through operations of ODE solver.
    """

    def __init__(self, *args, data_dim, hidden_dim, latent_dim=None, output_dim=1,
                 time_dependent=True, tol=1e-3, in_proj=False, out_proj=False, proj_norm='none',
                 f_add_blocks=0, h_add_blocks=0, g_add_blocks=0, **kwargs):
        super().__init__(*args, **kwargs)

        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.time_dependent = time_dependent
        self.tol = tol

        if latent_dim is None:
            latent_dim = data_dim
        self.latent_dim = latent_dim

        odefunc = MLPODEFunc(self.device, self.latent_dim, hidden_dim,
                             time_dependent, h_add_blocks=h_add_blocks)

        self.odeblock = ODEBlock(self.device, odefunc, tol=tol, adjoint=self.adjoint)

        if in_proj == 'identity' or in_proj is False:
            self.in_projection = nn.Flatten()
        elif in_proj == 'mlp' or in_proj is True:
            latent_dim = self.odeblock.odefunc.input_dim
            in_projection = [
                nn.Flatten(),
                nn.Linear(data_dim, latent_dim),
                nn.BatchNorm1d(latent_dim) if proj_norm == 'bn' else nn.Identity(),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim),
                nn.BatchNorm1d(latent_dim) if proj_norm == 'bn' else nn.Identity(),
            ]
            for _ in range(f_add_blocks):
                in_projection.extend([
                    nn.ReLU(),
                    nn.Linear(latent_dim, latent_dim),
                ])
            self.in_projection = nn.Sequential(
                *in_projection
            )
        else:
            raise ValueError(f'Invalid in_proj {type(in_proj)} {in_proj}')

        if out_proj == 'identity' or out_proj is False:
            self.out_projection = nn.Identity()
        elif out_proj == 'linear' or out_proj is True:
            self.out_projection = nn.Linear(self.odeblock.odefunc.input_dim,
                                            self.output_dim,)
        elif out_proj == 'mlp':
            latent_dim = self.odeblock.odefunc.input_dim
            self.out_projection = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.BatchNorm1d(latent_dim) if proj_norm == 'bn' else nn.Identity(),
                nn.ReLU(),
                nn.Linear(latent_dim, self.output_dim)
            )
        else:
            raise ValueError(f'Invalid out_proj {type(out_proj)} {out_proj}')

        if g_add_blocks > 0:
            out_proj = [MLPBlock(latent_dim, latent_dim) for _ in range(g_add_blocks)]
            out_proj += [nn.Linear(latent_dim, self.output_dim)]
            self.out_projection = nn.Sequential(*out_proj)

        latent_dim = self.odeblock.odefunc.input_dim
        self.label_projection = nn.Linear(self.output_dim, latent_dim)
