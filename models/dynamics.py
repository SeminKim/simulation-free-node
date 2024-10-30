import torch
import numpy as np
from torch.func import vmap, jacfwd
from utils import append_dims


class BaseDynamics:
    def __init__(self, auto_init=True):
        if auto_init:
            self._auto_init()

    def _auto_init(self):
        self._alpha_dot = vmap(jacfwd(self.alpha, argnums=0))
        self._beta_dot = vmap(jacfwd(self.beta, argnums=0))

    def get_zt(self, z0, z1, t):
        t = t.squeeze().reshape(z0.shape[0])
        return append_dims(self.alpha(t), z0.ndim) * z0 + append_dims(self.beta(t), z1.ndim) * z1

    def get_vt(self, z0, z1, t):
        t = t.squeeze().reshape(z0.shape[0])
        return append_dims(self.alpha_dot(t), z0.ndim) * z0 + append_dims(self.beta_dot(t), z1.ndim) * z1

    def alpha(self, t):
        raise NotImplementedError

    def beta(self, t):
        raise NotImplementedError

    def _alpha_dot(self, t):
        raise NotImplementedError

    def _beta_dot(self, t):
        raise NotImplementedError

    def alpha_dot(self, t):
        # can be auto computed
        return self._alpha_dot(t).to(t.dtype)

    def beta_dot(self, t):
        # can be auto computed
        return self._beta_dot(t).to(t.dtype)


class LinearDynamics(BaseDynamics):
    def alpha(self, t):
        return 1 - t

    def beta(self, t):
        return t


class ConcaveDynamics(BaseDynamics):
    def alpha(self, t):
        return torch.cos(t * np.pi / 2)

    def beta(self, t):
        return torch.sin(t * np.pi / 2)


class ConvexDynamics(BaseDynamics):
    def alpha(self, t):
        return 1 - torch.sin(t * np.pi / 2)

    def beta(self, t):
        return 1 - torch.cos(t * np.pi / 2)


def get_dynamics(name, return_v=True):
    '''
    just for backward compatibility
    '''
    assert name in [
        'linear',
        'concave',
        'covnex',
    ]
    assert not return_v
    if name == 'linear':
        return LinearDynamics()
    elif name == 'concave':
        return ConcaveDynamics()
    elif name == 'convex':
        return ConvexDynamics()
    else:
        raise ValueError(f'Unknown dynamics name: {name}')
