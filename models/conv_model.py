import torch
import torch.nn as nn
from utils import *
from .base import *


class ConcatConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class UnitBlock(nn.Module):
    def __init__(self, dim, conv_t_dependent=True):
        super(UnitBlock, self).__init__()

        self.act_func = nn.ReLU(inplace=False)
        self.conv_t_dependent = conv_t_dependent
        if conv_t_dependent:
            self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        else:
            self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, t, x):
        t_vec = torch.ones(x.shape[0], 1, device=x.device) * t.squeeze().reshape(-1, 1)
        out = self.act_func(x)
        if self.conv_t_dependent:
            out = self.conv1(t, out)
        else:
            out = self.conv1(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.act_func = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act_func(out)
        return out


class ConvODEfunc(nn.Module):

    def __init__(self, dim, hidden_dim=0, add_blocks=0, conv_t_dependent=True):
        super(ConvODEfunc, self).__init__()

        self.act_func = nn.ReLU(inplace=False)
        if hidden_dim == 0:
            hidden_dim = dim

        self.conv_t_dependent = conv_t_dependent
        conv_cls = ConcatConv2d if conv_t_dependent else nn.Conv2d
        self.conv1 = conv_cls(dim, hidden_dim, 3, 1, 1)
        self.conv2 = conv_cls(hidden_dim, dim, 3, 1, 1)
        self.nfe = 0

        blocks = [UnitBlock(
            hidden_dim, conv_t_dependent=conv_t_dependent) for _ in range(add_blocks)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, t, x):
        self.nfe += 1
        if t.ndim == 0:
            t = append_dims(torch.ones(x.shape[0], device=x.device), x.ndim) * t
        assert t.ndim == x.ndim, (t.ndim, x.ndim)

        out = self.act_func(x)
        if self.conv_t_dependent:
            out = self.conv1(t, out)
        else:
            out = self.conv1(out)
        # add_blocks
        for block in self.blocks:
            out = block(t, out)

        out = self.act_func(out)

        if self.conv_t_dependent:
            out = self.conv2(t, out)
        else:
            out = self.conv2(out)

        return out


class ConvModel(BaseModel):
    def __init__(self, *args, data_dim=3, emb_res=(7, 7), latent_dim=64,
                 in_latent_dim=64, hidden_dim=0, h_add_blocks=0, f_add_blocks=0,
                 g_add_blocks=0, num_classes=10, **kwargs):
        '''
        params:
        - data_dim: input dimension
        - emb_res: spatial resolution at embedding space
        '''
        super().__init__(*args, **kwargs)
        in_proj_layer = [
            nn.Conv2d(data_dim, in_latent_dim, 3, 1),
            nn.ReLU(inplace=False),
        ]

        for _ in range(f_add_blocks):
            in_proj_layer += [
                nn.Conv2d(in_latent_dim, in_latent_dim, 3, 1, 1),
                nn.ReLU(inplace=False),
            ]

        in_proj_layer += [
            nn.Conv2d(in_latent_dim, in_latent_dim, 4, 2, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_latent_dim, latent_dim, 4, 2, 1),
        ]
        self.in_projection = nn.Sequential(
            *in_proj_layer
        )

        # out projection
        out_proj_layer = [
            ConvBlock(latent_dim) for _ in range(g_add_blocks)
        ]
        out_proj_layer += [
            nn.ReLU(inplace=False) if g_add_blocks == 0 else nn.Identity(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(latent_dim, num_classes),
        ]
        self.out_projection = nn.Sequential(
            *out_proj_layer
        )
        # label projection
        label_proj_layer = [
            nn.Linear(num_classes, latent_dim),
            AppendRepeat(emb_res),
        ]
        self.label_projection = nn.Sequential(
            *label_proj_layer
        )

        odefunc = ConvODEfunc(latent_dim, hidden_dim=hidden_dim, add_blocks=h_add_blocks)
        self.odeblock = ODEBlock(self.device, odefunc, is_conv=True, adjoint=self.adjoint)
