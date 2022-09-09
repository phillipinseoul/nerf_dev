from locale import normalize
import os
import numpy as np
import torch
from torch import nn
from typing import Optional, Tuple, List, Union, Callable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import trange

torch.manual_seed(1)
device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

def get_rays(
    height: int,
    width: int,
    focal_length: float,
    c2w: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Find i) origin and ii) direction of reays through
    every pixel and camera origin.
    '''

    # Apply `pinhole camera model` to gather directions at each pixel.
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32).to(c2w),
        torch.arange(height, dtype=torch.float32).to(c2w),
        # indexing='ij'     # available from torch v1.10
    )
    i, j = i.transpose(-1, -2), j.transpose(-1, -2)

    directions = torch.stack([
        (i - width * 0.5) / focal_length,
        -(j - height * 0.5) / focal_length,
        -torch.ones_like(i)
    ], dim=-1)

    # Apply camera pose to directions.
    rays_d = torch.sum(directions[..., None, :] * c2w[:3, :3], dim=-1)

    # Origin is the same for ALL directions (the `optical center`)
    rays_o = c2w[:3, -1].expand(rays_d.shape)

    return rays_o, rays_d

'''
NeRF takes a `coarse-to-fine` sampling strategy, starting with the `stratified sampling` approach.
'''
def sample_stratified(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    n_samples: int,
    perturb: Optional[bool] = True,
    inverse_depth: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Stratified sampling splits the ray into `even-spaced` bins and randomly samples within each bin. 
    - perturb: Determines whether to i) sample points uniformly from each bin or 
        ii) simply use the `bin center` as the point.
    '''

    # Grab samples for space integration along the ray
    t_vals = torch.linspace(0., 1., n_samples,device=rays_o.device)

    if not inverse_depth:
        z_vals = near * (1. - t_vals) + far * t_vals            # Sample linearly between `near` and `far`
    else:
        temp = 1. / near * (1. - t_vals) + 1. / far * t_vals    # Sample linearly in `inverse depth` (disparity)
        z_vals = 1. / temp

    # Draw uniform sample from bins along the ray
    if perturb:
        mids = 0.5 * (z_vals[1:] + z_vals[:-1])
        upper = torch.cat([mids, z_vals[-1:]], dim=-1)
        lower = torch.cat([z_vals[:1], mids], dim=-1)
        t_rand = torch.rand([n_samples], device=z_vals.device)
        
        z_vals = lower + (upper - lower) + t_rand

    z_vals = z_vals.expand(
        list(rays_o.shape[:-1]) + [n_samples]
    )

    # Apply scale from `rays_d` and offset from `rays_o` to samples.
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    return pts, z_vals


class PositionalEncoder(nn.Module):
    '''
    Sine-cosine positional encoder for input points.

    NeRF uses positional encoding in order to map the inputs to a `higher frequency space`
    to solve the bias the neural networks tend to learn `lower-frequency` functions.
    The same PositionalEncoder is applied to i) input samples and ii) view directions, with 
    different parameters.
    '''
    def __init__(
        self,
        d_input: int,
        n_freqs: int,
        log_space: bool = False
        ):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)
        self.embed_fns = [lambda x: x]

        # Define frequencies in either `linear` or `log` scale
        if self.log_space:
            freq_bands = 2. ** torch.linespace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** (self.n_freqs - 1), self.n_freqs)

        # Alternate `sin` and `cos`
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))     # lambda x, freq: torch.sin(x * freq)
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))     # lambda x, freq: torch.cos(x * freq)

    def forward(
        self,
        x
    ) -> torch.Tensor:
        '''
        Apply positional encoding to input.
        '''
        return torch.cat([fn(x) for fn in self.embed_fns], dim=-1)

def main():
    return

if __name__=='__main__':
    main()    






