'''
(4) Helper functions

Define functions that help NeRF optimization process.
'''

import torch

def huber(x, y, scaling=0.1):
    '''
    Obtains `smooth L1 (huber) loss` between rendered silhouttes and colors.
    '''
    diff_sq = (x - y) ** 2
    loss = ((1 + diff_sq / (scaling**2)).clamp(1e-4).sqrt() - 1) * float(scaling)
    return loss

def sample_images_at_mc_locs(target_images, sampled_rays_xy):
    '''
    Given a set of Monte Carlo pixel locations `sampled_rays_xy`,
    this function samples the tensor `target_images` at the respective 2D locations.
    '''
    ba = target_images.shape[0]
    dim = target_images.shape[1]
    spatial_size = sampled_rays_xy.shape[1:-1]
    
    # We have to invert the sign of the `sampled ray positions` to
    # convert the `NDC xy locations` of the MonteCarloRaysampler to the 
    # `coordinate convention` of grid_sample.
    images_sampled = torch.nn.functional.grid_sample(
        target_images.permute(0, 3, 1, 2),
        -sampled_rays_xy.view(ba, -1, 1, 2),     # Note: sign inversion
        align_corners=True
    )
    
    return images_sampled.permute(0, 2, 3, 1).view(
        ba, *spatial_size, dim
    )