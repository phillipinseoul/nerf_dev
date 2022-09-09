import os
from random import sample
import torch
import numpy as np
import matplotlib.pyplot as plt

from nerf import (
    get_rays,
    sample_stratified,
    PositionalEncoder
)

torch.manual_seed(1)
device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

def test_1():
    '''
    Load the training data.
    '''
    data = np.load('tiny_nerf_data.npz')
    images = data['images']
    poses = data['poses']
    focal = data['focal']

    print(f'images.shape: {images.shape}')
    print(f'poses.shape: {poses.shape}')
    print(f'focal length: {focal}')

    H, W = images.shape[1:3]
    near, far = 2., 6.

    n_training = 100
    test_img_idx = 101
    test_img, test_pose = images[test_img_idx], poses[test_img_idx]

    plt.imshow(test_img)
    plt.savefig('sample_input.png') 

def test_2():
    data = np.load('tiny_nerf_data.npz')
    images = data['images']
    poses = data['poses']
    focal = data['focal']

    '''
    Convert 4x4 poses matrix into 3D coord. denoting i) the origin and ii) a 3D vector
    indicating the direction. These two together describe a vector that indicates where
    a camera was pointing when the photo was taken.
    '''
    dirs = np.stack([
        np.sum([0, 0, -1] * pose[:3, :3], axis=-1) 
        for pose in poses
    ])
    origins = poses[:, :3, -1]

    ax = plt.figure(figsize=(12, 8)).add_subplot(projection='3d')
    _ = ax.quiver(
        origins[..., 0].flatten(),
        origins[..., 1].flatten(),
        origins[..., 2].flatten(),
        dirs[..., 0].flatten(),
        dirs[..., 1].flatten(),
        dirs[..., 2].flatten(),
        length=0.5, normalize=True
    )
    plt.savefig('sample_dirs.png')

def test_3():
    near, far = 2., 6.
    n_training = 100
    test_img_idx = 101

    data = np.load('tiny_nerf_data.npz')
    images = torch.from_numpy(data['images'][:n_training]).to(device)
    poses = torch.from_numpy(data['poses']).to(device)
    focal = torch.from_numpy(data['focal']).to(device)
    test_img = torch.from_numpy(data['images'][test_img_idx]).to(device)
    test_pose = torch.from_numpy(data['poses'][test_img_idx]).to(device)

    H, W = images.shape[1:3]

    with torch.no_grad():
        ray_origin, ray_direction = get_rays(H, W, focal, test_pose)

    print('ray origin:')
    print(ray_origin.shape)
    print(ray_origin[H // 2, W // 2, :])
    print('')

    print('ray direction:')
    print(ray_direction.shape)
    print(ray_direction[H // 2, W // 2, :])
    print('')


def test_4():
    '''
    Draw stratified samples from example
    '''
    near, far = 2., 6.
    n_training = 100
    test_img_idx = 101

    data = np.load('tiny_nerf_data.npz')
    images = torch.from_numpy(data['images'][:n_training]).to(device)
    poses = torch.from_numpy(data['poses']).to(device)
    focal = torch.from_numpy(data['focal']).to(device)
    test_img = torch.from_numpy(data['images'][test_img_idx]).to(device)
    test_pose = torch.from_numpy(data['poses'][test_img_idx]).to(device)

    H, W = images.shape[1:3]

    with torch.no_grad():
        ray_origin, ray_direction = get_rays(H, W, focal, test_pose)

    rays_o = ray_origin.view([-1, 3])       # Fix last dimesion to 3: [_, 3]
    rays_d = ray_direction.view([-1, 3])
    n_samples = 10

    with torch.no_grad():
        pts, z_vals = sample_stratified(
            rays_o, rays_d, near, far, n_samples, 
            perturb=True,
            inverse_depth=False
        )

    # Create positional encoders for i) input points and ii) view directions
    encoder = PositionalEncoder(3, 10)          # d_input: 3, n_freqs: 10
    viewdirs_encoder = PositionalEncoder(3, 4)  # d_input: 3, n_freqs: 4

    # Grab i) flattened points and ii) view directions
    pts_flattened = pts.reshape(-1, 3)
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    flattened_viewdirs = viewdirs[:, None, ...].expand(pts.shape).reshape((-1, 3))

    # Encode inputs
    encoded_points = encoder(pts_flattened)
    encoded_viewdirs = viewdirs_encoder(flattened_viewdirs)

    print('Encoded Points')
    print(encoded_points.shape)
    print(torch.min(encoded_points), torch.max(encoded_points), torch.mean(encoded_points))

    print(encoded_viewdirs)
    print('Encoded Viewdirs')
    print(torch.min(encoded_viewdirs), torch.max(encoded_viewdirs), torch.mean(encoded_viewdirs))

    
    '''
    print(f'input points: {pts.shape}')
    print(f'directions along ray: {z_vals.shape}')

    y_vals = torch.zeros_like(z_vals)
    _, z_vals_unperturbed = sample_stratified(
        rays_o, rays_d, near, far, n_samples, 
        perturb=True,
        inverse_depth=False
    )

    plt.plot(z_vals_unperturbed[0].cpu().numpy(), 
                1 + y_vals[0].cpu().numpy(), 'b-o')
    plt.plot(z_vals[0].cpu().numpy(), 
                y_vals[0].cpu().numpy(), 'r-o')
    plt.ylim([-1, 2])
    plt.title('Stratified Sampling (blue) with Perturbation (red)')
    ax = plt.gca()      # gca: `get current Axes`
    ax.axes.yaxis.set_visible(False)
    plt.grid(True)
    plt.savefig('stratified_sampling.png')
    '''

if __name__=='__main__':
    test_4()
