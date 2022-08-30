# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import os
import sys
import time
import json
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from IPython import display
from tqdm.notebook import tqdm
import torch

# Data structures and functions for rendering
import pytorch3d
from pytorch3d.structures import Volumes
from pytorch3d.transforms import so3_exp_map
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
    RayBundle,
    ray_bundle_to_ray_points,
)

from utils.generate_cow_renders import generate_cow_renders
from utils.plot_image_grid import plot_image_grid

# Set CUDA device
device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)

'''
(1) Generate images of the scene and masks

Below code generates the training data. It renders the cow mesh from `fit_textured_mesh.ipynb`
tutorial from several viewpoints, and returns: 
    1. A batch of image and silhouette tensors produced by the cow mesh renderer.
    2. A set of cameras corresponding to each render.
'''

target_cameras, target_images, target_silhouettes = generate_cow_renders(num_views=40, azimuth_range=180)
print(f'Generated {len(target_images)} images/silhouettes/cameras.')


'''
(2) Initialize the implicit renderer

Below code initializes an `implicit renderer` that emits a ray from each pixel of a target image, and samples a set 
of `uniformly-spaced` points along the ray. At each ray-point, the corresponding i) density and ii) color value is obtained 
by querying the corresponding location in the neural model of the scene.

The renderer is composed of i) a raymarcher and ii) a raysampler.
i) The `raysampler` is responsible for emitting rays from image pixels and sampling the points along them. Here, we
use two different raysamplers:
    * `MonteCarloRaysampler`: Used for generating rays from a subset of pixels of the image plane. The `random subsampling`
    of pixels is carried out during training to decrease the memory consumption of the implicit model.
    * `NDCMultinomialRaysampler`: Follows the standard PyTorch3D coordinate grid convention.
                                  (+X from right to left; +Y from bottom to top; +Z away from the user)
                                  In combination with the implicit model of the scene, NDCMultinomalRaysampler comsumes a large 
                                  amount of memory and, thus is only used for visualizing the results of the training at test time.

ii) The `raymarcher` takes the densities and colors sampled along each ray and renders each ray into i) a color and ii) an opacity value 
of the ray's source pixel. Here we use the `EmissionAbsorptionRaymarcher` which implements the standard Emission-Absorption remarching algorithm.
'''

render_size = target_images.shape[1] * 2        # render_size: size of both sizes of the rendered images in pixels
volume_extent_world = 3.0                       # The rendered scene is centered around (0,0,0) and is enclosed inside a bounding box
                                                # whose side is roughly equal to 3.0 (world units)

# 1) Instantiate the raysamplers 

raysampler_grid = NDCMultinomialRaysampler(     # Here, `NDCMultinomialRaysampler` generates a rectangular image grid
    image_height=render_size,                   # of rays whose coordinates follow the PyTorch3D coordinate conventions.
    image_width=render_size,
    n_pts_per_ray=128,
    min_depth=0.1,
    max_depth=volume_extent_world,
)

raysampler_mc = MonteCarloRaysampler(           # `MonteCarloRaysampler` generates a random subset of `n_rays_per_image` rays
    min_x=-1.0,                                 # emitted from the image plane.
    max_x=1.0,
    min_y=-1.0,
    max_y=1.0,
    n_rays_per_image=750,
    n_pts_per_ray=128,
    min_depth=0.1,
    max_depth=volume_extent_world,
)

# 2) Instantiate the raymarcher

raymarcher = EmissionAbsorptionRaymarcher()     # `EmissionAbsorptionRaymarcher` marches along each ray in order to 
                                                # render the ray into i) a single 3D color vector and ii) an opacity scalar.

# Finally, instantiate the implicit renderers for both raysamplers.

renderer_grid = ImplicitRenderer(
    raysampler=raysampler_grid, raymarcher=raymarcher,  
)
renderer_mc = ImplicitRenderer(
    raysampler=raysampler_mc, raymarcher=raymarcher,
)


'''
(3) Define the Neural Radiance Field (NeRF) model

Below code defines the `NeuralRadianceField` module, which specifies a continuous field of 
i) colors and ii) opacities over the 3D domain of the scene.

The `forward` function of `NeuralRadianceField` receives as input `a set of tensors that parametrize 
a bundle of rendering rays`. The ray bundle is later converted to 3D ray points in the world 
coordinates of the scene. Each 3D point is then mapped to a `harmonic representation` using the 
`HarmonicEmbedding` layer. The harmonic embeddings then enter the color and opacity branches
of the NeRF model in order to label each ray point with a 3D vector and a 1D scalar ranging in [0-1]
which define the point's i) RGB color and ii) opacity respectively.

Since NeRF has a large memory footprint, we also implement the `NeuralRadianceField.forward_batched` method.
This method splits the input rays into `batches` and executes the `forward function` for each batch
separately in a for loop. This lets us render a large set of rays without running out of GPU memory.
Standardly, `forward_batched` would be used to render rays emitted from all pixels of an image in order
to produce a full-sized render of a scene.
'''
class HarmonicEmbedding(torch.nn.module):
    def __init__(self, n_harmonic_functions=60, omega0=0.1):
       return

    def forward(self, x):
        return

    
class NeuralRadianceField(torch.nn.Module):
    def __init__(self, n_harmonic_functions=60, n_hidden_neurons=256):
        super().__init__()
        '''
        args:
            - n_harmonic_functions: # of harmonic functions used to 
                form the harmonic embedding of each point.
            - n_hidden_neurons: # of hidden units in the fully connected
                layers of the MLPs of the model.
        '''

        self.harmonic_embedding = HarmonicEmbedding(n_harmonic_functions)   # The harmoinc embedding layer converts inputs 
                                                                            # 3D coordinates to a representation that is more
                                                                            # suitable for processing with a `deep neural network`.

        embedding_dim = n_harmonic_functions * 2 * 3                        # Dimension of harmonic embedding

        self.mlp = torch.nn.Sequential(                                     # `self.mlp` is a simple 2-layer multi-layer perceptron
            torch.nn.Linear(embedding_dim, n_hidden_neurons),               # that converts `the input per-point harmonic embeddings``
            torch.nn.Softplus(beta=10.0),                                   # to a latent representation.
            torch.nn.Linear(n_hidden_neurons, n_hidden_neurons),            # Note: Uses Softplus activation instead ReLU.
            torch.nn.Softplus(beta=10.0),
        )

        self.color_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons + embedding_dim, n_hidden_neurons),    # Given features predicted by `self.mlp`, `self.color_layer` is
            torch.nn.Softplus(beta=10.0),                                           # responsible for predicting a 3D per-point vector that 
            torch.nn.Linear(n_hidden_neurons, 3),                                   # represents `RGB color` of that point.
            torch.nn.Sigmoid(),                                                     # Ensures the colors range in [0-1]
        )

        self.density_layer = torch.nn.Sequential(                                   # Density layer converts the features of `self.mlp`
            torch.nn.Linear(n_hidden_neurons, 1),                                   # to a 1D density value representing the raw opacity
            torch.nn.Softplus(beta=10.0),                                           # of each point.
        )

        # We set the `bias of the density layer` to -1.5 in order to 
        # initialize the opacities of the ray points to values close to 0.
        # A crucial detail for ensuring convergence of the model!
        self.density_layer[0].bias.data[0] = -1.5

    def _get_densities(self, features):
        '''
        - input: `features` (predicted by self.mlp)
        - output: `raw_densities`

        Takes predicted features and converts them to `raw_densities` through 
        `self.density_layer`. And `raw_densities` are later mapped to [0-1] range.
        ''' 
        raw_densities = self.density_layer(features)
        return 1 - (-raw_densities).exp()

    def _get_colors(self, features, rays_directions):
        '''
        Takes `per-point predicted features` and evalutes the color model in order 
        to attach to each point `a 3D vector of its RGB color`.
        
        To represent `viewpoint dependent effects`, before evaluating `self.color_layer`,
        `NeuralRadianceField` concats i) `features` and ii) a harmonic embedding of `ray_directions`, 
        which are per-point directions of point rays expressed as 3D l2-normalized vectors in 
        world coordinates.
        '''
        spatial_size = features.shape[:-1]

        rays_directions_normed = torch.nn.functional.normalize(     # Normalize the ray_directions to unit l2 norm.
            rays_directions, dim=-1
        )

        rays_embedding = self.harmonic_embedding(                   # Obtain the harmonic embedding of the 
            rays_directions_normed                                  # normalized ray directions.
        )

        rays_embedding_expand = rays_embedding[..., None, :].expand(    # Expand the `ray directions tensor` so that its spatial size
            *spatial_size, rays_embedding.shape[-1]                     # is equal to the `size of features`.
        )

        color_layer_input = torch.cat(                              # Concat `ray direction embeddings` with `features`
            (features, rays_embedding_expand),                      # and evaluate the color model.
            dim=-1
        )
        return self.color_layer(color_layer_input)

    def forward(
        self, 
        ray_bundle: RayBundle,
        **kwargs,
    ):
        '''
        forward pass:
            - takes parametrizations of 3D points sampled along projection rays
            - attaches a `3D vector` and a `1D scalar`, representing the point's i) RGB color and ii) opacity, respectively
        
        input:
            - ray_bundle: A `RayBundle` object, containing the following variables:
                (i) origins: [minibatch, ..., 3]
                    --> denotes the `origins` of the sampling rays in world coords.
                (ii) directions: [minibatch, ..., 3]
                    --> contains the direction vectors of sampling rays in world coords.
                (iii) lengths: [minibatch, ..., num_points_per_ray]
                    --> contains the lengths at which the rays are sampled.
        output:
            - rays_densities: [minibatch, ..., num_points_per_ray, 1]
                --> denotes the `opacity` of each ray point
            - rays_colors: [minibatch, ..., num_points_per_ray, 3]
                --> denotes the `color` of each ray point
        '''
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)        # Convert ray parametrizations to world coords.

        embeds = self.harmonic_embedding(                               # For each 3D world coord, get its harmonic embedding.
            rays_points_world
        )
        features = self.mlp(embeds)                                     # Maps each harmonic embedding to a `latent feature space`.

        # Finally, given `per-point features`, execute the i) density and ii) color branches.
        rays_densities = self._get_densities(features)
        rays_colors = self._get_colors(features, ray_bundle.directions)

        return rays_densities, rays_colors

    def batched_forward(
        self,
        ray_bundle: RayBundle,
        n_batches: int = 16,
        **kwargs,
    ):
        '''
        Allows `memory efficient` processing of input rays.
        This method is used to export a fully-sized render of the radiance field for visualization purposes.

        input:
            - ray_bundle
            - n_batches: Specifies the # of batches the input rays are split into. 
                --> The larger the # of batches, the smaller the memory footprint and 
                the lower the processing speed.
        
        output:
            - rays_densities
            - rays_colors
        '''
        n_pts_per_ray = ray_bundle.lengths.shape[-1]                    # Parse out shapes needed for `tensor reshaping`.
        spatial_size = [*ray_bundle.origins.shape[:-1], n_pts_per_ray]

        tot_samples = ray_bundle.origins.shape[:-1].numel()             # Split rays into `n_batches`
        batches = torch.chunk(torch.arange(tot_samples), n_batches)

        # For each batch, execute forward pass
        batch_outputs = [
            self.forward(
                RayBundle(
                    origins=ray_bundle.origins.view(-1, 3)[batch_idx],
                    directions=ray_bundle.directions.view(-1, 3)[batch_idx],
                    lengths=ray_bundle.lengths.view(-1, n_pts_per_ray)[batch_idx],
                    xys=None
                )
            ) for batch_idx in batches
        ]

        # Concat the `per-batch` rays_densities and rays_colors and 
        # reshape according to the sizes of the inputs.
        rays_densities, rays_colors = [
            torch.cat(
                [batch_out[output_i] for batch_out in batch_outputs],
                dim=0
            ).view(*spatial_size, -1) for output_i in (0, 1)
        ]
        return rays_densities, rays_colors







