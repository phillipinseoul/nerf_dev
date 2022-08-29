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
(3) Initialize the implicit renderer

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











