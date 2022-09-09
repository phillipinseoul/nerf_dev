### Version 1. PyTorch3D NeRF Tutorial
* The original code is from [here](https://github.com/facebookresearch/pytorch3d/blob/main/docs/tutorials/fit_simple_neural_radiance_field.ipynb).
Fit a simple Neural Radiance Field via raymarching.

This tutorial shows how to fit Neural Radiance Field given a set of view of a scene using differentiable implicit function rendering. The presented implicit model is a simplified version of NeRF (Mildenhall et al., ECCV 2020)

The tutorial explains how to:
    1) Create a differentiable implicit function renderer with either `image-grid` or `Monte Carlo sampling`
    2) Create an `implicit model` of a scene
    3) Fit the implicit function (NeRF) based on input images using the `differentiable implicit renderer`
    4) Visualize the learnt implicit function
    
The simplications compared to the original NeRF model are as folows:
    * Ray sampling: This tutorial doesn't perform `stratified` ray sampling, but instead performs ray sampling at equi-distant depths.
    * Rendering: This tutorial does a `single` rendering pass, whereas the original NeRF does a `coarse-to-fine` rendering pass.
    * Architecture: The network is shallower, which allows for faster optimization possibly at the cost of surface details.
    * Mask loss: Since observations include segmentation masks, this tutorial also optimizes a `silhouette loss` that forces rays to either get fully absorbed inside the volume, or to completely pass through it.l

