import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
    data = np.load('tiny_nerf_data.npz')
    images = data['images']
    poses = data['poses']
    focal = data['focal']

    print(f'images shape: {images.shape}')
    print(f'poses shape: {poses.shape}')
    print(f'focal shape: {focal}')

    H, W = images.shape[1:3]
    near, far = 2., 6.

    n_training = 100
    test_img_idx = 101
    test_img, test_pose = images[test_img_idx], poses[test_img_idx]

    plt.imshow(test_img)
    plt.savefig('save_img.png')
    print(f'Pose: {test_pose}')



