# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Author: Wentao Yuan
'''
Utility functions for data preprocessing.
'''
import numpy as np
import torch


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, value):
        for transform in self.transforms:
            value = transform(value)
        return value


class ToTensor:
    def __call__(self, value):
        if isinstance(value, torch.Tensor):
            tensor = value.float()
        else:
            array = np.array(value)
            if array.ndim == 2:
                array = array[:, :, None]
            tensor = torch.from_numpy(array).float()
        if tensor.ndim == 3 and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1)
        if tensor.numel() > 0 and tensor.max() > 1:
            tensor = tensor / 255.0
        return tensor


class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean).view(-1, 1, 1)
        self.std = torch.as_tensor(std).view(-1, 1, 1)

    def __call__(self, tensor):
        mean = self.mean.to(device=tensor.device, dtype=tensor.dtype)
        std = self.std.to(device=tensor.device, dtype=tensor.dtype)
        return (tensor - mean) / std


class NormalizeInverse(Normalize):
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


normalize_rgb = Compose([
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


denormalize_rgb = Compose([
    NormalizeInverse(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def depth_to_xyz(depth, intrinsics):
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    u, v = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
    Z = depth
    X = (u - cx) * (Z / fx)
    Y = (v - cy) * (Z / fy)
    xyz = np.stack((X, Y, Z), axis=-1)
    return xyz


def jitter_gaussian(xyz, std, clip):
    return xyz + torch.clip(
        torch.randn_like(xyz) * std, -clip, clip
    )


def sample_points(xyz, num_points):
    num_replica = num_points // xyz.shape[0]
    num_remain = num_points % xyz.shape[0]
    pt_idx = torch.randperm(xyz.shape[0])
    pt_idx = torch.cat(
        [pt_idx for _ in range(num_replica)] + [pt_idx[:num_remain]]
    )
    return pt_idx
