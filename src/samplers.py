import math

import torch

import random

import numpy as np

class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
        "mix_gaussian": MixGaussianSampler
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


def sample_transformation(eigenvalues, normalize=False):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t


class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b


class MixGaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=0.5, scale=None):
        super().__init__(n_dims)
        self.bias1 = torch.normal(mean=bias, std=1.0, size=(1, n_dims))
        # self.bias2 = torch.normal(mean=(-1*bias), std=1.0, size=(1, n_dims))
        self.bias2 = -self.bias1
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None, frac_pos=0.5):
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias1 is not None:
            if n_points == 1:
                random_number = random.uniform(0, 1)
                if random_number>frac_pos:
                    xs_b += self.bias1
                else:
                    xs_b += self.bias2
            else:
                split_index = np.random.binomial(n_points, frac_pos, 1)[0] # for independent train
                # split_index = int(n_points * frac_pos)
                xs_b[:, :split_index, :] += self.bias1
                xs_b[:, split_index:, :] += self.bias2
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
            
        return xs_b