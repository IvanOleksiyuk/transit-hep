import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")

import argparse
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import make_moons, make_swiss_roll
from twinturbo.src.utils.plot import plot_dataframe

def set_seed(seed):
    np.random.seed(seed)

def generate_spherical_gaussian(n_samples, n_features):
    return np.random.randn(n_samples, n_features)

def generate_gaussian_with_covariance(n_samples, n_features, correlation=0.8):
    covariance_matrix = np.eye(n_features)
    for i in range(1, n_features):
        covariance_matrix[0, i] = correlation
        covariance_matrix[i, 0] = correlation
    for i in range(1, n_features):
        for j in range(1, n_features):
            if i != j:
                covariance_matrix[i, j] = correlation ** 2
    mean = np.zeros(n_features)
    return np.random.multivariate_normal(mean, covariance_matrix, n_samples)

def generate_uniform_cube(n_samples, n_features, low=0.0, high=1.0):
    return np.random.uniform(low, high, size=(n_samples, n_features))

def generate_uniform_ball(n_samples, n_features, radius=1.0):
    vec = np.random.randn(n_samples, n_features)
    vec /= np.linalg.norm(vec, axis=1)[:, np.newaxis]
    rad = np.random.uniform(0, radius, n_samples)**(1/n_features)
    return vec * rad[:, np.newaxis]

def generate_uniform_sphere(n_samples, n_features):
    vec = np.random.randn(n_samples, n_features)
    vec /= np.linalg.norm(vec, axis=1)[:, np.newaxis]
    return vec

def generate_moons(n_samples, noise=0.1, seed=None):
    return make_moons(n_samples=n_samples, noise=noise, random_state=seed)[0]

def generate_swiss_roll(n_samples, noise=0.5, seed=None):
    data, _ = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=seed)
    return data[:, [0, 2]]  # Use only the first and third columns for a 2D representation

def generate_toroidal(n_samples, n_features, R=1.0, r=0.2):
    theta = np.random.uniform(0, 2*np.pi, n_samples)
    phi = np.random.uniform(0, 2*np.pi, n_samples)
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return np.vstack([x, y, z]).T

def save_to_h5_pandas(data, path):
    columns = [f'x{i}' for i in range(data.shape[1])]
    df = pd.DataFrame(data, columns=columns)
    df.to_hdf(path, key='data', mode='w')

def main():
    parser = argparse.ArgumentParser(description='Generate dummy datasets.')
    parser.add_argument('--type', type=str, required=True, choices=['spherical_gaussian', 'gaussian_with_covariance', 'uniform_cube', 'uniform_ball', 'uniform_sphere', 'moons', 'swiss_roll', 'toroidal'], help='Type of dataset to generate')
    parser.add_argument('--output', type=str, required=True, help='Output path for the HDF5 file')
    parser.add_argument('--plot', type=str, nargs='?', const='default', help='Plot directory for the dataset')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--n_features', type=int, default=3, help='Number of features for n-dimensional datasets')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    if args.type == 'spherical_gaussian':
        data = generate_spherical_gaussian(args.n_samples, args.n_features)
    elif args.type == 'gaussian_with_covariance':
        data = generate_gaussian_with_covariance(args.n_samples, args.n_features)
    elif args.type == 'uniform_cube':
        data = generate_uniform_cube(args.n_samples, args.n_features)
    elif args.type == 'uniform_ball':
        data = generate_uniform_ball(args.n_samples, args.n_features)
    elif args.type == 'uniform_sphere':
        data = generate_uniform_sphere(args.n_samples, args.n_features)
    elif args.type == 'moons':
        data = generate_moons(args.n_samples, seed=args.seed)
    elif args.type == 'swiss_roll':
        data = generate_swiss_roll(args.n_samples, seed=args.seed)
    elif args.type == 'toroidal':
        data = generate_toroidal(args.n_samples, args.n_features)
    else:
        raise ValueError('Unknown dataset type')
    

    save_to_h5_pandas(data, args.output)
    df = pd.DataFrame(data, columns=[f'x{i}' for i in range(data.shape[1])])
    if args.plot:
        if args.plot == 'default':
            plot_dir = os.path.splitext(args.output)[0] + '_plots'
        else:
            plot_dir = args.plot
        plot_name = args.type
        plot_dataframe(df, plot_dir, plot_name)

if __name__ == '__main__':
    main()
