# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
import torch

def plot_means(means, X, train_set, test_close_set, test_far_set, n_points, save_path):
    print('plotting')

    labels = np.concatenate([np.ones((n_points),dtype=np.int)*i for i,mean in enumerate(means)])

    colors = np.array(['b', 'c', 'm', 'k', 'y'])
    markers = np.array(['^', '*', '+', 'P', 'o'])

    pca = PCA(2)
    # pca = pca.fit(np.vstack(X))
    pca = pca.fit(means)
    for i,x in enumerate(np.vstack(X)): 
        x_pca = pca.transform(x.reshape(1,-1))[0]
        plt.scatter(x_pca[0], x_pca[1], alpha=0.1, c=colors[labels[i]], marker=markers[labels[i]])

    for i, mean in enumerate(means):
        mean_pca = pca.transform(mean.reshape(1,-1))[0]
        plt.scatter(mean_pca[0], mean_pca[1], marker=f'$\mu_{i}$', s=400, c=colors[i])
    plt.savefig(save_path.joinpath('means.png'))
    plt.clf()

    for i,x in enumerate(train_set): 
        x_pca = pca.transform(x.reshape(1,-1))[0]
        plt.scatter(x_pca[0], x_pca[1], alpha=0.1, c=colors[0], marker=markers[0])

    for i,x in enumerate(test_close_set): 
        x_pca = pca.transform(x.reshape(1,-1))[0]
        plt.scatter(x_pca[0], x_pca[1], alpha=0.1, c=colors[1], marker=markers[1])

    for i,x in enumerate(test_far_set): 
        x_pca = pca.transform(x.reshape(1,-1))[0]
        plt.scatter(x_pca[0], x_pca[1], alpha=0.1, c=colors[2], marker=markers[2])

    for i, mean in enumerate(means):
        mean_pca = pca.transform(mean.reshape(1,-1))[0]
        plt.scatter(mean_pca[0], mean_pca[1], marker=f'$\mu_{i}$', s=400, c=colors[i])
    plt.savefig(save_path.joinpath('datasets.png'))

def sample_objectives(
        n_means, 
        sz_train, 
        sz_test, 
        d_objective, 
        save_path, 
        sigma=0.25, 
        low=-1, 
        high=1, 
        plot=False, 
        n_points=1000
    ):
    means = (high - low) * (np.random.rand(n_means, np.multiply(*d_objective))) + low
    X = [np.random.normal(mean, sigma, (n_points, np.multiply(*d_objective))) for mean in means] # n_points about each mean

    # find the furthest mean
    dist = lambda x,y: np.linalg.norm(x-y)
    dists = pdist(means, dist)
    distmat = squareform(dists)
    distmeans = np.mean(distmat,1)
    mu_far = [np.argmax(distmeans)] if n_means > 1 else []
    mu_close = [i for i in range(n_means) if i not in mu_far]

    train_points = [x for i, x in enumerate(X) if i not in mu_far]
    close_inds = [np.random.permutation(len(x)) for x in train_points]
    train_set = [x[close_ind[:sz_train // (n_means-1)]] for x, close_ind in zip(train_points, close_inds)]
    train_mean_inds = [np.array([i]*len(x)) for i,x in zip(mu_close, train_set)]
    train_mean_inds = np.vstack(train_mean_inds).reshape(-1)
    train_set = np.vstack(train_set)
    
    test_close_set = [x[close_ind[sz_train // (n_means-1):sz_train // (n_means-1) + sz_test // (n_means-1)]] for x, close_ind in zip(train_points, close_inds)]
    test_close_mean_inds = [[i]*len(x) for i,x in enumerate(test_close_set)]
    test_close_mean_inds = np.vstack(test_close_mean_inds).reshape(-1)
    test_close_set = np.vstack(test_close_set)

    far_inds = np.random.permutation(np.arange(len(X[mu_far[0]]))) if len(mu_far) else []
    test_far_set = X[mu_far[0]][far_inds[:sz_test]] if len(far_inds) else np.array([])
    test_far_mean_inds = np.array(mu_far * test_far_set.shape[0])

    train_set = train_set.reshape(sz_train, *d_objective)
    test_close_set = test_close_set.reshape(sz_test, *d_objective)
    test_far_set = test_far_set.reshape(sz_test, *d_objective) if len(test_far_set) else test_far_set

    if plot:
        plot_means(
            means, 
            X, 
            train_set, test_close_set, test_far_set, 
            n_points, 
            save_path
        )

    return {
        'train_set': train_set, 
        'train_mean_inds': train_mean_inds, 
        'test_close_set': test_close_set, 
        'test_close_mean_inds': test_close_mean_inds,
        'test_far_set': test_far_set, 
        'test_far_mean_inds': test_far_mean_inds, 
        'means': means, 
        'mu_far': mu_far, 
        'X': X
    }

@hydra.main(version_base=None, config_path='configs', config_name='sample_objectives')
def main(config):
    dataroot = Path(config['dataroot']).joinpath(
        f'means_{config["N_MEANS"]}-locations_{config["N_LOCATIONS"]}-objects_{config["N_OBJECTS"]}-stddev_{config["STDDEV"]}'
    )
    dataroot.mkdir(parents=True, exist_ok=True)

    sz_objective = (config['N_LOCATIONS'], config['N_OBJECTS'])
    objective_params =  sample_objectives(
        config['N_MEANS'], 
        config['N_OBJECTIVES_TRAIN'],
        config['N_OBJECTIVES_TEST'], 
        sz_objective,
        dataroot,
        plot=False,
        n_points=1000
    )

    for param in objective_params.keys():
        torch.save(objective_params[param], dataroot.joinpath(f'{param}.pt'))

if __name__ == '__main__':
    main()