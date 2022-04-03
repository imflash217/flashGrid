"""
##
## Projects: Exploring GridCells for Object Detection
## Author: Vinay Kumar (@imflash217)
## Guide: Dr. Tianfu Wu
## Place: North Carolina State University
## Time: August 2021
##
"""

import os
import numpy as np
from scipy.stats import norm
import torch

import matplotlib.pyplot as plt


def generate_velocity_list(
    max_velocity: float,
    min_velocity: float = 0,
    interval: float = 1,
    verbose: bool = True,
) -> list:
    """## VERIFIED 🎯🎯
    Args:
        max_velocity : maximum velocity in the experiment
        min_velocity : minimum velocity in the experiment
        interval : step of unit motion
        verbose : If True, then plot all the velocities in the generated velocity list.

    Returns:
        [a list of possible motion velocities]
    """
    velocity_list = []
    ## converting max velocity to integer
    max_velocity_int = int(np.ceil(max_velocity) + 1)
    ## sampling the unit-motion in x-direction
    for i in np.arange(0, max_velocity_int, interval):
        ## sampling the unit-motion in y-direction
        for j in np.arange(0, max_velocity_int, interval):
            speed = np.sqrt(i ** 2 + j ** 2)
            if min_velocity < speed <= max_velocity:
                velocity_list.append(np.array([i, j]))
                if i > 0:
                    ## stores movement along (-x) axis
                    velocity_list.append(np.array([-i, j]))
                if j > 0:
                    ## stores movement along (-y) axis
                    velocity_list.append(np.array([i, -j]))
                if i > 0 and j > 0:
                    ## stores diagonal movement along (-x) & (-y) axes
                    velocity_list.append(np.array([-i, -j]))
    velocity_list = np.stack(velocity_list)
    if verbose:
        origin = np.zeros_like(velocity_list.T)
        X = velocity_list[:, 0]
        Y = velocity_list[:, 1]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.quiver(*origin, X, Y, scale=1, units="xy", width=0.05)
        ax.scatter(X, Y, c="blue")
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.set_aspect("equal")
        ax.spines["left"].set_position("zero")
        ax.spines["bottom"].set_position("zero")
        ax.spines["right"].set_position("zero")
        ax.spines["top"].set_position("zero")
        plt.show()
    return velocity_list


def shape_mask(size: int = 40, shape: str = "square", verbose: bool = True):
    """## VERIFIED 🎯🎯
    Args:
        size : size of the playground
        shape : dimensions of the playground (eg.: square, rect, circle, triangle....etc)
        verbose : whether to show the plot of the mask or not

    Returns:
        [a numpy.ndarray of bools]
    """
    ## creating the mesh playground
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    mask = None
    if shape == "square":
        mask = np.ones_like(x, dtype=bool)
    elif shape == "circle":
        mask = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2) <= 0.5
    elif shape == "triangle":
        mask = (y + 2 * x >= 1) * (y - 2 * x >= -1)
    else:
        raise NotImplementedError

    if verbose:
        plt.figure(figsize=(8, 8))
        plt.imshow(mask)
    return mask


def plot_2d_heatmap(data, ax, shape: str = "square", min_val=None, max_val=None):
    """
    Plots the data as a 2d-heatmap
    Args:
        data : the data to be plotted
        shape : the playground shape ("square" / "circle", / "triangle")
        min_val : the minimum value to be plotted
        max_val : the maximum value to be plotted

    Returns : None

    """
    data = np.array(data)
    size_place, *_ = data.shape
    mask = shape_mask(size=size_place, shape=shape)
    if min_val == None:
        min_val = data[mask].min() - 0.01
    if max_val == None:
        max_val = data[mask].max()
    data[~mask] = min_val - 1

    cmap = plt.cm.get_cmap("rainbow", 100)
    cmap.set_under(color="k")
    ## print(f"XXXXXXXXXXXXXXXXXXXX {data.shape}")
    ax.imshow(
        data,
        cmap=cmap,
        aspect="equal",
        interpolation="nearest",
        vmin=min_val,
        vmax=max_val,
    )


def mu2map(mu, dim_lattice, std=0.02):
    mu = np.array(mu)  ## conversion to numpy
    mu /= dim_lattice  ## normalization
    if len(mu.shape) == 1:
        ## 1D mu data
        discrete_x = np.linspace(0, 1, dim_lattice)[..., None]  ## unsqueeze
        max_pdf = pow(norm.pdf(0, loc=0, scale=std), 2)
        x = norm.pdf(discrete_x, loc=mu[0], scale=std)
        y = norm.pdf(discrete_x, loc=mu[1], scale=std)
        map_ = x @ y.T
        map_ /= max_pdf
    elif len(mu.shape) == 2:
        ## 2D mu data
        std = 0.005
        map_list = []
        max_pdf = pow(norm.pdf(0, loc=0, scale=std), 2)
        for mu_ in mu:
            discrete_x = np.linspace(0, 1, dim_lattice)[..., None]  ## unsqueeze
            x = norm.pdf(discrete_x, loc=mu_[0], scale=std)
            y = norm.pdf(discret_x, loc=mu_[1], scale=std)
            map_ = x @ y.T
            map_ /= max_pdf
            map_list.append(map_)
        map_ = np.stack(map_list, axis=0)
    return map_


def visualize(model, epoch, path):
    """
    Plots the weights of the GridCells's weights.
    Args:
        model: thrained grid cell model
        epoch: the current epoch of the model
        path: the path to store the plots
    """
    weights_A = model.weights_A.detach().clone()
    alpha = model.alpha
    sorting_order = torch.argsort(alpha)
    print(sorting_order)

    weights_A = torch.reshape(
        weights_A,
        shape=(
            model.ARGS.dim_lattice,
            model.ARGS.dim_lattice,
            model.ARGS.num_multiverse,
            model.ARGS.dim_universe,
        ),
    )
    weights_A = weights_A[:, :, sorting_order]
    weights_A = torch.reshape(
        weights_A,
        shape=(model.ARGS.dim_lattice, model.ARGS.dim_lattice, model.dim_gc),
    )
    weights_A = weights_A.permute(2, 0, 1)

    ## reshaping for sub-plotting
    weights_A = weights_A.reshape(
        (
            model.ARGS.num_multiverse,
            model.ARGS.dim_universe,
            model.ARGS.dim_lattice,
            model.ARGS.dim_lattice,
        )
    )

    ## creating a plot with each row corrsponding to a multiverse's universe
    ## and each column corresponds to a grid cell
    ## as per paper: rows=16 & cols=6 (so 16x6 plots) & each plot is 40x40 size
    nrows = model.ARGS.num_multiverse
    ncols = model.ARGS.dim_universe
    ## plt.figure(figsize=(nrows, ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(20,20))
    for i, weights in enumerate(weights_A):
        for j, W in enumerate(weights):
            ## print(f"WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW: [{i},{j}] -> {W.shape}")
            plot_2d_heatmap(W, axes[i,j], shape=model.ARGS.shape_lattice)
        ## plt.subplot(nrows, ncols, i + 1)
        ## plot_2d_heatmap(weights, shape=model.ARGS.shape_lattice)
    fig.tight_layout()
    fig.savefig(os.path.join(path, f"weights_A_{epoch}.png"))
    return fig
    ## plt.close()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
