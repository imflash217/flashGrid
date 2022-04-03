##
## Projects: Exploring GridCells for Object Detection
## Author: Vinay Kumar (@imflash217)
## Supervisor: Dr. Tianfu Wu
## Place: North Carolina State University
## Update: September 2021
##

import torch
import torch.nn.functional as F


def create_block_diagonal_weights(num_channels, num_blocks, block_size):
    ## TODO: Consider reordering the parameters to (num_blocks, num_channels, block_size)
    """## VERIFIED
    Create Block Diagonal Weights using Normal Distribution
    Args:
        num_channels: number of channels
        num_blocks: number of bloacks (usually 16 i.e. "ARGS.num_multiverse")
        block_size: the size of each block (usually 6 i.e. "ARGS.dim_universe")
    Returns:
        A block diagonal matrix (blocked in dimension -1 & -2)
    """
    matrices = torch.distributions.normal.Normal(loc=0, scale=0.01).sample(
        sample_shape=(num_blocks, num_channels, block_size, block_size)
    )
    matrices = torch.unbind(matrices, dim=0)
    return block_diagonal(matrices)


def block_diagonal(matrices):
    """## VERIFIED
    Constructs block-diagonal matrices from a list of batched 2D tensors
    Args:
        matrices : a list of tensors with shape (..., N_i, M_i)
                   i.e. a list of matrices with the same batch dimension
    Returns:
        a matrix with the input matrices stacked along its diagonal having
        shape =  (..., \sum_i N_i, \sum_i M_i)
    """
    block_mat_rows = 0  ## the number of rows in block-matrix
    block_mat_cols = 0  ## the number of columns in block-matrix
    ## the dimension of the rest part of the block-matrix
    batch_shape = matrices[0].shape[:-2]
    for mat in matrices:
        assert (
            mat.shape[:-2] == batch_shape
        ), "All other dimensions (except the row & cols) of every matrix to be blocked MUSt match"
        block_mat_rows += mat.shape[-2]
        block_mat_cols += mat.shape[-1]
    pivot_start = 0
    pivot_end = 0
    row_blocks = []
    for mat in matrices:
        pivot_end += mat.shape[-1]
        pad_before = pivot_start
        pad_after = block_mat_cols - pivot_end
        ## TODO: verify wheter to use `mat.ndim`. assert it
        no_pad_dims = tuple(0 for _ in range(2 * (mat.ndim - 1)))
        yes_pad_dims = (pad_before, pad_after)
        pad_dims = yes_pad_dims + no_pad_dims
        padded_mat = F.pad(mat, pad_dims, "constant", 0)
        row_blocks.append(padded_mat)
        pivot_start = pivot_end
    block_diag_matrix = torch.cat(tuple(row_blocks), -2)
    assert block_diag_matrix.shape[:-2] == batch_shape
    return block_diag_matrix
