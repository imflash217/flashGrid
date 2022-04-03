"""
## 🎯
## Project: Exploring GridCells for Object Detection
## Author: Vinay Kumar
## Guide: Dr. Tianfu Wu
## Place: North Carolina State University
## Time: October 2021

The design of the exeriment is shown graphically below:

A [10x10] playground_field

    |<--      dim_lattice     -->|
    ------------------------------
    |*  *  *  *  *  *  *  *  *  *|
    |*  *  *  *  *  *  *  *  *  *|
    |*  *  *  *  *  *  *  *  *  *|
    |*  *  *  *  *  *  *  *  *  *|
    |*  *  *  *  *  *  *  *  *  *|
    |*  *  *  *  *  *  *  *  *  *|
    |*  *  *  *  *  *  *  *  *  *|
    |*  *  *  *  *  *  *  *  *  *|
    |*  *  *  *  *  *  *  *  *  *|
    |*  *  *  *  *  *  *  *  *  *|
    ------------------------------

"""

import numpy as np
import torch
import torch.nn.functional as F
import utils
import ops
from einops import rearrange, reduce, repeat


class GridCells(torch.nn.Module):
    """This models the Grid Cells"""

    def __init__(self, ARGS):
        super(GridCells, self).__init__()
        self.ARGS = ARGS
        self.dim_gc = self.ARGS.dim_universe * self.ARGS.num_multiverse
        if self.ARGS.not_multiverse:
            self.max_velocity_l2 = self.ARGS.dim_lattice * np.sqrt(
                3 / (2 * self.ARGS.alpha)
            )
            self.alpha = torch.nn.Parameter(torch.tensor([self.ARGS.alpha], requires_grad=True))
        else:
            self.max_velocity_l2 = self.ARGS.max_velocity_l2
            self.alpha = torch.nn.Parameter(torch.randn(self.ARGS.num_multiverse, requires_grad=True))

        self.min_velocity_l2 = self.ARGS.min_velocity_l2
        self.velocity_l2 = utils.generate_velocity_list(
            self.max_velocity_l2, self.min_velocity_l2
        )
        ## self.num_velocity_l2 = len(self.velocity_l2)
        self.len_interval = 1 / (self.ARGS.dim_lattice - 1)

        ## initialize grid-cells weights A
        self.weights_A = torch.nn.Parameter(
            torch.from_numpy(
                np.random.normal(
                    scale=1e-3,
                    size=(self.ARGS.dim_lattice, self.ARGS.dim_lattice, self.dim_gc),
                )
            )
        )
        ## self.weights_A = torch.nn.Parameter(
        ##     torch.normal(
        ##         mean=0,
        ##         std=1e-3,
        ##         size=(self.ARGS.dim_lattice, self.ARGS.dim_lattice, self.dim_gc),
        ##     )
        ## )

        ## initilize weights for motion matrix M (if the motion is discrete)
        if not self.ARGS.continuous_motion_type:
            self.weights_M = ops.create_block_diaginal_matrix(
                num_channels=len(self.velocity_l2),
                num_blocks=self.ARGS.num_multiverse,
                block_size=self.ARGS.dim_universe,
            )

        self.mm_linear = torch.nn.Linear(
            in_features=5,
            out_features=self.ARGS.num_multiverse * (self.ARGS.dim_universe ** 2),
            bias=False,
        )

    def forward(self, **kwargs):
        """
        Args:
            kwargs: (args_l1, args_l2, args_l3)
                    args_l1 = (place_before_l1, place_after_l1, velocity_l1,)
                    args_l2 = (place_sequence_l2, velocity_l2,)
                    args_l3 = (place_sequence_l3,)
        """
        loss = self.loss(**kwargs)
        return loss

    def loss1(self, place_before_l1, place_after_l1, velocity_l1):
        grid_code_before_l1 = self.get_grid_code(place_before_l1.unsqueeze(dim=0))
        grid_code_after_l1 = self.get_grid_code(place_after_l1.unsqueeze(dim=0))
        print(
            f"Inside loss1(): grid_code_before_l1.shape = {grid_code_before_l1.shape}"
        )
        print(f"Inside loss1(): grid_code_after_l1.shape = {grid_code_after_l1.shape}")
        print(f"Inside loss1(): velocity_l1.shape = {velocity_l1.shape}")
        disp_1 = self.ARGS.GE * torch.exp(
            -(velocity_l1 ** 2) / (2 * (self.ARGS.std ** 2))
        )
        disp_2 = (1 - self.ARGS.GE) * torch.exp(-velocity_l1 / 0.3)
        disp_net = disp_1 + disp_2
        delta_motion = torch.sum(
            grid_code_before_l1 * grid_code_after_l1, dim=0
        )  ## change from tf1 (dim=1)
        loss1 = torch.sum((delta_motion - disp_net) ** 2)
        return torch.tensor(loss1, requires_grad=True)

    def loss2(self, place_sequence_l2, velocity_l2):
        """calculates loss2 term as per paper."""
        print(
            f"Inside loss2() : \
                \n\tplace_sequence_l2.shape = {place_sequence_l2.shape} \
                \n\tvelocity_l2.shape = {velocity_l2.shape}"
        )
        grid_code_sequence_l2 = self.get_grid_code(place_sequence_l2).permute(1, 0, 2)
        grid_code = grid_code_sequence_l2[..., 0]
        loss2 = 0
        for step in range(self.ARGS.num_steps):
            current_M = self.create_motion_matrix_M(velocity_l2[:, step])
            print(
                f"Inside loss2(): \
                  \n\tcurrent_M.shape = {current_M.shape} \
                  \n\tgrid_code.shape = {grid_code.shape}"
            )
            grid_code = self.motion_model(current_M, grid_code)
            print(
                f"~~~~~~~~~~~~~~~~ {grid_code.shape}, {grid_code_sequence_l2[..., step + 1].shape}"
            )
            loss2 += torch.sum(
                torch.pow(grid_code - grid_code_sequence_l2[..., step + 1], 2)
            )
        return loss2

    def loss3(self, place_sequence_l3):
        loss3 = 0
        for verse_id in range(self.ARGS.num_multiverse):
            verse_slice = torch.arange(
                verse_id * self.ARGS.dim_universe,
                (verse_id + 1) * self.ARGS.dim_universe,
            )
            ## print(
            ##     f"//*/*/*/**/*/*/* place_sequence_l3.shape = {place_sequence_l3.shape}"
            ## )

            place_sequence = place_sequence_l3[:, verse_id, ...]
            ## place_sequence = torch.index_select(
            ##     place_sequence_l3, 1, torch.tensor(verse_id)
            ## ).squeeze(dim=1)
            ## print(
            ##     f"//*/*/*/**/*/*/* INSIDE loss3():\n\tplace_sequence.shape = {place_sequence.shape}"
            ## )

            grid_code = self.get_grid_code(place_sequence).permute(1, 0, 2)
            ## print(f"//*/*/*/**/*/*/* grid_code.shape = {grid_code.shape}")

            ## change from tf1 (sxis=-1); bcoz the multiverse dimension is 1 (not -1 as in tf1)
            grid_code = torch.index_select(grid_code, 1, verse_slice)
            ## print(
            ##     f"//*/*/*/**/*/*/* AFTER index_select: grid_code.shape = {grid_code.shape}"
            ## )

            _alpha = self.alpha[verse_id]
            disp = self.len_interval * (place_sequence[:, 0] - place_sequence[:, 1])
            local_kernel = 1 - _alpha * torch.sum(torch.pow(disp, 2), dim=-1)
            local_kernel /= self.ARGS.num_multiverse
            inner_prod = torch.sum(grid_code[:, 0] * grid_code[:, 1], dim=-1)
            loss3 += torch.sum(torch.pow(local_kernel - inner_prod, 2))
        return loss3

    def loss4(self):
        loss4 = torch.sum(torch.pow(self.weights_A, 2), dim=2)
        loss4 = torch.abs(loss4 - 1)
        loss4 = torch.sum(loss4)
        return loss4

    def loss_reg(self):
        A = torch.reshape(self.weights_A, (self.ARGS.dim_lattice ** 2, self.dim_gc))
        mask = utils.shape_mask(self.ARGS.dim_lattice, self.ARGS.shape_lattice)
        mask = torch.tensor(mask)
        mask = torch.reshape(mask, (-1,))
        A_masked = A[mask]
        reg = torch.sum(torch.pow(A_masked, 2), dim=0) / torch.sum(mask)
        reg = torch.sum(torch.pow(reg - 1 / self.dim_gc, 2))
        return reg

    def loss(self, args_l1=None, args_l2=None, args_l3=None):
        loss2 = self.ARGS.lamda_l2 * self.loss2(*args_l2)
        print(f"loss2.requires_grad = {loss2.requires_grad}")
        loss3 = self.ARGS.lamda_l3 * self.loss3(*args_l3)
        reg = self.ARGS.lamda_reg * self.loss_reg()
        if self.ARGS.not_multiverse:
            return loss2 + loss3 + reg
        loss1 = self.ARGS.lamda_l1 * self.loss1(*args_l1)
        loss = loss1 + loss2 + loss3 + reg
        print(f">>>> loss1={loss1}\tloss2={loss2}\tloss3={loss3}\treg={reg}")
        return loss

    def get_grid_code(self, place):
        A = rearrange(self.weights_A, "w h c -> () c h w").double()
        ## A = self.weights_A.unsqueeze(dim=0).permute(0, 3, 2, 1).double()
        ## place = place.unsqueeze(dim=0)
        place = rearrange(place, "... -> () ...")
        grid_code = F.grid_sample(A, place).squeeze()
        print(
            f"Inside get_grid_code(): \
                \n\tA.shape = {A.shape} \
                \n\tplace.shape = {place.shape} \
                \n\tgrid_code.shape = {grid_code.shape}"
        )
        return grid_code

    def create_motion_matrix_M(self, velocity):
        if self.ARGS.continuous_motion_type:
            print(f"@@@@@@@@ : velocity.shape = {velocity.shape}")
            velocity = torch.reshape(velocity, (-1, 2))
            print(f"!!!!!!: velocity_reshape = velocity.shape")
            input_ = torch.cat(
                (
                    velocity,
                    torch.pow(velocity, 2),
                    (velocity[:, 0] * velocity[:, 1]).unsqueeze(dim=-1),
                ),
                dim=-1,
            )
            print(f"########## input_.shape = {input_.shape}")
            output = self.mm_linear(input_)
            print(f"%%%%% output.shape = {output.shape}")
            output = torch.reshape(
                output,
                (
                    -1,
                    self.ARGS.num_multiverse,
                    self.ARGS.dim_universe,
                    self.ARGS.dim_universe,
                ),
            )
            output = torch.unbind(output, dim=1)  ## return "num_multiverse" tensors
            print(f"^^^^^^^^^^^ output.shape = {len(output)}")
            output = ops.block_diagonal(output)
            print(
                f"Inside create_motion_matrix_M(): \
                  \n\tcontinuous_motin_type=True \
                  \n\toutput.shape = {output.shape}"
            )
            return output.squeeze()
        output = torch.index_select(self.weights_M, 0, velocity)
        print(f"Inside create_motion_matrix_M(): \n\toutput.shape = {output.shape}")
        return output.squeeze()

    def motion_model(self, M, grid_code):
        ## TODO: write code for "testing" mode
        print(
            f"Inside motion_model(): \
            \n\tM.shape = {M.shape} \
            \n\tgrid_code.shape = {grid_code.shape}"
        )
        t1 = M + torch.diag(torch.ones(self.dim_gc))
        t2 = grid_code.unsqueeze(dim=-1)
        print(
            f"Inside motion_model: \
              \n\tt1.shape = {t1.shape} \
              \n\tt2.shape = {t2.shape}"
        )
        grid_code = t1 @ t2
        return grid_code.squeeze()

    def localization_model(self):
        ...
