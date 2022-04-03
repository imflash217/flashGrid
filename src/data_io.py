## 🎯
## Projects: Exploring GridCells for Object Detection
## Author: Vinay Kumar
## Guide: Dr. Tianfu Wu
## Place: North Carolina State University
## Time: August/September 2021
##

import itertools
import numpy as np
import utils
import torch

class DataGenerator:
    def __init__(
        self,
        num_interval: int = 1000,
        min: int = 0,
        max: int = 1,
        shape: str = "square",
    ):
        """
        Args:
            num_interval : number of interval steps to take
            min : minimum length of the playground
            max : maximum length of the playground
            shape : shape of the playground
        Returns:
            ....
        """
        self.num_interval = num_interval
        self.min = min
        self.max = max
        self.shape = shape
        self.len_interval = (self.max - self.min) / (self.num_interval - 1)

    def generate(
        self,
        num_data: int = 30000,
        max_velocity: float = 3.0,
        min_velocity: float = 0.0,
        num_steps: int = 1,
        dtype: int = 2,
        test: bool = False,
        visualize: bool = False,
        motion_type: str = "continuous",
    ):
        """
        Args:
            num_data : number of data samples
            max_velocity : maximum allowed velocity (i.e. steps)
            min_velocity : minimum allowed velocity
            num_steps : number of allowed steps to be taken at a time instant
            dtype : type of dataset needed
            test : Is it in_training? or in_testing? phase
            visualize :
            motion_type : Is the motion "continuous" or "discrete"?

        Raises: [NotImplementedError]
        Returns: place_pair ::
        """

        if dtype == 1:
            place_pair = self.gen_2D_multi_dtype1(num_data=num_data)
        elif dtype == 2:
            place_pair = self.gen_2D_multi_dtype2(
                num_data=num_data,
                max_velocity=max_velocity,
                min_velocity=min_velocity,
                num_steps=num_steps,
                test=test,
                visualize=visualize,
                motion_type=motion_type,
            )
        else:
            raise NotImplementedError
        return place_pair

    def gen_2D_multi_dtype1(self, num_data: int):
        """## VERIFIED 🎯🎯
        Args:
            num_data : number of positions and velocity data points
        Raise: NotImplementedError
        Returns:
            place_pair : a dict of before & after positions (mu) and velocities
        """
        # 🎯 Generating random before and after positions (mu) of motion
        ##   used to calculate velocity & direction
        if self.shape == "square":
            ## 🎯  Square Playground
            mu_before = (self.num_interval - 1) * torch.rand((num_data, 2))
            mu_after = (self.num_interval - 1) * torch.rand((num_data, 2))
        elif self.shape == "circle":
            ## 🎯  Circular Playground
            ## "factor" determines how much datapoints are thrown out of the playable area
            factor = 3
            radius = 0.5  ## diameter of the playground is 1; so radius is 0.5
            mu_sequence = torch.rand((num_data * factor, 2))
            x = mu_sequence[:, 0]
            y = mu_sequence[:, 1]
            inside_circle_condition = (
                torch.sqrt((x - radius) ** 2 + (y - radius) ** 2) < radius
            )
            select_idx = torch.where(inside_circle_condition)[0]
            mu_sequence = mu_sequence[
                select_idx[: num_data * 2]
            ]  ## selecting twice the samples; half for mu_before and half for mu_after
            mu_before = (self.num_interval - 1) * mu_sequence[:num_data]
            mu_after = (self.num_interval - 1) * mu_sequence[num_data:]
        elif self.shape == "triangle":
            ## 🎯  Triangular Playground
            factor = 4.2
            mu_sequence = torch.rand((num_data * factor, 2))
            x = mu_sequence[:, 0]
            y = mu_sequence[:, 1]
            inside_triangle_condition = (x + 2 * y > 1) * (x - 2 * y > -1)
            select_idx = torch.where(inside_triangle_condition)[0]
            mu_sequence = mu_sequence[select_idx[: num_data * 2]]
            mu_before = (self.num_interval - 1) * mu_sequence[:num_data]
            mu_after = (self.num_interval - 1) * mu_sequence[num_data:]
        else:
            raise NotImplementedError

        ## 🎯 calculate velocity
        ## multiplying with len_interval to balance mu_interval magnification
        ## while generating mu_before and mu_after
        velocity = (
            torch.sqrt(torch.sum(torch.pow(mu_after - mu_before, 2), dim=1)) * self.len_interval
        )

        assert len(mu_before) == num_data
        assert len(mu_after) == num_data
        assert len(velocity) == num_data

        place_pair = {
            "mu_before": mu_before.double(),
            "mu_after": mu_after.double(),
            "velocity": velocity.double(),
        }

        return place_pair

    def gen_2D_multi_dtype2(
        self,
        num_data: int,
        max_velocity: float,
        min_velocity: float,
        num_steps: int,
        test: bool,
        visualize: bool,
        motion_type: str,
    ):
        """## VERIFIED 🎯🎯
        Samples discretized motions and their corresponding place pairs
        Args:
            num_data :
            max_velocity :
            min_velocity :
            num_steps :
            test :
            visualize :
            motion_type :

        Raises: NotImplementedError
        Returns:
            place_pair : a dict of before and after positions (mu) and velcities & its indices
        """

        velocity_idx = None
        if not test and motion_type == "discrete":
            velocity = utils.generate_velocity_list(
                max_velocity=max_velocity, min_velocity=min_velocity
            )
            num_velocity = len(velocity)
            if num_velocity ** num_steps >= num_data:
                ## We have more velocities than required.
                ## So, sample it to make (size = [num_data, num_steps])
                velocity_idx = np.random.choice(
                    num_velocity, size=(num_data, num_steps)
                )
            else:
                ## We don't have enough velocity data;
                ## So we generate it by iterative multiplication
                ## size=[num_velocity**num_steps, num_steps]
                velocity_list = np.asarray(
                    list(itertools.product(np.arange(num_velocity), repeat=num_steps))
                )
                num_velocity_list = len(velocity_list)  ## num_velocity ** num_steps
                quotient = num_data // num_velocity_list
                remainder = num_data % num_velocity_list
                velocity_idx = np.vstack(
                    (
                        np.tile(velocity_list, reps=(quotient, 1)),
                        velocity_list[
                            np.random.choice(num_velocity_list, size=remainder)
                        ],
                    )
                )
                np.random.shuffle(velocity_idx)

            ## indexing the velocities. same as -> velocity[velocity_idx, ...]
            ## velocity grid contains the various step motions (i.e. velocities)
            ## for every data point (num_data). Its shape is [num_data, num_steps, 2]
            ## here 2 represents the velocities in x & y direction
            velocity_grid = np.take(velocity, velocity_idx, axis=0)
            print(f"velocity.shape = {velocity.shape}")
            print(f"velocity_idx.shape = {velocity_idx.shape}")
            print(f"velocity_grid.shape = {velocity_grid.shape}")

            ## magnifying the velocity's speed by the interval length
            ## i.e. magnifying by the step size of each motion step
            velocity = velocity_grid * self.len_interval

            ## calculating the cummulative displacement in x & y direction (axis=1)
            ## for each data point. shape=[num_data, 2]
            ## NOTE: velocity in unit-time in x-axis gives displacement in x-axis ;
            ##       similar rule for y-axis
            velocity_grid_cumsum = np.cumsum(velocity_grid, axis=1)

            ## finding the max & min displacement along x & y axes for every data point
            ## mu_max & mu_min has the same shape as velocity_grid_cumsum (i.e. [num_data,2])
            mu_max = np.fmin(
                self.num_interval,
                np.min(self.num_interval - velocity_grid_cumsum, axis=1),
            )
            mu_min = np.fmax(0, np.max(0 - velocity_grid_cumsum, axis=1))

            ## sampling the start displacements by randomly scaling
            ## the calculated min & max displacements in each x & y-axis.
            ## mu_start has shape=[num_data, 1, 2]
            mu_start = np.expand_dims(
                (np.random.random(size=(num_data, 2)) * (mu_max - mu_min - 1) + mu_min),
                axis=1,
            )
            assert len(mu_start) == len(velocity_grid_cumsum)  ## == num_data
            ## assert mu_start.shape == velocity_grid_cumsum.shape  ## == [num_data, 1, 2]

            ## adding aggregated velocities into starting positions to get the
            ## next positions for every step.
            ## mu_steps.shape = [num_data, num_steps, 2]
            mu_steps = mu_start + velocity_grid_cumsum

            ## creating a sequence of positions including the starting position
            ## mu_sequence.shape = [num_data, num_steps+1, 2]
            mu_sequence = np.concatenate((mu_start, mu_steps), axis=1)

        elif not test and motion_type == "continuous":
            if self.shape == "square":
                num_data_sample = num_data
            elif self.shape == "circle":
                ## area_of_square / area_of_incircle = 1.27 ~= 1.5
                ## (we choose a bit higher value for safety)
                num_data_sample = int(num_data * 1.5)
            elif self.shape == "triangle":
                ## area_of_square / are_of_equilateral_traingle_inside_square = 4
                num_data_sample = num_data * 4
            else:
                raise NotImplementedError

            ## sampling theta in range [-π, +π)
            theta = (
                np.random.random(size=(num_data_sample, num_steps)) * 2 * np.pi - np.pi
            )

            ## sampling random distance for each step of each data
            ## and scaling it by the max_velocity limit and
            ## shift it by min_velocity limit
            distance = (
                np.sqrt(np.random.random(size=(num_data_sample, num_steps)))
                * (max_velocity - min_velocity)
                + min_velocity
            )

            ## calculating the velocity components from the random distances
            ## vel_x = r*cos(theta)
            ## vel_y = r*sin(theta)
            vel_x = np.expand_dims(distance * np.cos(theta), axis=-1)
            vel_y = np.expand_dims(distance * np.sin(theta), axis=-1)

            ## concatenation the vel_x & vel_y. Shape=[num_data_sample, num_steps, 2]
            velocity_sequence = np.concatenate((vel_x, vel_y), axis=-1)

            velocity_sequence_cumsum = np.cumsum(velocity_sequence, axis=1)

            mu_max = np.fmin(
                self.num_interval,
                np.min(self.num_interval - velocity_sequence_cumsum, axis=1),
            )
            mu_min = np.fmax(0, np.min(0 - velocity_sequence_cumsum, axis=1))
            mu_start = np.expand_dims(
                np.random.random(size=(num_data_sample, 2)) * (mu_max - mu_min - 1)
                + mu_min,
                axis=1,
            )
            assert len(mu_start) == len(velocity_sequence)  ## == num_data_sample

            ## mu_steps is the displacement along x & y-axes for every step of each data point
            ## shape = [num_data_sample, num_steps, 2]
            mu_steps = mu_start + velocity_sequence

            ## concatenate the start position and future positions along the "num_steps" axis
            ## this creates a sequence of motion for every step (including the starting position)
            ## shape = [num_data_sample, num_steps+1, 2]
            mu_sequence = np.concatenate((mu_start, mu_steps), axis=1)

            ## magnifying the speed of each data for every step
            ## in x & y axis by the length of allowed step size
            ## shape=[num_data_sample, num_steps, 2]
            velocity = velocity_sequence * self.len_interval

            if self.shape == "circle":
                ## modification for circular playground
                radius = 0.5
                mu_sequence_magnified = mu_sequence * self.len_interval
                x = mu_sequence_magnified[..., 0]
                y = mu_sequence_magnified[..., 1]

                ## creates a mask for every step of all data points
                ## True  = if a particular step for a data point lies outside the circle
                ## False = if a particular step for a data point lies inside the circle
                mask = np.sqrt((x - radius) ** 2 + (y - radius) ** 2) > radius

                ## only select those data points "nonw of whose steps are outside
                ## the circle boundary.
                ## (that's why we sum across axis=-1 i.e. the num_steps dimension)
                select_idx = np.where(np.sum(mask, axis=-1) == 0)[0]

                ## pick the data points whose v=every step-motion is inside the circle
                ## potentially we could have more valid samples; so we pick only "num_data" data-points
                ## NOTE: Earlier we had defined "num_data_sample" and "num_data". REMEMBER!!
                mu_sequence = mu_sequence[select_idx[:num_data]]
                velocity = velocity[select_idx[:num_data]]
            elif self.shape == "triangle":
                mu_sequence_magnified = mu_sequence * self.len_interval
                x = mu_sequence_magnified[..., 0]
                y = mu_sequence_magnified[..., 1]
                mask = (x + 2 * y > 1) * (x - 2 * y > -1)
                select_idx = np.where(np.sum(mask, axis=-1) == num_steps + 1)[0]
                mu_sequence = mu_sequence[select_idx[:num_data]]
                velocity = velocity[select_idx[:num_data]]
        else:
            ## testing mode
            velocity = utils.generate_velocity_list(
                max_velocity=max_velocity, min_velocity=min_velocity
            )
            num_velocity = len(velocity)
            if visualize:
                ## length of playground is 80x80; so the center is (40,40)
                mu_start = np.reshape([40, 40], newshape=(1, 1, -1))
                velocity_valid = np.where((velocity[:0] >= -1) & (velocity[:1] >= -1))
                ## sampling 10X more data points
                velocity_idx = np.random.choice(
                    velocity_valid[0], size=(num_data * 10, num_steps)
                )
                velocity_grid_cumsum = np.cumsum(
                    np.take(velocity, velocity_idx, axis=0), axis=1
                )
                mu_end = mu_start + velocity_grid_cumsum
                mu_sequence = np.concatenate(
                    (np.tile(mu_start, reps=[num_data * 10, 1, 1]), mu_end), axis=1
                )
                mu_sequence_new = [
                    x for x in mu_sequence if len(x) == len(np.unique(x, axis=0))
                ]
                mu_sequence = np.stack(mu_sequence_new, axis=0)
                velocity_idx_new = [
                    velocity_idx[i]
                    for i, x in enumerate(mu_sequence)
                    if len(x) == len(np.unique(x, axis=0))
                ]
                velocity_idx = np.stack(velocity_idx_new, axis=0)
                mu_sequence_reshaped = np.reshape(
                    mu_sequence, newshape=(-1, (num_steps + 1) * 2)
                )
                mask = mu_sequence_reshaped >= self.num_interval
                select_idx = np.where(np.sum(mask, axis=1) == 0)[0]

                mu_sequence = mu_sequence[select_idx[:num_data]]
                velocity_idx = velocity_idx[select_idx[:num_data]]
                velocity = np.take(velocity, velocity_idx, axis=0) * self.len_interval
            else:
                ## This is the code for testing phase
                velocity_idx = np.random.choice(
                    num_velocity, size=(num_data * 20, num_steps)
                )
                velocity_grid = np.take(velocity, velocity_idx, axis=0)
                velocity_grid_cumsum = np.cumsum(velocity_grid, axis=1)

                mu_max = np.fmin(
                    self.num_interval,
                    np.min(self.num_interval - velocity_grid_cumsum, axis=1),
                )
                mu_min = np.fmax(0, np.max(0 - velocity_grid_cumsum, axis=1))
                select_idx = np.where(np.sum(mu_max <= mu_min, axis=1) == 0)[0][
                    :num_data
                ]
                velocity_idx = velocity_idx[select_idx]
                velocity_grid_cumsum = velocity_grid_cumsum[select_idx]
                velocity_grid = np.take(velocity, select_idx, axis=0)
                mu_max = mu_max[select_idx]
                mu_min = mu_min[select_idx]
                mu_start = np.random.random_sample(size=[num_data, 2])
                mu_start = np.expand_dims(
                    np.round(mu_start * (mu_max - mu_min) + mu_min - 0.5), axis=1
                )
                mu_sequence = np.concatenate(
                    (mu_start, mu_start + velocity_grid_cumsum), axis=1
                )
                velocity = velocity_grid * self.len_interval

        assert len(mu_sequence) == num_data
        place_sequence = {
            "mu_sequence": mu_sequence,
            "velocity": velocity,
            "velocity_idx": velocity_idx,
        }

        return place_sequence
