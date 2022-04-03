#########################################################################################################
## model.py
## PyTorch old-version
##


class GridCellsOLD:
    """The GridCells class"""

    def __init__(self, ARGS):
        """
        Args:
            FLAGS : the hyper-parameters
        """
        self.ARGS = ARGS

        if self.single_block:
            self.max_velocity2 = (self.ARGS.dim_lattice - 1) * (
                np.sqrt(3 / (2 * self.ARGS.alpha))
            )
            self.alpha = torch.tensor([self.ARGS.alpha], dtype=torch.float32)
        else:
            self.max_velocity2 = self.ARGS.max_velocity2
            self.alpha = torch.from_numpy(
                100 * np.random.random(size=(self.num_group)).astype(np.float32)
            )
        self.min_velocity2 = self.ARGS.min_velocity2
        self.velocity2 = utils.generate_velocity_list(
            max_velocity=self.max_velocity, min_velocity=self.min_velocity
        )
        self.num_velocity2 = len(self.velocity2)

        ## INITIALIZE GRIDCELL WEIGHTS
        ## size = (N, N, K*d) = (80, 80, 16*6)
        init_weights_gc = np.random.normal(
            scale=0.001, size=(self.dim_lattice, self.dim_lattice, self.dim_gc)
        ).astype(np.float32)
        self.weights_gc = torch.from_numpy(init_weights_gc)

        ## TODO
        if self.motion_type == "discrete":
            self.weights_M = None  ## TODO

        ## parameters for loss computation
        self.velocity1 = None
        self.place_mu_before = None
        self.place_mu_after = None
        self.lmda2_epoch = None
        self.mu_sequence2 = None
        self.vel2 = None

    def get_grid_code(self, place):
        """ """
        place = torch.unsqueeze(place, dim=0)
        weights_gc = self.unsqueeze(self.weights_gc, dim=0).permute(0, 2, 1, 3)
        grid_code = F.grid_sample(weights_gc, place).squeeze()
        return grid_code

    def create_motion_matrix(self, velocity):
        if self.motion_type == "continuous":
            velocity = torch.reshape(velocity, shape=(-1, 2))
            input_reform = torch.cat(
                (
                    velocity,
                    velocity ** 2,
                    (velocity[:, 0] * velocity[:, 1]).unsqueeze(dim=1),
                ),
                dim=1,
            )
            output = input_reform  ## TODO: implement this with a FC-1 layer

            if self.save_mem:
                weights_M_current = torch.reshape(
                    output, shape=(-1, self.num_group, self.size_block, self.size_block)
                )
            else:
                output = torch.reshape(
                    output, shape=(-1, self.num_group, self.size_block, self.size_block)
                )
                output = torch.unbind(output, dim=1)
                weights_M_current = torch.block_diag(output)
        else:
            weights_M_current = torch.index_select(
                self.weights_M, dim=0, index=velocity
            )  ## TODO: give dim=0
        return torch.squeeze(weights_M_current)

    def motion_model(self, M, grid_code):
        """
        Args:
            M : Motion_matrix contructed using create_motion_matrix() method
        """
        if self.save_mem:
            indices = np.reshape(
                np.arange(self.dim_gc), newshape=(self.num_group, self.size_block)
            )
            grid_code_gt = torch.unsqueeze(
                torch.index_select(grid_code, dim=-1, index=indices), dim=-1
            )
            grid_code_new = torch.matmul(
                M + torch.diag(torch.ones(size=(self.size_block))), grid_code_gt
            )
            grid_code_new = torch.reshape(grid_code_new, shape=(-1, self.dim_gc))
        else:
            grid_code_new = torch.matmul(
                M + torch.diag(torch.ones(size=(self.dim_gc))),
                torch.unsqueeze(grid_code, dim=-1),
            )
        return torch.squeeze(grid_code_new)

    def localization_model(
        self, weights_gc, grid_code, dim_gc, do_pred_percentile=False
    ):
        """ """
        grid_code = torch.reshape(grid_code, shape=(-1, dim_gc))
        weights_gc_reshape = torch.reshape(weights_gc, shape=(-1, dim_gc))
        place_code = torch.matmul(weights_gc_reshape, grid_code.t())
        place_percentile_pred = None
        place_code = torch.reshape(
            place_code,
            shape=(self.dim_lattice, self.dim_lattice, -1).permute((2, 0, 1)),
        )

        if do_pred_percentile:
            place_quantile = np.percentile(place_code, q=98)
            place_percentile_pool = torch.where(place_code - place_quantile >= 0)
            place_percentile_pred_x = np.percentile(place_percentile_pool[:, 1], q=50.0)
            place_percentile_pred_y = np.percentile(place_percentile_pool[:, 2], q=50.0)
            place_percentile_pred = torch.stack(
                (place_percentile_pred_x, place_percentile_pred_y)
            ).type(torch.float32)
        return place_code.squeeze(), place_percentile_pred

    def build_model(self):
        self.place_before1 = None
        self.place_after1 = None
        self.velocity1 = None

        self.place_sequence2 = None
        self.velocity2 = None
        self.lmda = None

        self.place_sequence3 = None

        self.loss1 = self.compute_loss1()
        self.loss2 = self.compute_loss2()
        self.loss3 = self.compute_loss3(self.place_sequence3)
        self.loss4 = self.compute_loss4()

        ## computing total loss
        weights_gc = torch.reshape(
            self.weights_gc, shape=(self.dim_lattice, self.dim_gc)
        )
        mask = torch.reshape(utils.shape_mask(self.dim_lattice, self.shape), shape=(-1))
        weight_gc_masked = weights_gc[
            mask
        ]  ## refer tf.boolean_mask() & torch.masked_select()

        self.reg = self.lmda2 * torch.sum(
            torch.square(
                torch.sum(torch.square(weights_gc_masked), dim=0) / torch.sum(mask)
                - 1.0 / self.dim_gc
            )
        )

        if self.single_block:
            ## NOTE: i.e. in testing mode
            self.loss = self.loss2 + self.loss3 + self.reg
        else:
            self.loss = self.loss1 + self.loss2 + self.loss3 + self.reg

        ## TODO: record the loss in WandB
        ## Optimization step
        ## TODO: define params
        optim = torch.optim.Adam(params, lr=self.lr, betas=(self.beta1, self.beta2))

        ## logic to update the weights_gc and calculate the I2
        ## check TF1 model.py & main.py for more info
        self.I2 = weights_gc @ weights_gc.t()

    def compute_loss1(self):
        grid_code_before1 = self.get_grid_code(self.place_before1)
        grid_code_after1 = self.get_grid_code(self.place_after1)
        self.dp1 = self.GE * torch.exp(-self.velocity1 ** 2 / (2 * ARGS.std ** 2))
        self.dp2 = (1 - self.GE) * torch.exp(-self.velocity1 / 0.3)
        displacement = self.dp1 + self.dp2
        loss1 = torch.sum(
            (torch.sum(grid_code_before1 * grid_code_after1, dim=1) - displacement) ** 2
        )
        return loss1

    def compute_loss2(self):
        grid_code_sequence2 = self.get_grid_code(self.place_sequence2)
        grid_code = grid_code_sequence2[:, 0]
        for step in range(self.num_steps):
            weights_M_current = self.create_motion_matrix(self.velocity2[:, step])
            grid_code = self.motion_model(weight_M_current, grid_code)
            loss2 = torch.sum(
                torch.square(grid_code - grid_code_sequence2[:, step + 1])
            )

        grid_code_end_pred = grid_code
        self.place_end_pred, _ = self.localization_model(
            self.weights_gc, grid_code_end_pred, self.dim_gc
        )
        self.place_start_infer, _ = self.localization_model(
            self.weights_gc, self.grid_code_sequence2[:, 0], self.dim_gc
        )
        self.place_end_infer, _ = self.localization_model(
            self.weights_gc, self.grid_code_sequence2[:, -1], self.dim_gc
        )
        self.place_start_gt = self.place_sequence2[:, 0]
        self.place_end_gt = self.place_sequence2[:, -1]

        loss2 *= self.lmda
        return loss2

    def compute_loss3(self, place_sequence):
        loss3 = 0.0
        for block_idx in range(self.num_group):
            block_slice = torch.arange(
                block_idx * self.size_block, (block_idx + 1) * self.size_block
            )
            place_sequence_block = torch.index_select(
                place_sequence, dim=1, index=torch.tensor(block_idx)
            )
            grid_code = self.get_grid_code(place_sequence_block)
            grid_code_block = torch.index_select(
                grid_code, dim=-1, index=torch.tensor(block_idx)
            )
            alpha_block = torch.index_select(
                self.alpha, dim=0, index=torch.tensor(block_idx)
            )
            displacement = (
                place_sequence_bloack[:, 0] - place_sequence_block[:, 1]
            ) * self.len_interval
            self.local_kernel = (
                1 - alpha_block * torch.sum(displacement ** 2, dim=-1)
            ) / self.num_group
            self.grid_code_block_inner_product = torch.sum(
                grid_code_block[:, 0] * grid_code_block[:, 1], dim=-1
            )
            loss3 += torch.sum(
                torch.square(self.local_kernel - self.grid_code_block_inner_product)
            )
        loss3 *= self.lmda3
        return loss3

    def compute_loss4(self):
        loss4 = torch.sum(torch.abs(torch.sum(self.weights_gc ** 2, dim=2) - 1.0))
        return loss4


#########################################################################################################


def train(model, output_dir: str = "."):
    log_dir = os.path.join(output_dir, "log")
    model_dir = os.path.join(output_dir, "model")

    ## build the model
    model.build_model()
    num_batch = int(np.ceil(FLAGS.num_data / FLAGS.size_batch))
    ## NOTE: check whether using numpy or torch works better
    lmda_list = torch.linspace(FLAGS.lmda, FLAGS.lmda, FLAGS.num_epochs)

    ## TODO: 1. initialize the variables
    ## TODO: 2. plug the WandB socket for metric capture

    ## Generate the data
    data_generator = DataGenerator(
        num_interval=model.num_interval, max=FLAGS.size_place, shape=model.shape
    )

    ## NOTE: TODO: this should be in the forward() method
    start_time = time.time()
    for epoch in range(FLAGS.num_epochs):
        if epoch < FLAGS.iter:
            lmda_list[epoch] = 0
        place_pair1 = data_generator.generate(num_data=FLAGS.num_data, dtype=1)
        place_sequence2 = data_generator.generate(
            num_data=FLAGS.num_data,
            max_velocity=model.max_velocity2,
            min_velocity=model.min_velocity2,
            num_steps=model.num_steps,
            dtype=2,
            motion_type=model.motion_type,
        )
        alpha = model.alpha
        place_sequence3 = []

        ## NOTE: How could epoch be < 0 ?
        if epoch < 0:
            place_sequence3 = data_generator.generate(
                num_data=FLAGS.num_data,
                max_velocity=model.max_velocity2,
                num_steps=1,
                dtype=2,
            )["mu_sequence"]

            place_sequence3 = np.tile(
                np.expand_dims(place_sequence3, axis=1), reps=(1, model.num_group, 1, 1)
            )
        else:
            for block_idx in range(model.num_group):
                max_velocity = min(
                    (np.sqrt((3 / 2) / alpha[block_idx]) / model.len_interval), 10
                )
                place_sequence = data_generator.generate(
                    num_data=FLAGS.num_data,
                    max_velocity=max_velocity,
                    num_steps=1,
                    dtype=2,
                )["mu_sequence"]
                assert len(place_sequence) == FLAGS.num_data
                place_sequence3.append(place_sequence)
            place_sequence3 = np.stack(place_sequence3, axis=1)

        ## minibatch training
        loss_avg = []
        loss1_avg = []
        loss2_avg = []
        loss3_avg = []
        loss4_avg = []
        reg_avg = []

        for minibatch in range(num_batch):
            start_idx = minibatch * FLAGS.size_batch
            end_idx = start_idx + FLAGS.size_batch

            ## update the weights
            model.place_before1 = place_pair1["mu_before"][start_idx:end_idx]
            model.place_after1 = place_pair1["mu_after"][start_idx:end_idx]
            model.velocity1 = place_pair1["velocity"][start_idx:end_idx]
            model.place_sequence2 = place_sequence2["mu_sequence"][start_idx:end_idx]
            model.place_sequence3 = place_sequence3[start_idx:end_idx]
            model.lmda = lmda_list[epoch]
            if model.motion_type == "continuous":
                model.velocity2 = place_sequence2["velocity"][start_idx:end_idx]
            else:
                model.velocity2 = place_sequence2["velocity_idx"][start_idx:end_idx]

            ## TODO: DO a forward pass to update the weights
            ## TODO: Regulaize the weights

            if epoch > 8000 and not model.single_block:
                print("TODO: Add regularization. normalize gradients")

            ## TODO: add WandB hook to capture
            loss_avg.append(model.loss)
            loss1_avg.append(model.loss1)
            loss2_avg.append(model.loss2)
            loss3_avg.append(model.loss3)
            loss4_avg.append(model.loss4)
            reg_avg.append(model.reg)

            if epoch % 20 == 0:
                loss_avg = np.mean(np.asarray(loss_avg))
                loss1_avg = np.mean(np.asarray(loss1_avg))
                loss2_avg = np.mean(np.asarray(loss2_avg))
                loss3_avg = np.mean(np.asarray(loss3_avg))
                loss4_avg = np.mean(np.asarray(loss4_avg))
                reg_avg = np.mean(np.asarray(reg_avg))

                I2 = model.I2
                end_time = time.time()
                print(
                    f"[{epoch}] : alpha = {alpha}\n\t",
                    f"output_dir = {output_dir}\n\t",
                    f"time = {end_time-start_time}\n\t",
                    f"loss_avg = {loss_avg}\n\t",
                    f"loss1_avg = {loss1_avg}\n\t",
                    f"loss2_avg = {loss2_avg}\n\t",
                    f"loss3_avg = {loss3_avg}\n\t",
                    f"loss4_avg = {loss4_avg}\n\t",
                    f"reg_avg = {reg_avg}\n\t",
                    f"max_inner_prod = {I2.max()}\n\t",
                    f"min_inner_prod = {I2.min()}",
                )
                start_time = time.time()
