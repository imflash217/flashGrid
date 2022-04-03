"""
## 🎯
## Projects: Exploring GridCells for Object Detection
## Author: Vinay Kumar
## Guide: Dr. Tianfu Wu
## Place: North Carolina State University
## Time: October 2021
##
"""

import wandb
import numpy as np
import torch

from data_io import DataGenerator
from model import GridCells
import utils as utils
from hyperparams import ARGS


def train(model, device):
    num_batches = int(np.ceil(model.ARGS.num_data / model.ARGS.batch_size))
    data_generator = DataGenerator(model.ARGS.dim_lattice)
    print(f"data created successfully")
    ## step-0 building optimizers and model params
    ## move model to the right device
    model.to(torch.double).to(device)

    ## construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"Number of params: {utils.count_parameters(model)}")

    ## optimizer = torch.optim.SGD(
    ##     params,
    ##     lr=model.ARGS.lr,
    ##     momentum=model.ARGS.mom,
    ##     weight_decay=model.ARGS.weight_decay,
    ## )

    optimizer = torch.optim.Adam(
        params, lr=model.ARGS.lr, weight_decay=model.ARGS.weight_decay
    )

    ## and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(model.ARGS.num_epochs):
        ## step-1: gather data
        place_pair_l1 = data_generator.generate(model.ARGS.num_data, dtype=1)
        place_sequence_l2 = data_generator.generate(
            model.ARGS.num_data,
            max_velocity=model.max_velocity_l2,
            min_velocity=model.min_velocity_l2,
            num_steps=model.ARGS.num_steps,
            dtype=2,
        )
        place_sequence_l3 = []
        for verse_id in range(model.ARGS.num_multiverse):
            max_v = 3.0
            ## max_v = 
            place_seq = data_generator.generate(
                model.ARGS.num_data,
                max_velocity=max_v,
                min_velocity=0.0,
                num_steps=1,
                dtype=2,
            )["mu_sequence"]
            place_sequence_l3.append(torch.tensor(place_seq))
        place_sequence_l3 = torch.stack(place_sequence_l3, dim=1)

        ## step-2: create batches
        for batch in range(num_batches):
            start = batch * model.ARGS.batch_size
            end = (batch + 1) * model.ARGS.batch_size
            ## print(f"+++++++++++++++ start, end = {start}, {end}")
            args_l1 = (
                place_pair_l1["mu_before"][start:end],
                place_pair_l1["mu_before"][start:end],
                place_pair_l1["velocity"][start:end],
            )
            if model.ARGS.continuous_motion_type:
                v_l2 = place_sequence_l2["velocity"][start:end]
            else:
                v_l2 = place_sequence_l2["velocity_idx"][start:end]
            args_l2 = (
                torch.tensor(
                    place_sequence_l2["mu_sequence"][start:end], dtype=torch.double
                ),
                torch.tensor(v_l2, dtype=torch.double),
            )
            ## print(f"++++++++++++ args_l2 : {args_l2[0].shape}, {args_l2[1].shape}")
            args_l3 = (place_sequence_l3[start:end],)

            ## step-3: forward pass & "implicit" loss calculation
            loss = model(args_l1=args_l1, args_l2=args_l2, args_l3=args_l3)
            ## print(f"epoch [{epoch}] : loss = {loss}")
            wandb.log({"loss": loss})

            ## backward pass
            loss.backward()

            ## optimizer step
            optimizer.step()

        ## update the learning rate
        lr_scheduler.step()

        wandb.watch(model)

        ## visualize the trained GridCells weight
        fig = utils.visualize(model, epoch, "../plots")
        wandb.log({"trained weights": wandb.Image(fig)})


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ## device = torch.device("cpu")
    model = GridCells(ARGS)
    print(
        "....................................\n", dict(model.named_parameters()).keys()
    )
    print("....................................")
    print(model)

    ## training
    train(model, device)


if __name__ == "__main__":
    wandb.init(project="GridTr", entity="imflash217")
    main()
