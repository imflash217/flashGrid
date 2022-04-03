"""
## 🎯
## Projects: Exploring GridCells for Object Detection
## Author: Vinay Kumar
## Guide: Dr. Tianfu Wu
## Place: North Carolina State University
## Time: October 2021
##
"""

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dim_lattice", type=int, default=40, help="Dimension of the SQUARE lattice"
)
parser.add_argument(
    "--num_multiverse", type=int, default=16, help="Number of multiverse"
)
parser.add_argument(
    "--dim_universe",
    type=int,
    default=6,
    help="Dimension of each universe. i.e. num of gridcells per universe",
)
parser.add_argument("--std", type=float, default=0.08, help="std of gaussian kernel")

parser.add_argument("--beta1", type=float, default=0.9, help="Beta_1 in ADAM")
parser.add_argument(
    "--not_multiverse", type=bool, default=False, help="True if in testing phase"
)
parser.add_argument(
    "--shape_lattice", type=str, default="square", help="square / circle / triangle"
)
parser.add_argument(
    "--max_velocity_l2",
    type=float,
    default=3.0,
    help="Maximum velocity in Loss2 (measured in grid steps)",
)
parser.add_argument(
    "--min_velocity_l2",
    type=float,
    default=0.0,
    help="minimum velocity in Loss2 (measured in grids steps)",
)
parser.add_argument("--lamda_l1", type=float, default=1)
parser.add_argument("--lamda_l2", type=float, default=0.1)
parser.add_argument("--lamda_l3", type=float, default=5000)
parser.add_argument("--lamda_reg", type=float, default=9)
parser.add_argument(
    "--continuous_motion_type",
    type=bool,
    default=True,
    help="continuos / discrete motion-type",
)
parser.add_argument("--num_steps", type=int, default=1)
parser.add_argument("--GE", type=float, default=1.0)
parser.add_argument("--lr", type=float, default=3e-3)
parser.add_argument("--mom", type=float, default=0.99)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--batch_size", type=int, default=30000)
parser.add_argument("--num_epochs", type=int, default=2)
parser.add_argument("--num_data", type=int, default=30000)
parser.add_argument("--data_type", type=int, default=1)
parser.add_argument("--alpha", type=float, default=72.0)

parser.add_argument(
    "--gpus",
    type=list,
    default=[1],
    help="The list of GPUs to use for distributed training",
)
ARGS = parser.parse_args()
