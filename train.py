"""Training script for interfacing with library. This script
can be used from the commandline/yaml to run any of the
algorithms in squiRL

Attributes:
    args (argparse.Namespace): Parsed config arguments for running script
    parser (argparse.ArgumentParser): Argument parser
"""
import argparse
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
import pytorch_lightning as pl
import squiRL


def main(hparams) -> None:
    """Runs algorithm

    Args:
        hparams (argparse.Namespace): Stores all passed args
    """
    if hparams.debug:
        hparams.logger = None
        hparams.profiler = True
        hparams.num_workers = 1
    else:
        hparams.logger = WandbLogger(project=hparams.project)
    seed_everything(hparams.seed)
    algorithm = squiRL.reg_algorithms[hparams.algorithm](hparams)
    trainer = pl.Trainer.from_argparse_args(hparams)
    trainer.fit(algorithm)


parser = argparse.ArgumentParser()

# add PROGRAM level args
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--algorithm', type=str, default='VPG')
args, _ = parser.parse_known_args()
parser.add_argument('--project', type=str, default=args.algorithm)

# add environment specific args
parser.add_argument("--env",
                    type=str,
                    default="CartPole-v0",
                    help="gym environment tag")
parser.add_argument("--episode_length",
                    type=int,
                    default=200,
                    help="max length of an episode")
parser.add_argument("--max_episode_reward",
                    type=int,
                    default=200,
                    help="max episode reward in the environment")

# add algorithm specific args
parser = squiRL.reg_algorithms[args.algorithm].add_model_specific_args(parser)

# add all the available trainer options to argparse
parser = pl.Trainer.add_argparse_args(parser)

args = parser.parse_args()
args, _ = parser.parse_known_args()

main(args)
