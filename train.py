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
from pytorch_lightning.profiler import AdvancedProfiler


def train(hparams) -> None:
    """Runs algorithm

    Args:
        hparams (argparse.Namespace): Stores all passed args
    """
    if hparams.debug:
        hparams.logger = None
        profiler = True
    else:
        hparams.logger = WandbLogger(project=hparams.project)
    seed_everything(hparams.seed)
    algorithm = squiRL.reg_algorithms[hparams.algorithm](hparams)
    trainer = pl.Trainer.from_argparse_args(hparams, profiler=profiler)
    trainer.fit(algorithm)


if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    parser = argparse.ArgumentParser(add_help=False)
    group_prog = parser.add_argument_group("program_args")
    group_env = parser.add_argument_group("environment_args")

    # add PROGRAM level args
    group_prog.add_argument('--seed',
                            type=int,
                            default=42,
                            help="experiment seed")
    group_prog.add_argument(
        '--debug',
        type=bool,
        default=False,
        help="stops logging to wandb, turns on profiler, sets num_workers "
        "to None, to allow debugging on a single thread")
    group_prog.add_argument('--algorithm',
                            type=str,
                            default='VPG',
                            help="DRL algorithm")
    args, remaining_args = parser.parse_known_args()
    group_alg = parser.add_argument_group(args.algorithm + "_args")
    group_prog.add_argument('--project',
                            type=str,
                            default=args.algorithm,
                            help="project name for wandb logs")

    # add environment specific args
    group_env.add_argument("--env",
                           type=str,
                           default="CartPole-v0",
                           help="gym environment tag")
    group_env.add_argument("--episode_length",
                           type=int,
                           default=200,
                           help="max length of an episode")
    group_env.add_argument("--max_episode_reward",
                           type=int,
                           default=200,
                           help="max episode reward in the environment")

    # add algorithm specific args
    group_alg = squiRL.reg_algorithms[args.algorithm].add_model_specific_args(
        group_alg)

    # add all the available trainer options to argparse
    parser = pl.Trainer.add_argparse_args(parser)

    # this is done to add all args to help
    parser = argparse.ArgumentParser(
        parents=[parser],
        epilog="Trainer args docs can be found at PyTorch Lightning.")

    args = parser.parse_args()
    args, _ = parser.parse_known_args()

    print(args)

    train(args)
