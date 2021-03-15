"""Training script for interfacing with library. This script
can be used from the commandline/yaml to run any of the
algorithms in squiRL

Attributes:
    args (argparse.Namespace): Parsed config arguments for running script
    parser (argparse.ArgumentParser): Argument parser
"""
import os
import json
import argparse
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
import pytorch_lightning as pl
import squiRL
import gym
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
        hparams.logger.experiment
        profiler = None
        cwd = os.getcwd()
        path = os.path.join(cwd, 'models')
        if not os.path.exists(path):
            os.mkdir(path)
        path = os.path.join(path, hparams.logger.version)
        if not os.path.exists(path):
            os.mkdir(path)
        path = os.path.join(path, hparams.logger.version)
        if hparams.save_config:
            with open(path + '.json', 'wt') as f:
                config = vars(hparams).copy()
                config.pop("logger")
                config.pop("gpus")
                config.pop("tpu_cores")
                json.dump(config, f, indent=4)

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
    parser.add_argument(
        '--save_config',
        type=bool,
        default=True,
        help='Save settings to file in json format. Ignored in json file')
    parser.add_argument('--load_config',
                        type=str,
                        help='Load from json file. Command line override.')
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
    args, _ = parser.parse_known_args()
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
    args, remaining_args = parser.parse_known_args()
    env = gym.make(args.env)
    parser.add_argument('--observation_space',
                        type=int,
                        default=env.observation_space.shape[0],
                        help='env state space')
    parser.add_argument('--action_space',
                        type=int,
                        default=env.action_space.n,
                        help='env action space')
    parser.add_argument('--max_reward',
                        type=int,
                        default=env.spec.reward_threshold,
                        help='Max reward allowed')
    env.close()

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
    if args.load_config:
        with open(args.load_config, 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)

    print(args)

    train(args)
