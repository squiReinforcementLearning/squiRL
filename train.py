import numpy as np
import argparse
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
import pytorch_lightning as pl
from squiRL.vpg.vpg import VPGLightning


def main(hparams) -> None:
    seed_everything(42)

    wandb_logger = WandbLogger(project='vpg-lightning-test')
    model = VPGLightning(hparams)

    trainer = pl.Trainer(
        gpus=1,
        # distributed_backend='dp',
        max_epochs=2000,
        reload_dataloaders_every_epoch=False,
        logger=wandb_logger,
    )

    trainer.fit(model)


parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")
parser.add_argument("--eps",
                    type=float,
                    default=np.finfo(np.float32).eps.item(),
                    help="small offset")
parser.add_argument("--env",
                    type=str,
                    default="CartPole-v0",
                    help="gym environment tag")
parser.add_argument("--gamma",
                    type=float,
                    default=0.99,
                    help="discount factor")
parser.add_argument("--episode_length",
                    type=int,
                    default=200,
                    help="max length of an episode")
parser.add_argument("--max_episode_reward",
                    type=int,
                    default=200,
                    help="max episode reward in the environment")

args, _ = parser.parse_known_args()

main(args)
