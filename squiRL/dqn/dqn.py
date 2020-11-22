import argparse

import gym3
import pytorch_lightning as pl

from squiRL.common import reg_policies
from squiRL.common.agents import GreedyAgent
from squiRL.common.data_stream import ExperienceReplayBuffer


class DQN(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace) -> None:
        super(DQN, self).__init__()
        self.hparams = hparams
        self.save_hyperparameters()

        self.env = gym3.vectorize_gym(num=self.hparams.num_envs,
                                      env_kwargs={"id": self.hparams.env})
        self.eps = self.hparams.eps
        obs_size = self.env.obs_space.shape
        action_size = self.env.ac_space.eltype.n

        self.net = reg_policies[self.hparams.policy](obs_size, action_size)
        self.replay_buffer = ExperienceReplayBuffer(self.hparams.buffer_size, self.hparams.batch_size)

        self.agent = GreedyAgent(self.env, self.replay_buffer)
