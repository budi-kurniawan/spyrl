import pickle
import numpy as np
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from spyrl.util.util import override
from spyrl.activity.activity_context import ActivityContext
from spyrl.agent.torch_seedable_agent import TorchSeedableAgent
from spyrl.agent.impl.ppo.core import MLPActorCritic
from spyrl.agent.impl.ppo.ppo_buffer import PPOBuffer
from spyrl.agent.impl.ppo.mpi_tools import mpi_avg
from spyrl.agent.impl.ppo.mpi_pytorch import mpi_avg_grads


""" A Torch-based class representing PPO agents 
    This is a modified version of https://github.com/openai/spinningup/blob/20921137141b154454c0a2698709d9f9a0302101/spinup/algos/pytorch/ppo/ppo.py
"""

__author__ = 'bkurniawan'

class PPOAgent(TorchSeedableAgent):
    #def __init__(self, memory_size, batch_size, dqn_dims, normaliser, seed=None) -> None:
    def __init__(self, observation_space, action_space, local_steps_per_epoch, seed=None) -> None:
        super().__init__(seed)
        gamma = 0.99
        lam = 0.97
        pi_lr = 3e-4
        vf_lr = 1e-3
        self.local_steps_per_epoch = local_steps_per_epoch
#         epochs=50, ,
#         max_ep_len=1000,
#         
        obs_dim = observation_space.shape
        act_dim = action_space.shape

        self.ac = MLPActorCritic(observation_space, action_space)
        self.buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)
        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)

    @override(TorchSeedableAgent)
    def update(self, activity_context: ActivityContext, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, terminal: bool, env_data: Dict[str, object]) -> None:
        self.buf.store(state, action, reward, self.v, self.logp)
        epoch_ended = activity_context.total_steps % self.local_steps_per_epoch == 0
        timeout = activity_context.step % 1000 == 0
        if terminal or timeout or epoch_ended:
            if timeout or epoch_ended:
                _, v, _ = self.ac.step(torch.as_tensor(next_state, dtype=torch.float32))
            else:
                v = 0
            self.buf.finish_path(v)
        if epoch_ended:
            self.ppo_update()

    def ppo_update(self):
        train_pi_iters = 80
        train_v_iters = 80
        target_kl = 0.01
        data = self.buf.get()

        pi_l_old, pi_info_old = self.compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                break
            loss_pi.backward()
            mpi_avg_grads(self.ac.pi)    # average grads across MPI processes
            self.pi_optimizer.step()


        # Value function learning
        for i in range(train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(self.ac.v)    # average grads across MPI processes
            self.vf_optimizer.step()

    def save_model(self, path): # used to save a model for intermediate learning
        # See https://pytorch.org/tutorials/beginner/saving_loading_models.html
        # TODO, need to save random generators too
        torch.save({
            'model_state_dict': self.dqn.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'loss': self.loss,
            'memory': self.memory
        }, path)

    @override(TorchSeedableAgent)
    def save_policy(self, path): # used to save a policy that can be used for activity
        pass
#         file = open(path, 'wb')
#         pickle.dump(self.dqn, file)
#         file.close()
        
    def load_model(self, path):
        pass

    @override(TorchSeedableAgent)
    def select_action(self, state: np.ndarray) -> int:
        a, v, logp = self.ac.step(torch.as_tensor(state, dtype=torch.float32))
        self.v = v
        self.logp = logp
        return a

    def compute_loss_pi(self, data):
        clip_ratio = 0.2
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']
        return ((self.ac.v(obs) - ret)**2).mean()