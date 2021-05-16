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
from spyrl.agent.agent import Agent


""" A Torch-based class representing PPO agents 
    Inspired by https://github.com/openai/spinningup/blob/20921137141b154454c0a2698709d9f9a0302101/spinup/algos/pytorch/ppo/ppo.py
"""

__author__ = 'bkurniawan'

class PPOAgent(TorchSeedableAgent):
    def __init__(self, nn_dims, normaliser=None, seed=None, **kwargs) -> None:
        super().__init__(seed)
        self.local_steps_per_epoch = kwargs.get('local_steps_per_epoch', 4000)
        gamma = kwargs.get('gamma', 0.99)
        lam = kwargs.get('lam', 0.97)
        pi_lr = kwargs.get('pi_lr', 3e-4)
        vf_lr = kwargs.get('vf_lr', 1e-3)
        self.train_pi_iters = kwargs.get('train_pi_iters', 80)
        self.train_v_iters = kwargs.get('train_v_iters', 80)
        self.target_kl = kwargs.get('target_kl', 0.01)
        self.clip_ratio = kwargs.get('clip_ratio', 0.2)
        self.max_ep_len = kwargs.get('max_ep_len', 1000)
        num_states = nn_dims[0]
        num_actions = nn_dims[-1]
        hidden_sizes = nn_dims[1:-1]
        
        self.ac = MLPActorCritic(num_states, num_actions, hidden_sizes)
        self.buf = PPOBuffer(num_states, self.local_steps_per_epoch, gamma, lam)
        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)

    @override(TorchSeedableAgent)
    def update(self, activity_context: ActivityContext, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, terminal: bool, env_data: Dict[str, object]) -> None:
        self.buf.store(state, action, reward, self.v, self.logp)
        epoch_ended = activity_context.total_steps % self.local_steps_per_epoch == 0
        timeout = activity_context.step % self.max_ep_len == 0
        if terminal or timeout or epoch_ended:
            if timeout or epoch_ended:
                _, v, _ = self.ac.step(torch.as_tensor(next_state, dtype=torch.float32))
            else:
                v = 0
            self.buf.finish_path(v)
        if epoch_ended:
            self.ppo_update()

    def ppo_update(self):
        data = self.buf.get()
        pi_l_old, pi_info_old = self.compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for _ in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * self.target_kl:
                break
            loss_pi.backward()
            #mpi_avg_grads(self.ac.pi)    # average grads across MPI processes
            self.pi_optimizer.step()

        # Value function learning
        for _ in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            #mpi_avg_grads(self.ac.v)    # average grads across MPI processes
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
        file = open(path, 'wb')
        pickle.dump(self.ac, file)
        file.close()
        
    @override(Agent)
    def load_policy(self):
        file = open(self.policy_path, 'rb')
        self.ac = pickle.load(file)
        file.close()    

    @override(TorchSeedableAgent)
    def select_action(self, state: np.ndarray) -> int:
        a, v, logp = self.ac.step(torch.as_tensor(state, dtype=torch.float32))
        self.v = v
        self.logp = logp
        return a

    def compute_loss_pi(self, data):
        clip_ratio = self.clip_ratio
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