""" A class representing DQN agents for MORL and multiple neural networks, one for each reward"""

from spyrl.agent.torch_seedable_agent import TorchSeedableAgent
from spyrl.util.util import override
import numpy as np
import torch
from typing import List, Dict
from spyrl.activity.activity_context import ActivityContext
from spyrl.agent.impl.ppo_agent import PPOAgent

__author__ = 'Budi Kurniawan'


class MultiObjectivePPOAgent(TorchSeedableAgent):
    def __init__(self, nn_dims, normaliser, reward_builder, seed=None) -> None:
        super().__init__(seed)
        self.output_dim = nn_dims[-1]
        self.reward_builder = reward_builder
        num_rewards = self.num_rewards = reward_builder.get_num_rewards()
        self.normaliser = normaliser
        self.agents = [PPOAgent(nn_dims, normaliser, seed) for _ in range(num_rewards)]

    @override(TorchSeedableAgent)
    def clean_up(self)->None:
        [agent.clean_up() for agent in self.agents]
                
    def update(self, activity_context: ActivityContext, state:np.ndarray, action: int, reward: List[float], next_state:np.ndarray, terminal: bool, env_data: Dict[str, object]) -> None:
        rewards = self.reward_builder.get_rewards(env_data)
        self.total_rewards = np.add(rewards, self.total_rewards)
        self.total_redefined_reward += np.sum(rewards)
        for i in range(self.num_rewards):
            self.agents[i].update(activity_context, state, action, rewards[i], next_state, terminal, env_data)

    def save_policy(self, path): # used to save a policy that can be used for activity
        for i in range(self.num_rewards):
            path_i = path + '_' + str(i)
            self.agents[i].save_policy(path_i)
        
    def select_action(self, state: np.ndarray) -> int:
        normalised_state = state if self.normaliser is None else self.normaliser.normalise(state)
        q_value_list = []
        for agent in self.agents:
            with torch.no_grad():
                pi = agent.ac.pi._distribution(normalised_state)
                probs_2d = pi.probs.reshape(-1, pi._num_events)
                q_value_list.append(probs_2d.numpy()[0])
        sum_of_q_values = np.sum(q_value_list, axis=0)
        prob = self.softmax(sum_of_q_values)
        return self.np_random.choice(self.actions, p=prob)
