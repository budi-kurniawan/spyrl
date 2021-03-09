import numpy as np
import torch
from spyrl.agent.impl.dqn_agent import DQNAgent
from spyrl.agent.impl.dqn.dqn import DQN
from spyrl.util.util import override
from typing import Dict
from spyrl.activity.activity_context import ActivityContext

class DQNWithTargetNetworkAgent(DQNAgent):
    def __init__(self, memory_size, batch_size, dqn_dims, normaliser, target_refresh_interval, seed=None) -> None:
        super().__init__(memory_size, batch_size, dqn_dims, normaliser, seed)
        self.dqn2 = DQN(dqn_dims) # target network
        self.target_refresh_interval = target_refresh_interval
        self.count = 0

    @override(DQNAgent)
    def update(self, activity_context: ActivityContext, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, terminal: bool, env_data: Dict[str, object]) -> None:
        if self.count % self.target_refresh_interval == 0:
            self.dqn2.load_state_dict(self.dqn.state_dict()) # copy weights from dqn1 to dqn2
            self.count = 0
        super().update(activity_context, state, action, reward, next_state, terminal, env_data)
        self.count += 1
    
    def get_Q2(self, normalised_states: np.ndarray) -> torch.FloatTensor:
        normalised_states = torch.Tensor(normalised_states.reshape(-1, self.input_dim))
        self.dqn2.train(mode=False)
        return self.dqn2(normalised_states)
    
    @override(DQNAgent)
    def get_q_update(self, rewards: np.ndarray, normalised_next_states: np.ndarray, done: np.ndarray):
        return rewards + self.gamma * np.max(self.get_Q2(normalised_next_states).data.numpy(), axis=1) * ~done