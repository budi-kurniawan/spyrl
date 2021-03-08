import numpy as np
import torch
from spyrl.agent.impl.dqn.dqn import DQN
from spyrl.agent.impl.dqn_agent import DQNAgent
from spyrl.util.util import override
from spyrl.activity.activity_context import ActivityContext

class DoubleDQNAgent(DQNAgent):

    def __init__(self, memory_size, batch_size, dqn_dims, normaliser, seed=None) -> None:
        super().__init__(memory_size, batch_size, dqn_dims, normaliser, seed)
        self.dqn1 = self.dqn
        self.dqn2 = DQN(dqn_dims)
    
    @override(DQNAgent)
    def episode_start(self, activity_context: ActivityContext):
        super().episode_start(activity_context)
        self.dqn2.load_state_dict(self.dqn.state_dict()) # copy weights from dqn1 to dqn2

    @override(DQNAgent)
    def select_action(self, state: np.ndarray) -> int:
        if self.np_random.random() < self.current_epsilon:
            return self.np_random.choice(self.output_dim)
        else:
            r = self.random_0_or_1()
            self.dqn = self.dqn1 if r == 0 else self.dqn2
            self.dqn.train(mode=False)
            q_values = self.get_Q(state) if self.normaliser is None else self.get_Q(self.normaliser.normalise(state))
            return int(torch.argmax(q_values))

    @override(DQNAgent)
    def train(self) -> None:
        if len(self.memory) <= self.batch_size:
            return
        minibatch = self.memory.pop(self.batch_size)
        normalised_states = np.vstack([x.state for x in minibatch])
        actions = np.array([x.action for x in minibatch])
        rewards = np.array([x.reward for x in minibatch])
        normalised_next_states = np.vstack([x.next_state for x in minibatch])
        done = np.array([x.done for x in minibatch])

        r = self.random_0_or_1()
        Q_predict = self.get_Q1(normalised_states) if r == 0 else self.get_Q2(normalised_states)
        Q_target = Q_predict.clone().data.numpy()
        if r == 0:
            Q_target[np.arange(len(Q_target)), actions] = rewards + self.gamma * np.max(self.get_Q2(normalised_next_states).data.numpy(), axis=1) * ~done
        else:
            Q_target[np.arange(len(Q_target)), actions] = rewards + self.gamma * np.max(self.get_Q1(normalised_next_states).data.numpy(), axis=1) * ~done
        Q_target = torch.Tensor(Q_target)
        return self._train(Q_predict, Q_target)

    def get_Q1(self, states: np.ndarray):
        self.dqn = self.dqn1
        return self.get_Q(states)

    def get_Q2(self, states: np.ndarray):
        self.dqn = self.dqn2
        return self.get_Q(states)
    
    def random_0_or_1(self):
        return self.np_random.integers(2) # return 0 or 1