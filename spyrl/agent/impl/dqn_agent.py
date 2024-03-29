import pickle
import numpy as np
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
from spyrl.util.util import override
from spyrl.activity.activity_context import ActivityContext
from spyrl.agent.torch_seedable_agent import TorchSeedableAgent
from spyrl.agent.impl.dqn.dqn import DQN, ReplayMemory

""" A Torch-based class representing DQN agents """

__author__ = 'bkurniawan'

class DQNAgent(TorchSeedableAgent):
    def __init__(self, memory_size, batch_size, dqn_dims, normaliser, seed=None) -> None:
        super().__init__(seed)
        self.memory = ReplayMemory(memory_size, self.random)
        self.batch_size = batch_size
        self.dqn = DQN(dqn_dims)
        self.input_dim = dqn_dims[0]
        self.output_dim = dqn_dims[-1]
        self.loss_fn = nn.MSELoss()
        self.optim = optim.Adam(self.dqn.parameters())
        self.normaliser = normaliser
        self.gamma = 0.99

    @override(TorchSeedableAgent)
    def update(self, activity_context: ActivityContext, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, terminal: bool, env_data: Dict[str, object]) -> None:
        self.add_sample(state, action, reward, next_state, terminal)
        self.train()
        
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
        pickle.dump(self.dqn, file)
        file.close()
        
    def load_model(self, path):
        # See https://pytorch.org/tutorials/beginner/saving_loading_models.html
        self.dqn = DQN(self.input_dim, self.output_dim, self.hidden_dim)
        checkpoint = torch.load(path)
        self.dqn.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss = checkpoint['loss']
        self.memory = checkpoint['memory']
        self.dqn.eval()

    @override(TorchSeedableAgent)
    def select_action(self, state: np.ndarray) -> int:
        if self.np_random.random() < self.current_epsilon:
            return self.np_random.choice(self.output_dim)
        else:
            self.dqn.train(mode=False)
            if self.normaliser is not None:
                state = self.normaliser.normalise(state)
            q_values = self.get_Q(state)
            return int(torch.argmax(q_values))

    # Returns rows of values. Each row has n columns, where n = output_dim or num_actions
    # Each column contains the value for the corresponding action. For example, the first column
    # contains the Q-value of the first action given the input state.
    def get_Q(self, normalised_states: np.ndarray) -> torch.FloatTensor:
        # normalised_states is a tensor of shape (len(batch_size), input_dim) when called from train()
        # or of shape (1, input_dim) when called from select_action()
        normalised_states = torch.Tensor(normalised_states.reshape(-1, self.input_dim))
        self.dqn.train(mode=False)
        return self.dqn(normalised_states)
    
    def add_sample(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        if self.normaliser is None:
            self.memory.push(state, action, reward, next_state, done)
        else:
            normalised_state = self.normaliser.normalise(state)
            normalised_next_state = self.normaliser.normalise(next_state)
            self.memory.push(normalised_state, action, reward, normalised_next_state, done)
        
    def train(self) -> None:
        if len(self.memory) <= self.batch_size:
            return
        minibatch = self.memory.pop(self.batch_size)
        normalised_states = np.vstack([x.state for x in minibatch])
        actions = np.array([x.action for x in minibatch])
        rewards = np.array([x.reward for x in minibatch])
        normalised_next_states = np.vstack([x.next_state for x in minibatch])
        done = np.array([x.done for x in minibatch])

        # Q_predict and Q_target are numpy.ndarrays of size (len(minibatch), num_actions).
        Q_predict = self.get_Q(normalised_states)
        Q_target = Q_predict.clone().data.numpy()
        """ Update exactly one column of every row in Q_target.
            Q_target[np.arange(len(Q_target)), actions]) is a 1-dim array of size len(minibatch) and each element is selected from
            the corresponding row. Which cell in the row is used depends on the value of the corresponding actions
            For example, suppose len(minibatch) = 2 and num_actions=3 and Q_target is [[1, 2, 3], [4, 5, 6]] and actions = [1, 0]
            Q_target[np.arange(len(Q_target)), actions]) is then a 1-dim array of size len(minibatch) -> [2, 4], 
            where 2 is taken from the 1st element of [1,2,3] and 4 from the zeroth of [4,5,6].
            However, more importantly here, Q_target[np.arange(len(Q_target)), actions]) represents locations whose values are to be replaced
        """
        Q_target[np.arange(len(Q_target)), actions] = self.get_q_update(rewards, normalised_next_states, done)
        Q_target = torch.Tensor(Q_target)
        """
            Exactly one cell in each row in Q_target has been updated. In other words, the nth row of Q_predict and 
            the nth row of Q_target differs by one value
        """
        return self._train(Q_predict, Q_target)
    
    def get_q_update(self, rewards: np.ndarray, normalised_next_states: np.ndarray, done: np.ndarray):
        return rewards + self.gamma * np.max(self.get_Q(normalised_next_states).data.numpy(), axis=1) * ~done

    def _train(self, Q_pred: torch.FloatTensor, Q_true: torch.FloatTensor) -> float:
        """Computes loss and backpropagation
        Args:
            Q_pred (torch.FloatTensor): Predicted value by the network,
                2-D Tensor of shape(n, output_dim)
            Q_true (torch.FloatTensor): Target value obtained from the game,
                2-D Tensor of shape(n, output_dim)
        Returns:
            float: loss value
        """
        self.dqn.train(mode=True)
        self.optim.zero_grad()
        loss = self.loss_fn(Q_pred, Q_true)
        loss.backward()
        self.optim.step()
        self.loss = loss
        return loss
    
    @override(TorchSeedableAgent)
    def episode_start(self, activity_context: ActivityContext):
        min_eps = 0.01
        slope = (min_eps - 1.0) / (activity_context.num_episodes - 1)
        self.current_epsilon = max(slope * activity_context.episode + 1.0, min_eps)
            
    def get_epsilon(self):
        return self.current_epsilon