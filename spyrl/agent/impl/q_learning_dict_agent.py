import numpy as np
import pickle
from spyrl.agent.seedable_agent import SeedableAgent

""" An implementation of Q-learning agents using dict instead of array. Use this when the number of states is not known or 
hard to predict """
from spyrl.util.util import override
from spyrl.activity.activity_context import ActivityContext

__author__ = 'bkurniawan'

class QLearningDictAgent(SeedableAgent):
    EPSILON = 0.2
    ALPHA = 0.7
    GAMMA = 0.95
    LAMBDA = 0.9    
    
    def __init__(self, num_actions: int, discretiser, seed=None, initial_policy_path=None):
        super().__init__(seed)
        self.discretiser = discretiser
        #self.num_states = discretiser.get_num_discrete_states()
        self.num_actions = num_actions
        if initial_policy_path is not None:
            self.load_policy(initial_policy_path)
        else:
            self.q = {} #np.zeros([self.num_states, num_actions], dtype=np.float64)

    @override(SeedableAgent)
    def episode_start(self, activity_context: ActivityContext) -> None:
        #min_eps = 0.01
        #slope = (min_eps - 1.0) / (activity_context.num_episodes - 1)
        #self.current_epsilon = max(slope * activity_context.episode + 1.0, min_eps)
        self.current_epsilon = (activity_context.num_episodes - activity_context.episode) * QLearningDictAgent.EPSILON / (activity_context.num_episodes - 1)


    def select_action(self, state: int) -> int:
        if self.np_random.random() < self.current_epsilon:
            return self.np_random.choice(self.num_actions)
        else:
            discrete_state = self.discretiser.discretise(state)
            return self.__get_action_with_max_q_value(discrete_state)

    def update(self, activity_context, state, action, reward, next_state, terminal, env_data) -> None:
        q = self.q
        discrete_state = self.discretiser.discretise(state)
        # Though select_action has been called with the same state, it may have been exploring, so q may not have it as a key
        if discrete_state not in q:
            q[discrete_state] = np.zeros(self.num_actions)
        
        next_discrete_state = self.discretiser.discretise(next_state)
        if next_discrete_state not in q:
            q[next_discrete_state] = np.zeros(self.num_actions)
        next_max = np.max(q[next_discrete_state])
        delta = reward + QLearningDictAgent.GAMMA * next_max - q[discrete_state][action]
        q[discrete_state][action] += QLearningDictAgent.ALPHA * delta

    def save_policy(self, path) -> None:
        file = open(path, 'wb')
        pickle.dump(self.q, file)
        file.close()
        
    def load_policy(self, path) -> None:
        file = open(path,'rb')
        self.q = pickle.load(file)
        file.close()
        
    # return the index with the greatest Q for the given state, if there are duplicates, pick one at random
    def __get_action_with_max_q_value(self, discrete_state):
        array = self.q.get(discrete_state, np.zeros(self.num_actions))
        best_actions = np.argwhere(array==np.max(array)).flatten()
        return self.np_random.choice(best_actions)