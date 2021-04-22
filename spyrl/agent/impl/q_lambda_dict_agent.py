from spyrl.util.util import override
from spyrl.agent.impl.q_learning_dict_agent import QLearningDictAgent
import numpy as np
from spyrl.activity.activity_context import ActivityContext

__author__ = 'bkurniawan'

"""
This class represents a Q(lambda) or Q-learning with eligibility traces agent, using replacing traces 
(instead of accumulating traces). As such, e[s][a] is set to 1, instead of e[s][a] *= increment. 
Therefore, we do not need another 2-dimensional array. Rather, we'll use visited to make the code faster

This algorithm is called Watkin's Q(lambda) (Q-learning + eligibility traces) and can be found in
    https://stackoverflow.com/questions/40862578/how-to-understand-watkinss-q%CE%BB-learning-algorithm-in-suttonbartos-rl-book
    
Do not use http://www-anw.cs.umass.edu/~barto/courses/cs687/Chapter%207.pdf (wrong)
"""
class QLambdaDictAgent(QLearningDictAgent):
    def __init__(self, num_actions: int, discretiser, seed=None, initial_policy_path=None):
        super().__init__(num_actions, discretiser, seed, initial_policy_path)
        self.e = {} #np.zeros([self.num_states, num_actions], dtype=np.float64)
        self.visited = []

    @override(QLearningDictAgent)
    def episode_start(self, activity_context: ActivityContext)->None:
        super().episode_start(activity_context)
        self.reset_traces()

    @override(QLearningDictAgent)
    def update(self, activity_context, state, action, reward, next_state, terminal, env_data) -> None:
        # this is a better implementation than the one in Sutton's book (1st edition) because you don't need to
        # calculate a' (next action)
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

        # we want to know if action was obtained by exploration or exploitation
        exploit = q[discrete_state][action] == np.max(q[discrete_state])
        if exploit:
            e = self.e
            if discrete_state not in e:
                e[discrete_state] = np.zeros(self.num_actions)
            for s, a in self.visited:
                q[s][a] += QLearningDictAgent.ALPHA * delta * e[s][a]
                e[s][a] *= self.GAMMA * self.LAMBDA
            e[discrete_state][action] = 1
            if (discrete_state, action) not in self.visited:
                self.visited.append((discrete_state, action)) # record visited state/action pairs so we don't have to update state/actions that were never visited
        else:
            self.reset_traces()

    def reset_traces(self):
        self.e.clear()
        del self.visited[:]