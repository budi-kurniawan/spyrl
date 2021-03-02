#!/usr/bin/env python3
"""
    LunarLander learning with ActorCritic agent with traces
"""
import gym
import sys
sys.path.insert(0, "../spyrl")
from spyrl.activity.learning import Learning
from spyrl.activity.activity_config import ActivityConfig
from spyrl.listener.impl.basic_functions import BasicFunctions
from spyrl.agent_builder.impl.actor_critic_traces_agent_builder import ActorCriticTracesAgentBuilder
from example.lunarlander.helper.lunarlander_discretiser import LunarLanderDiscretiser

__author__ = "Budi Kurniawan"
__copyright__ = "Copyright 2021, Budi Kurniawan"
__license__ = "GPL"
__version__ = "0.1.0"

if __name__ == '__main__':
    id = 'LunarLander-v2'
#     gym.envs.register(
#         id=id,
#         entry_point='gym.envs.classic_control:CartPoleEnv',
#         max_episode_steps=100_000
#     )
    env = gym.make(id)
    num_actions = env.action_space.n
    print('num_actions:', num_actions)
    config = ActivityConfig(num_episodes=10_000, out_path='result/lunarlander/test1/')
    agent_builder = ActorCriticTracesAgentBuilder(num_actions, discretiser=LunarLanderDiscretiser())
    learning = Learning(listener=BasicFunctions(render=True))
    learning.learn(env, agent_builder, config)