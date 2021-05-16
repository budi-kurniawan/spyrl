#!/usr/bin/env python3
"""
    Cartpole learning with PPO
"""
import gym
import sys
sys.path.insert(0, "../spyrl")
from spyrl.agent.impl.ppo_agent import PPOAgent
from spyrl.agent_builder.impl.pass_through_agent_builder import PassThroughAgentBuilder
from spyrl.activity.learning import Learning
from spyrl.activity.activity_config import ActivityConfig
from spyrl.listener.impl.basic_functions import BasicFunctions
from example.cartpole.helper.env_wrapper import GymEnvWrapper

__author__ = "Budi Kurniawan"
__copyright__ = "Copyright 2021, Budi Kurniawan"
__license__ = "GPL"
__version__ = "0.1.0"

if __name__ == '__main__':
    id = 'CartPole-v2'
    gym.envs.register(
        id=id,
        entry_point='gym.envs.classic_control:CartPoleEnv',
        max_episode_steps=100_000
    )
    env = gym.make(id)
    num_actions = env.action_space.n
    num_states = env.observation_space.shape[0]
    
    nn_dims = (num_states, 64, 64, num_actions)
    config = ActivityConfig(num_episodes=750, out_path='result/cartpole/ppo_test01/')
    
    agent = PPOAgent(nn_dims)
    agent_builder = PassThroughAgentBuilder(agent)
    learning = Learning(listener=BasicFunctions(render=False))
    learning.learn(env, agent_builder, config)