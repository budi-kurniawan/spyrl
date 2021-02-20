#!/usr/bin/env python3
"""
    Cartpole learning with ActorCritic agent with traces
"""
import gym
import sys
sys.path.insert(0, "../spyrl")
from spyrl.activity.learning import Learning
from spyrl.activity.activity_config import ActivityConfig
from spyrl.listener.impl.basic_functions import BasicFunctions
from spyrl.agent_builder.impl.actor_critic_traces_agent_builder import ActorCriticTracesAgentBuilder
from example.cartpole.helper.cartpole_discretiser import CartpoleDiscretiser
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
    #env = GymEnvWrapper(env)
    config = ActivityConfig(num_episodes=10, out_path='result/cartpole/test4/')
    agent_builder = ActorCriticTracesAgentBuilder(num_actions, discretiser=CartpoleDiscretiser())
    learning = Learning(listener=BasicFunctions(render=False))
    learning.learn(env, agent_builder, config)