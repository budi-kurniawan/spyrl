#!/usr/bin/env python3
import gym
import sys
sys.path.insert(0, "../../../spyrl")
from spyrl.activity.learning import Learning
from spyrl.activity.activity_config import ActivityConfig
from spyrl.agent_builder.impl.actor_critic_traces_agent_builder import ActorCriticTracesAgentBuilder
from spyrl.listener.impl.basic_functions import BasicFunctions
from spyrl.listener.impl.file_log_listener import RewardType
from example.lunarlander.helper.lunarlander_discretiser import LunarLanderDiscretiser24576

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    num_actions = env.action_space.n
    config = ActivityConfig(start_trial=1, num_trials=1, num_episodes=100, out_path='results/acet-21/')
    agent_builder = ActorCriticTracesAgentBuilder(num_actions, discretiser=LunarLanderDiscretiser24576())
    milestone_episodes = [1000, 2000, 4000, 5000, 8000]
    learning = Learning(listener=BasicFunctions(render=False, draw=False, reward_type=RewardType.TOTAL, 
                        milestone_episodes=milestone_episodes))
    learning.learn(env, agent_builder, config)