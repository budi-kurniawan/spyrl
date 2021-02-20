""" A class representing RL learning"""
from datetime import datetime
import os
from spyrl.event.session_event import SessionEvent
from spyrl.event.trial_event import TrialEvent
from spyrl.event.episode_event import EpisodeEvent
from spyrl.event.step_event import StepEvent
from spyrl.activity.activity import Activity
from spyrl.activity.activity_context import ActivityContext
from spyrl.agent_builder.agent_builder import AgentBuilder
from spyrl.activity.activity_config import ActivityConfig

class Learning(Activity):
    def __init__(self, **kwargs):
        super().__init__()
        listener = kwargs.get('listener', None)
        if listener is not None:
            self.add_listener(listener)
        listeners = kwargs.get('listeners', [])
        for listener in listeners:
            self.add_listener(listener)
    
    def learn(self, env, agent_builder: AgentBuilder, config: ActivityConfig) -> None:
        end_trial = config.start_trial + config.num_trials
        activity_context = ActivityContext()
        activity_context.out_path = config.out_path
        activity_context.num_episodes = config.num_episodes
        self.fire_before_session_event(SessionEvent(activity_context))
        
        for trial in range(config.start_trial, end_trial):
            trial_start_time = datetime.now()
            activity_context.trial = trial
            activity_context.trial_start_time = trial_start_time
            activity_context.trial_end_time = None
            self.fire_before_trial_event(TrialEvent(activity_context)) # allows a listener to change start_episode and search for a policy path
            seed = trial
            env.seed(seed)
            agent = agent_builder.load_or_create_agent(activity_context, seed)
            agent.trial_start(activity_context)
            max_reward = 0
            max_num_steps = 0
            for episode in range(activity_context.start_episode, config.num_episodes + 1):
                activity_context.episode = episode
                # env.reset() and fire_before_episode_event must be reversed, just like in Testing.test()
                self.fire_before_episode_event(EpisodeEvent(activity_context, agent=agent, env=env))
                state = env.reset()
                self.fire_after_env_reset_event(EpisodeEvent(activity_context, agent=agent, env=env))
                agent.episode_start(activity_context)
                ep_reward = 0.0
                step = 1
                while True:
                    activity_context.step = step
                    self.fire_before_step_event(StepEvent(activity_context, env=env))
                    action = agent.select_action(state)
                    next_state, reward, terminal, env_data = env.step(action)
                    agent.update(activity_context, state, action, reward, next_state, terminal, env_data)
                    state = next_state
                    ep_reward += reward
                    self.fire_after_step_event(StepEvent(activity_context, env=env, reward=reward, agent=agent))
                    if terminal:
                        break
                    step += 1
                if max_reward < ep_reward:
                    max_reward = ep_reward
                
                if max_num_steps < step:
                    max_num_steps = step
                    self.fire_max_num_steps_event(EpisodeEvent(activity_context, agent=agent, env=env))
                
                agent.episode_end(activity_context)
                self.fire_after_episode_event(EpisodeEvent(activity_context, reward=ep_reward,
                        avg_reward=(ep_reward/step), agent=agent, env=env))
            agent.trial_end(activity_context)
            trial_end_time = datetime.now()
            activity_context.trial_end_time = trial_end_time
            self.fire_after_trial_event(TrialEvent(activity_context, agent=agent, env=env))
        self.fire_after_session_event(SessionEvent(activity_context))