""" A BaseTesting is a test session"""
import os
from datetime import datetime
from spyrl.event.session_event import SessionEvent
from spyrl.event.trial_event import TrialEvent
from spyrl.event.episode_event import EpisodeEvent
from spyrl.event.step_event import StepEvent
from spyrl.util.stopper import Stopper
from spyrl.activity.activity_context import ActivityContext
from spyrl.activity.activity_config import ActivityConfig
from spyrl.activity.activity import Activity
from spyrl.tester_builder.tester_builder import TesterBuilder
class Testing(Activity):
    def __init__(self, **kwargs):
        super().__init__()
        listener = kwargs.get('listener', None)
        if listener is not None:
            self.add_listener(listener)
        listeners = kwargs.get('listeners', [])
        for listener in listeners:
            self.add_listener(listener)
    
    def test(self, env, tester_builder: TesterBuilder, config: ActivityConfig) -> None:
        end_trial = config.start_trial + config.num_trials
        activity_context = ActivityContext()
        activity_context.out_path = config.out_path
        activity_context.num_episodes = config.num_episodes
        self.fire_before_session_event(SessionEvent(activity_context))
    
        activity_context = ActivityContext()
        self.fire_before_session_event(SessionEvent(activity_context))
        for trial in range(config.start_trial, end_trial):
            trial_start_time = datetime.now()
            activity_context.trial = trial
            activity_context.trial_start_time = trial_start_time
            activity_context.trial_end_time = None
            self.fire_before_trial_event(TrialEvent(activity_context))
            seed = trial
            env.seed(seed)
            if tester_builder is not None:
                tester = tester_builder.create_tester(trial)
            tester.trial_start(activity_context)
            max_reward = 0
            max_num_steps = 0
            for episode in range(activity_context.start_episode, activity_context.start_episode + config.num_episodes):
                activity_context.episode = episode
                stopper = Stopper()
                self.fire_before_episode_event(EpisodeEvent(activity_context, tester=tester, env=env, stopper=stopper))
                state = env.reset()
                self.fire_after_env_reset_event(EpisodeEvent(activity_context, tester=tester, env=env, stopper=stopper))
                tester.episode_start(activity_context)
                ep_reward = 0.0
                step = 0
                while True:
                    step += 1
                    activity_context.step = step
                    self.fire_before_step_event(StepEvent(activity_context, env=env))
                    action = tester.select_action(state)
                    next_state, reward, terminal, env_data = env.step(action)
                    state = next_state
                    ep_reward += reward
                    self.fire_after_step_event(StepEvent(activity_context, env=env, reward=reward, 
                                                         state=state, action=action, env_data=env_data))
                    if terminal:
                        break
                if max_reward < ep_reward:
                    max_reward = ep_reward
                
                if max_num_steps < step:
                    max_num_steps = step
                    self.fire_max_num_steps_event(EpisodeEvent(activity_context))
                
                self.fire_after_episode_event(EpisodeEvent(activity_context, reward=ep_reward,
                        avg_reward=(ep_reward/step), tester=tester, env=env, stopper=stopper))
                tester.episode_end(activity_context)
                if stopper.active:
                    break
            tester.trial_end(activity_context)
            trial_end_time = datetime.now()
            activity_context.trial_end_time = trial_end_time
            self.fire_after_trial_event(TrialEvent(activity_context, trial_start_time=trial_start_time, trial_end_time=trial_end_time))
        self.fire_after_session_event(SessionEvent(activity_context))