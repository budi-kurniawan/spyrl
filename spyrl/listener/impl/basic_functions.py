from spyrl.listener.trial_listener import TrialListener
from spyrl.listener.episode_listener import EpisodeListener
from spyrl.util.util import override
from spyrl.listener.step_listener import StepListener
from spyrl.listener.impl.console_log_listener import ConsoleLogListener
from spyrl.listener.impl.file_log_listener import FileLogListener, RewardType
from spyrl.listener.impl.policy_maker import PolicyMaker
from spyrl.listener.impl.renderer import Renderer
from spyrl.listener.session_listener import SessionListener

class BasicFunctions(SessionListener, TrialListener, EpisodeListener, StepListener):
    def __init__(self, **kwargs):
        self.reward_type = kwargs.get('reward_type', RewardType.AVERAGE)
        self.chart_offset = kwargs.get('chart_offset', 0)
        self.top_n = kwargs.get('top_n', 0)
        self.min_recorded_reward = kwargs.get('min_recorded_reward', 0.1)
        self.milestone_episodes = kwargs.get('milestone_episodes', [])
        self.use_num_steps_as_reward = kwargs.get('use_num_steps_as_reward', False)
        self.save_agent = kwargs.get('save_agent', False)
        self.render = kwargs.get('render', False)        
        self.console_log_listener = ConsoleLogListener()
        self.file_log_listener = FileLogListener(self.chart_offset, self.reward_type)
        self.policy_maker = PolicyMaker(self.top_n, self.min_recorded_reward, self.milestone_episodes)
        if self.render:
            self.renderer = Renderer()

    @override(SessionListener)
    def before_session(self, event):
        import os
        out_path = event.activity_context.out_path
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            print('Created ' + out_path)

    @override(TrialListener)
    def before_trial(self, event):
        self.file_log_listener.before_trial(event)
        self.policy_maker.before_trial(event)

    @override(TrialListener)
    def after_trial(self, event):
        self.console_log_listener.after_trial(event)
        self.file_log_listener.after_trial(event)
        self.policy_maker.after_trial(event)

    @override(EpisodeListener)
    def before_episode(self, event):
        pass
    
    @override(EpisodeListener)
    def after_episode(self, event):
        self.console_log_listener.after_episode(event)
        self.file_log_listener.after_episode(event)
        self.policy_maker.after_episode(event)

    @override(StepListener)
    def after_step(self, event):
        if self.render:
            self.renderer.after_step(event)
