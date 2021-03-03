from spyrl.listener.episode_listener import EpisodeListener
from spyrl.listener.session_listener import SessionListener
from spyrl.listener.step_listener import StepListener
from spyrl.listener.trial_listener import TrialListener
from spyrl.util.util import override

class OutPathCreator(SessionListener, TrialListener, EpisodeListener, StepListener):
    @override(SessionListener)
    def before_session(self, event):
        import os
        out_path = event.activity_context.out_path
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            print(out_path + ' created.')