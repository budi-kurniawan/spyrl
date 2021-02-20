import os
from spyrl.util.util import get_project_dir
class ActivityConfig(object):
    def __init__(self, **kwargs):
        self.start_trial = kwargs.get('start_trial', 1)
        self.num_trials = kwargs.get('num_trials', 1)
        self.num_episodes = kwargs.get('num_episodes', 1)
        self.out_path = os.path.join(get_project_dir(), kwargs.get('out_path', '.'))