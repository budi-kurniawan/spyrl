import os
import matplotlib.pyplot as plt
import csv
from spyrl.listener.trial_listener import TrialListener
from spyrl.listener.episode_listener import EpisodeListener
from spyrl.util.util import override
from spyrl.listener.impl.file_log_listener import RewardType

class TestResultLogger(TrialListener, EpisodeListener):
    def __init__(self, reward_type=RewardType.AVERAGE):
        self.reward_type = reward_type
        self.scores_file = None
        self.writer = None
        self.chart_offset = 0

    @override(EpisodeListener)
    def after_episode(self, event):
        if self.scores_file is not None:
            activity_context = event.activity_context
            episode = activity_context.episode
            step = activity_context.step
            if self.reward_type == RewardType.AVERAGE:
                self.scores_file.write(str(episode) + "," + str(event.avg_reward) + '\n')
            elif self.reward_type == RewardType.TOTAL:
                self.scores_file.write(str(episode) + "," + str(event.reward) + '\n')
                
    @override(TrialListener)
    def before_trial(self, event):
        out_path = event.activity_context.out_path
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            print('Created ' + out_path)
        trial = event.activity_context.trial
        self.scores_file = open(out_path + '/scores-' + str(trial).zfill(2) + '.txt', 'w')

    @override(TrialListener)
    def after_trial(self, event):
        self.scores_file.close()
        ac = event.activity_context
        out_path = ac.out_path
        trial = ac.trial
        duration_in_seconds = (ac.trial_end_time - ac.trial_start_time).total_seconds()
        times_file = open(os.path.join(out_path, 'times.txt'), 'a+')
        msg = 'Trial ' + str(trial) + ' finished in ' + str(duration_in_seconds) + ' seconds.'
        times_file.write(msg + '\n')
        times_file.close()
        self.draw_chart(trial, out_path)

    def draw_chart(self, trial, out_path):
        plt.rcParams["figure.figsize"] = (20, 4)
        plt.rcParams["legend.loc"] = 'lower center'
        scores_path = os.path.join(out_path, 'scores-' + str(trial).zfill(2) + '.txt')        
        x = []
        y = []
        offset = self.chart_offset
        with open(scores_path, 'r') as csvfile:
            plots = csv.reader(csvfile, delimiter=',')
            for row in plots:
                x.append(int(row[0]))
                y.append(float(row[1]) + offset)
        plt.plot(x, y, 'bo')
        plt.axis([1, len(x), 0, 1])
        plt.xlabel('Test')
        plt.ylabel('Total score')
        plt.savefig(scores_path + '.png')
        plt.close()