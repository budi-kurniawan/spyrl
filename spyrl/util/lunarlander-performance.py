import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
from collections import namedtuple

DataSource = namedtuple('DataSource', 'label data_parent_path num_trials', defaults=[None, None, 10])
context = {'palette': 'Blues', 'baseline_color': 'red', 'figsize': (15, 5)}
""" info about the palette: http://seaborn.pydata.org/tutorial/color_palettes.html """

def draw(data_sources, result_path=None):
    plt.rcParams["figure.figsize"] = context['figsize']
    data = []
    baselines = []
    legend_labels = []
    offset = 0
    for ds in data_sources:
        label = ds.label
        legend_labels.append(label)
        values = []
        num_solves = 0
        num_solves2 = []
        print(label, '==============================')
        for trial in range(1, 1 + ds.num_trials):
            file = ds.data_parent_path + '/scores-' + str(trial).zfill(2) + '.txt'
            if not os.path.exists(file):
                print(file + " does not exist.")
                break
            trial_values = []
            solves = 0
            with open(file,'r') as csvfile:
                plots = csv.reader(csvfile, delimiter=',')
                for row in plots:
                    value = float(row[1])
                    if value >= 200:
                        solves += 1
                        num_solves += 1
                        value = 200
                    if value < 0:
                        value = 0
                    trial_values.append(value)
                    values.append(value)
                    sample = {'value': value, 'label':label}
                    data.append(sample)
            num_solves2.append(solves)
            print(int(np.mean(trial_values)))
            #print('trial:', trial, ', mean:', np.mean(trial_values), ', median:', np.median(trial_values))
        print(label, np.mean(values), np.median(values), num_solves, num_solves2)

    sns.set(style="whitegrid")
    dataFrame = pd.DataFrame(data)
    bplot = sns.boxplot(x="label", y="value", hue="label", data=dataFrame, linewidth=2.5)
    #bplot = sns.boxplot(x="label", y="value", data=dataFrame, linewidth=2.5)
    handles, _ = bplot.get_legend_handles_labels()
    bplot.legend(handles, legend_labels)#    facecolors = ('orange', 'lightblue', 'lightgreen', 'green', 'lightyellow', 'lightcyan', 'yellow', 'lightpink', 'red')
    plt.xlabel('Agent')
    plt.ylabel('Average Score')
    if 'ylim' in context:
        plt.ylim(context['ylim'])
    plt.tight_layout(pad=0.05) # will change ax dimension, make them bigger since margins are reduced        
#     if result_path is not None:
#         plt.savefig(result_path)
    plt.show()


if __name__ == '__main__':
    parent = '../../result/lunarlander/'
    #https://www.w3schools.com/colors/colors_gradient.asp
    colors = ['LightCyan', 'LightSkyBlue', 'MediumBlue', #'Blue',
              #'coral', 'darksalmon', 
              '#f8f8f8', '#e8e8e8', '#d7d7d7', '#909090', '#808080', '#707070',
              'LightGreen', 'MediumSpringGreen', 'GreenYellow', 'Green', 'LightGreen', 'MediumSpringGreen', 'GreenYellow', 'Green'] 
    context['palette'] = colors
    context['figsize'] = (5, 5)

    result_path = './ac-dqn-so/ac-dqn-morl-performance-boxplot-1.pdf'
    data_sources = [
#             DataSource(label='acet-02-1K', data_parent_path=parent + 'acet-02/performance-1000'),
#             DataSource(label='acet-02-2K', data_parent_path=parent + 'acet-02/performance-2000'),
#             DataSource(label='acet-03-4K', data_parent_path=parent + 'acet-03/performance-4000'),
#             DataSource(label='acet-01-5K', data_parent_path=parent + 'acet-01/performance-5000'),
#             DataSource(label='acet-01-10K', data_parent_path=parent + 'acet-01/performance-10000'),
#             DataSource(label='acet-21-1K', data_parent_path=parent + 'acet-21/performance-1000'),
#             DataSource(label='acet-21-2K', data_parent_path=parent + 'acet-21/performance-2000'),
#             DataSource(label='acet-21-4K', data_parent_path=parent + 'acet-21/performance-4000'),
#             DataSource(label='acet-21-5K', data_parent_path=parent + 'acet-21/performance-5000'),
#            DataSource(label='acet-11-8K', data_parent_path=parent + 'acet-11/performance-8000'),
#             DataSource(label='acet-21-10K', data_parent_path=parent + 'acet-21/performance-10000'),
#             DataSource(label='d2dspl-5K-01', data_parent_path=parent + 'd2dspl-5000-01/performance'),
#             DataSource(label='d2dspl-5K-02', data_parent_path=parent + 'd2dspl-5000-02/performance'),
#             DataSource(label='d2dspl-5K-03', data_parent_path=parent + 'd2dspl-5000-03/performance'),
#             DataSource(label='d2dspl-5K-04', data_parent_path=parent + 'd2dspl-5000-04/performance'),
#             DataSource(label='d2dspl-5K-05', data_parent_path=parent + 'd2dspl-5000-05/performance'),
#             DataSource(label='d2dspl-5K-06', data_parent_path=parent + 'd2dspl-5000-06/performance'),
#             DataSource(label='d2dspl-5K-07', data_parent_path=parent + 'd2dspl-5000-07/performance', num_trials=3),
#             DataSource(label='d2dspl-5K-08', data_parent_path=parent + 'd2dspl-5000-08/performance', num_trials=3),
#             DataSource(label='d2dspl-5K-21', data_parent_path=parent + 'd2dspl-5000-21/performance'),
            DataSource(label='ACET-5K-22', data_parent_path=parent + 'd2dspl-acet-10000-22/performance-acet-5000'),
            DataSource(label='ACET-10K-22', data_parent_path=parent + 'd2dspl-acet-10000-22/performance-acet-10000'),
            DataSource(label='d2dspl-5K-22', data_parent_path=parent + 'd2dspl-acet-10000-22/performance-d2dspl'),
# just ok            DataSource(label='d2dspl-5K-23', data_parent_path=parent + 'd2dspl-acet-10000-23/performance-d2dspl'),
#             DataSource(label='ddqn-03-2K', data_parent_path=parent + 'ddqn-03/performance-2000'),
            DataSource(label='dqn-03-5K', data_parent_path=parent + 'dqn-03/performance-5000'),
            DataSource(label='dqn-03-10K', data_parent_path=parent + 'dqn-03/performance-5000'),
            DataSource(label='ddqn-03-5K', data_parent_path=parent + 'ddqn-03/performance-5000'),
#             DataSource(label='ddqn-03-6K', data_parent_path=parent + 'ddqn-03/performance-6000'),
#             DataSource(label='ddqn-03-7K', data_parent_path=parent + 'ddqn-03/performance-7000'),
#             DataSource(label='ddqn-03-8K', data_parent_path=parent + 'ddqn-03/performance-8000'),
#             DataSource(label='ddqn-03-9K', data_parent_path=parent + 'ddqn-03/performance-9000'),
            DataSource(label='ddqn-03-10K', data_parent_path=parent + 'ddqn-03/performance-10000'),
# bad result            DataSource(label='d2dspl-5K-24', data_parent_path=parent + 'd2dspl-acet-10000-24/performance-d2dspl'),
#             DataSource(label='dqn-old-1K', data_parent_path=parent + 'dqn-old/performance-1000', num_trials=1),
#             DataSource(label='dqn-old-2K', data_parent_path=parent + 'dqn-old/performance-2000', num_trials=1),
            #DataSource(label='d2dspl-1K-06', data_parent_path=parent + 'd2dspl-1000-06/performance'),
        ]
    draw(data_sources, result_path)