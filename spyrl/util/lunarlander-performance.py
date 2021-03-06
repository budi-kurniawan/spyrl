import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
from collections import namedtuple

BehaviourDataSource = namedtuple('BehaviourDataSource', 'label data_parent_path num_trials', defaults=[None, None])
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
        for trial in range(1, 11):
            file = ds.data_parent_path + '/scores-' + str(trial).zfill(2) + '.txt'
            if not os.path.exists(file):
                print(file + " does not exist.")
                break
            with open(file,'r') as csvfile:
                plots = csv.reader(csvfile, delimiter=',')
                solves = 0
                for row in plots:
                    value = float(row[1])
                    if value >= 200:
                        solves += 1
                        num_solves += 1
                    values.append(value)
                    sample = {'value': value, 'label':label}
                    data.append(sample)
                num_solves2.append(solves)
        print(label, ', mean:', np.mean(values), ', median:', np.median(values), ', solves:', num_solves, num_solves2)
    
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
            BehaviourDataSource(label='acet-02-1K', data_parent_path=parent + 'acet-02/performance-1000'),
            BehaviourDataSource(label='acet-02-2K', data_parent_path=parent + 'acet-02/performance-2000'),
            BehaviourDataSource(label='acet-03-4K', data_parent_path=parent + 'acet-03/performance-4000'),
            BehaviourDataSource(label='acet-01-5K', data_parent_path=parent + 'acet-01/performance-5000'),
            BehaviourDataSource(label='acet-01-10K', data_parent_path=parent + 'acet-01/performance-10000'),
            BehaviourDataSource(label='d2dspl-1K-01', data_parent_path=parent + 'd2dspl-1000-01/performance'),
            BehaviourDataSource(label='d2dspl-5K-01', data_parent_path=parent + 'd2dspl-5000-01/performance'),
            BehaviourDataSource(label='d2dspl-5K-02', data_parent_path=parent + 'd2dspl-5000-02/performance'),
            BehaviourDataSource(label='d2dspl-5K-03', data_parent_path=parent + 'd2dspl-5000-03/performance'),
            BehaviourDataSource(label='d2dspl-5K-04', data_parent_path=parent + 'd2dspl-5000-04/performance'),
            BehaviourDataSource(label='d2dspl-5K-05', data_parent_path=parent + 'd2dspl-5000-05/performance'),
            BehaviourDataSource(label='d2dspl-5K-06', data_parent_path=parent + 'd2dspl-5000-06/performance'),
            BehaviourDataSource(label='dqn-01-10K', data_parent_path=parent + 'dqn-01/performance-10000'),
            BehaviourDataSource(label='dqn-01-20K', data_parent_path=parent + 'dqn-01/performance-20000'),
            #BehaviourDataSource(label='d2dspl-1K-06', data_parent_path=parent + 'd2dspl-1000-06/performance'),
        ]
    draw(data_sources, result_path)