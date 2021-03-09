import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import csv
import os
import numpy as np
from collections import namedtuple
from matplotlib.ticker import FuncFormatter

DataSource = namedtuple('DataSource', 'name data_paths result_path labels image_path image_xy title', defaults=[None, None, None])
context = {'start_trial': 1, 'num_trials': 10, 'max_records': 100_000, 'num_avg_samples': 50, 'offset': 0,
           'figsize': (14, 3), 'show_figures': False, 'ylim': (0, .8)}

def get_data(start_trial, num_trials, path, max_records, num_avg_samples, offset):
    X = []
    Y = []
    y_max = float('-inf')
    for i in range(start_trial, start_trial + num_trials):
        file = os.path.join(path, 'scores-' + str(i).zfill(2) + '.txt')
        print(file)
        # allow this function to use a smaller number of data sources
        if not os.path.exists(file):
            print(file + ' does not exist. Use existing data only.')
            break 
        x = []
        y = []
        with open(file,'r') as csvfile:
            plots = csv.reader(csvfile, delimiter=',')
            for row in plots:
                x.append(int(row[0]))
                y_value = float(row[1])
                if y_max < y_value:
                    y_max = y_value
                y.append(y_value + offset)
                if len(y) == max_records:
                    break
    
        if num_avg_samples != 1:
            for j in range(0, int(len(x)/num_avg_samples)):
                x[j] = x[(j + 1) * num_avg_samples - 1]
                index = j * num_avg_samples
                y[j] = np.mean(y[index : index + num_avg_samples])
            x = x[0 : int(len(x)/num_avg_samples)]
            y = y[0 : len(x)]
        X.append(x)
        Y.append(y)
    return X, Y, y_max

def draw(axs, data_paths, labels):
    colors = ['blue', 'orange', 'green']
    edgecolors = ['lightblue', 'coral', 'green']
    facecolors = ['lightblue', 'coral', 'green']
    
    for i in range(len(data_paths)):
        X, Y, y_max = get_data(context['start_trial'], context['num_trials'], data_paths[i], context['max_records'], context['num_avg_samples'], context['offset'])
        all_runs = np.stack(Y)
        means = np.mean(all_runs, axis=0)
        stddev = np.std(all_runs, axis=0)
        label = labels[i] + ' (avg: {0:.4f}'.format(np.mean(means)) + ', max: {0:.4f}'.format(y_max) + ')' 
        axs.plot(X[0], means, color=colors[i], label=label) # no std
        axs.fill_between(X[0], means-stddev, means+stddev, alpha=0.2, edgecolor=edgecolors[i], facecolor=facecolors[i],
                         linewidth=2, linestyle='dashdot', antialiased=True)
        #plt.errorbar(X[0], means, stddev, linestyle='None', marker='.', label=labels[i])

def add_image(ax, data_source):
    if data_source.image_path is None:
        return
    artist_array = mpimg.imread(data_source.image_path)
    imagebox = OffsetImage(artist_array, zoom=0.5)
    ab = AnnotationBbox(imagebox, xy=data_source.image_xy, xybox=(0, 0), xycoords='data', boxcoords=("offset points"), box_alignment=(0, 1), pad=0.5)
    ax.add_artist(ab)

def get_axis_dim(fig, ax):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    arrow_width, height = bbox.width, bbox.height
    arrow_width *= fig.dpi
    height *= fig.dpi
    return arrow_width, height
    
def create_charts(data_sources):
    plt.rcParams["figure.figsize"] = context['figsize']
    plt.rcParams["legend.loc"] = 'upper left'
    for data_source in data_sources:
        fig, axs = plt.subplots(1)
        axs.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
        axs.set_ylim(context['ylim'])
        plt.margins(0, x=None, y=None, tight=True)
        plt.xlabel('Episode')
        plt.ylabel('Average Score')
        if data_source.title is not None:
            axs.text(.5,.9, data_source.title, horizontalalignment='center', transform=axs.transAxes, 
                    fontsize=13)
            
        draw(axs, data_source.data_paths, data_source.labels)
        add_image(axs, data_source)
        axs.legend()
        plt.tight_layout(pad=0.05) # will change ax dimension, make them bigger since margins are reduced        
        print(' Saved ' + data_source.result_path)
        #plt.savefig(data_source.result_path)
        if context['show_figures']:
            plt.show()
            
if __name__ == '__main__':
    parent = '../../result/lunarlander/acet-01/'
    context['max_records'] = 10000
    context['num_trials'] = 10
    context['start_trial'] = 1
    context['num_avg_samples'] = 20
    context['show_figures'] = True
    context['ylim'] = (-200, 300)
    
    data_sources = [
            DataSource(name='acet', data_paths=[parent], labels=['lunarlander-acet-01', 'ac-002'],
                       image_path=None, image_xy=None, result_path='./temp/temp-01.png')
    ]
    #create_charts(data_sources)
    
    parent = '../../result/lunarlander/dqn-old/'
    parent2 = '../../result/lunarlander/dqn-old-02/'
    parent3 = '../../result/lunarlander/dqn-hacked/'
    parent4 = '../../result/lunarlander/dqn-03/'
    context['max_records'] = 10000
    context['start_trial'] = 1
    context['num_trials'] = 10
    context['num_avg_samples'] = 10
    context['show_figures'] = True
    context['ylim'] = (-800, 200)
    
    data_sources = [
            DataSource(name='dqn', data_paths=[parent4], 
                       labels=['dqn-03'],
                       image_path=None, image_xy=None, result_path='./temp/temp-01.png')
    ]
    create_charts(data_sources)
    
    
    parent = '../../../joadia/results/'
    context['max_records'] = 100000
    context['start_trial'] = 2
    context['num_trials'] = 1
    context['num_avg_samples'] = 20
    context['show_figures'] = True
    context['ylim'] = (-60, -30)
    
    data_sources = [
            DataSource(name='Q-learning', data_paths=[parent], labels=['Joadia Q-learning'],
                       image_path=None, image_xy=None, result_path='./temp/temp-01.png')
    ]
    create_charts(data_sources)

    parent = '../../result/lunarlander/d2dspl-01/'
    context['max_records'] = 5000
    context['start_trial'] = 1
    context['num_trials'] = 10
    context['num_avg_samples'] = 10
    context['show_figures'] = True
    context['ylim'] = (-300, 300)
    
    data_sources = [
            DataSource(name='Q-learning', data_paths=[parent], labels=['D2D-SPL'],
                       image_path=None, image_xy=None, result_path='./temp/temp-01.png')
    ]
    #create_charts(data_sources)
