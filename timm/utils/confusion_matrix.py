#! python3
# -*- encoding: utf-8 -*-
"""
@file    :   confusion_matrix.py
@time    :   2023/03/01 16:20:57
@author  :   mnt
@version :   1.0
@contact :   ecnuzdm@gmail.com
@subject :   
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# target_names = ['Neu', 'Hap', 'Sad', 'Ang', 'Fear']

colors = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r']

def plot_confusion_matrix(confusion_matrix,target_names,fig_name,save_dir):
    conf_matrix = confusion_matrix
    categories = target_names
    root_dir = save_dir
    img_id = fig_name
    fig = plt.figure(figsize=(5.4,5.4))
    # fig = plt.figure(figsize=(6,6))
    # plt.subplots_adjust(right=0.996, left=0.070,bottom=0.160,top=0.990,)
    plt.subplots_adjust(right=1.050, left=0.050, bottom=0.150, top=0.990,)
    conf_matrix = np.array(conf_matrix)
    normalized_result = conf_matrix / conf_matrix.astype(np.float).sum(axis=1).reshape(-1, 1)
    
    # blanks = ['' for i in range(conf_matrix.size)]
    cf = conf_matrix
    group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    # group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    group_percentages = ["{0:.2%}".format(value) for value in normalized_result.flatten()]
    box_labels = [f"{v1}{v2}".strip() for v1, v2 in zip(group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])
    accuracy  = np.trace(cf) / float(np.sum(cf))
    stats_text = "\n Accuracy={:0.3f}".format(accuracy)

    ax = sns.heatmap(
                    normalized_result, 
                    #  cmap=sns.color_palette('Blues',n_colors=15), 
                    cmap = sns.color_palette("flare", as_cmap=True), # 1
                    # cmap = 'Blues', # default settings
                    # cmap = 'crest', # 2
                    # cmap=sns.cubehelix_palette(as_cmap=True),
                    # fmt='.2f', 
                    # fmt='.2%',
                    fmt="",
                    # annot=True,
                    annot=box_labels,
                    xticklabels=categories, 
                    yticklabels=categories, 
                    annot_kws={"fontsize":13}, 
                    # annot_kws={"fontsize":18}, 
                    linewidths = 0, 
                    square=True, 
                    cbar=False
                )
    
    # ax.set_xticklabels(target_names, size=15)
    # ax.set_yticklabels(target_names, size=15)

    # plt.xlabel('Predicted Label' + stats_text, fontsize=18)
    # plt.ylabel('True Label', fontsize=18, labelpad=-0.0)
    
    plt.xlabel('Predicted Label' + stats_text, fontsize=13)
    plt.ylabel('True Label', fontsize=13)
    
    # plt.xticks(fontsize=15)
    # plt.yticks(np.arange(len(categories))+0.5,(categories),rotation=90, fontsize=15, va="center")
    
    # plt.xticks(fontsize=10)
    # plt.yticks(np.arange(len(categories))+0.5,(categories), rotation=90, fontsize=12, va="center")
    # y_categories=['Benign\nLesions','(Mild)\nLeukoplakia','(Moderate)\nLeukoplakia','(Severe)\nLeukoplakia', 'Cancer']
    # plt.yticks(np.arange(len(y_categories))+0.5, y_categories, rotation=90, fontsize=11, va='center')
    
    # plt.tick_params(axis='x', pad=-2.2)
    plt.tick_params(axis='x', pad=1)
    plt.tick_params(axis='y', pad=1)

    # plt.tight_layout()
    plt.show()
    
    # fig.savefig("./paper_confusion/confusion_ours.pdf",dpi=1000)
    fig.savefig(root_dir+f"/{img_id}_conf_matrix.png",dpi=500)
    # 1、cla()：Clear axis即清除当前图形中的当前活动轴（轴就是子图）。其他轴不受影响。
    # 2、clf() ：Clear figure清除所有轴，但是窗口打开，这样它可以被重复使用。
    # 3、close()：Close a figure window    
    plt.close()



def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)


if __name__ == '__main__':
    plot_confusion_matrix(
        confusion_matrix=np.asarray([[47,10,4,5,6],[4,105,21,10,12],[1,25,103,10,13],[1,25,103,10,13],[1,25,103,10,13]]),
        target_names = ['Benign\n Lesions','Leukoplakia\n (Mild)','Leukoplakia\n (Moderate)','Leukoplakia\n (Severe)', 'Cancer'],
        fig_name='test_crest_finals',
        save_dir='./plots'
    )
