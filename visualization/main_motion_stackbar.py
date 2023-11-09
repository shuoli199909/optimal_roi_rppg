"""
the experiment for examining the effect of ROI selection under different subject's motion types.
"""

# Author: Shuo Li
# Date: 2023/09/11

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main(list_motion, list_algorithm):
    """main function to evaluate the effect of ROI selection under different subject's motion types.
    Parameters
    ----------
    list_motion: list of subject's motion types.   # ['resting', 'gym', 'rotation', 'talk'].
    list_algorithm: list of selecte rPPG algorithms.   # ['CHROM', 'POS', 'LGI', 'OMIT'].
    
    Returns
    -------

    """
    # get project directory.
    dir_crt = os.getcwd()
    # list of ROI names.
    list_roi_name = ['lower medial forehead', 'glabella', 'left lower lateral forehead', 
                     'right lower lateral forehead', 'left temporal', 'right temporal',
                     'upper nasal dorsum', 'lower nasal dorsum', 'left mid nasal sidewall', 
                     'right mid nasal sidewall', 'left lower nasal sidewall', 'right lower nasal sidewall', 
                     'nasal tip', 'soft triangle', 'left ala', 
                     'right ala', 'left nasolabial fold', 'right nasolabial fold',
                     'left upper lip', 'right upper lip', 'left malar', 
                     'right malar', 'philtrum', 'left lower cheek', 
                     'right lower cheek', 'left marionette fold', 'right marionette fold', 
                     'chin']
    # loop over all motion types.
    list_stack = []
    for i_motion in range(len(list_motion)):
        motion = list_motion[i_motion]
        # loop over all selected algorithms and then compute the average.
        for i_algorithm in range(len(list_algorithm)):
            algorithm = list_algorithm[i_algorithm]
            # metric dataframe.
            # LGI-PPGI dataset.
            dir_LGI_PPGI = os.path.join(dir_crt, 'result', 'LGI-PPGI', 'evaluation_'+algorithm+'.csv')
            df_LGI_PPGI = pd.read_csv(dir_LGI_PPGI, index_col=0)
            df_LGI_PPGI = df_LGI_PPGI.loc[df_LGI_PPGI['motion'].values==motion, :]
            df_LGI_PPGI = df_LGI_PPGI.reset_index()
            # collect performance evaluation results.
            if i_algorithm == 0:
                data_roi_mae = []
                data_roi_pcc = []
            for i in range(len(list_roi_name)):
                # LGI-PPGI dataset.
                # MAE metric.
                metric_mae = df_LGI_PPGI.loc[df_LGI_PPGI['ROI'].values==list_roi_name[i], 'MAE'].dropna().values
                if i_algorithm == 0:
                    data_roi_mae.append(np.array(metric_mae))
                else:
                    data_roi_mae[i] = data_roi_mae[i] + metric_mae
                # PCC metric.
                metric_pcc = df_LGI_PPGI.loc[df_LGI_PPGI['ROI'].values==list_roi_name[i], 'PCC'].dropna().values
                if i_algorithm == 0:
                    data_roi_pcc.append(np.array(metric_pcc))
                else:
                    data_roi_pcc[i] = data_roi_pcc[i] + metric_pcc
        # average over all included algorithms.
        data_roi_mae = np.array(data_roi_mae, dtype=object)/len(list_algorithm)  # MAE metric.
        data_roi_mae = np.median(data_roi_mae, axis=1)
        data_roi_pcc = np.array(data_roi_pcc, dtype=object)/len(list_algorithm)  # PCC metric.
        data_roi_pcc = np.median(data_roi_pcc, axis=1)
        # overall score.
        data_roi_os = 0.5*(np.max(data_roi_mae) - data_roi_mae)/(np.max(data_roi_mae) - np.min(data_roi_mae)) + \
                      0.5*(data_roi_pcc - np.min(data_roi_pcc))/(np.max(data_roi_pcc) - np.min(data_roi_pcc))
        list_stack.append(data_roi_os)
    # data visualization.
    plt.cla()
    ind = np.arange(0, 28, 1)
    list_stack_sum = np.array(list_stack[0] + list_stack[1] + list_stack[2] + list_stack[3])
    list_sort = np.argsort(list_stack_sum)
    ind = ind[list_sort][::-1]
    # stacked bar chart.
    list_stack[0] = list_stack[0][list_sort]
    list_stack[1] = list_stack[1][list_sort]
    list_stack[2] = list_stack[2][list_sort]
    list_stack[3] = list_stack[3][list_sort]
    width = 0.72
    p0 = plt.barh(np.arange(1, 29, 1), list_stack[0], width, edgecolor='black', color=np.array([2, 48, 74])/255)
    p1 = plt.barh(np.arange(1, 29, 1), list_stack[1], width, left=list_stack[0], edgecolor='black', color=np.array([33, 158, 188])/255)
    p2 = plt.barh(np.arange(1, 29, 1), list_stack[2], width, left=list_stack[0]+list_stack[1], edgecolor='black', color=np.array([254, 183, 5])/255)
    p3 = plt.barh(np.arange(1, 29, 1), list_stack[3], width, left=list_stack[0]+list_stack[1]+list_stack[2], edgecolor='black', color=np.array([250, 134, 0])/255)
    plt.grid(visible=None, which='major', axis='x')
    plt.tight_layout()
    plt.ylim([0.5, 28.5])
    plt.yticks(ticks=np.linspace(0, 29, 30))
    dir_save = os.path.join(dir_crt, 'plot', 'motion', 'motion.png')
    print('ROI list ranked in the descending order:')
    print(np.array(list_roi_name)[ind])
    print(ind+1)
    figure = plt.gcf()
    figure.set_size_inches(13, 9)
    #plt.savefig(dir_save, dpi=600, bbox_inches='tight')
    print('Overall scores of each video activity:')
    print(list_motion[0], list_stack[0][::-1])
    print(list_motion[1], list_stack[1][::-1])
    print(list_motion[2], list_stack[2][::-1])
    print(list_motion[3], list_stack[3][::-1])

if __name__ == "__main__":
    list_algorithm = ['CHROM', 'POS', 'LGI', 'OMIT']   # ['LGI', 'OMIT', 'CHROM', 'POS'].
    list_motion = ['resting', 'rotation', 'gym', 'talk']   # ['resting', 'gym', 'rotation', 'talk'].
    # loop over all performance evaluation metrics.
    main(list_motion=list_motion, list_algorithm=list_algorithm)