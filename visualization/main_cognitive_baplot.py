"""
the experiment for examining the effect of ROI selection under different subject's motion types and cognitive tasks.
"""

# Author: Shuo Li
# Date: 2024/09/15

import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from tqdm import tqdm
dir_crt = os.getcwd()
sys.path.append(os.path.join(dir_crt, 'util'))
import util_pre_analysis

def main_cognitive_baplot(roi):
    """main function to visualize the Bland-Altmann plot on the cognitive dataset.
    Parameters
    ----------
    roi: selected facial roi.  # ['lower medial forehead', 'left lower lateral forehead', 'right lower lateral forehead', 
                                  'glabella', 'upper nasal dorsum', 'lower nasal dorsum', 
                                  'soft triangle', 'left ala', 'right ala', 'nasal tip',
                                  'left lower nasal sidewall', 'right lower nasal sidewall', 'left mid nasal sidewall', 
                                  'right mid nasal sidewall', 'philtrum', 'left upper lip', 
                                  'right upper lip', 'left nasolabial fold', 'right nasolabial fold', 
                                  'left temporal', 'right temporal', 'left malar', 
                                  'right malar', 'left lower cheek', 'right lower cheek', 
                                  'chin', 'left marionette fold', 'right marionette fold']
    
    Returns
    -------

    """

    # get project directory.
    list_algorithm = ['CHROM', 'POS', 'LGI', 'OMIT']
    dir_option = os.path.join(dir_crt, 'config', 'options.yaml')
    len_time = 25
    # UBFC-Phys dataset (speech).
    list_condition = [1, 3]
    # parameter class initialization.
    Params = util_pre_analysis.Params(dir_option=dir_option, name_dataset='UBFC-Phys')
    # groundtruth class initialization.
    GT = util_pre_analysis.GroundTruth(dir_dataset=Params.dir_dataset, name_dataset='UBFC-Phys')
    len_window = int(len_time * Params.fps)
    # Loop over all attendants.
    for i_num_attendant in tqdm(range(1, 57)):
        # Loop over all conditions.
        for i_condition in range(0, len(list_condition)):
            condition = list_condition[i_condition]
            # Loop over all rPPG algorithms.
            for i_algorithm in range(0, len(list_algorithm)):
                algorithm = list_algorithm[i_algorithm]
                dir_es = os.path.join(dir_crt, 'data', 'UBFC-Phys', 'hr', str(i_num_attendant)+'_'+str(condition)+'_'+algorithm+'.csv')
                df_es = pd.read_csv(dir_es)
                if i_algorithm == 0:
                    esHR = df_es.loc[df_es['ROI'].values==roi, 'BPM'].values
                    # load groundtruth.
                    gtTime, gtTrace, gtHR = GT.get_GT(specification=[i_num_attendant, condition], 
                                                      num_frame_interp=int(len(esHR)), 
                                                      slice=[0, 1])
                else:
                    esHR = esHR + df_es.loc[df_es['ROI'].values==roi, 'BPM'].values
            esHR = esHR/len(list_algorithm)
            #t_sample = np.linspace(0, len(esHR)-1, 20).astype(np.int32)
            gtHR = gtHR[:int(np.floor(len(gtHR)/len_window)*len_window)]
            esHR = esHR[:int(np.floor(len(esHR)/len_window)*len_window)]
            gtHR = np.reshape(a=gtHR, newshape=(int(np.floor(len(gtHR)/len_window)), len_window))
            gtHR_median = np.median(gtHR, axis=1)
            esHR = np.reshape(a=esHR, newshape=(int(np.floor(len(esHR)/len_window)), len_window))
            esHR_median = np.median(esHR, axis=1)
            if (i_num_attendant == 1) and (i_condition == 0):
                esHR_all = esHR_median
                gtHR_all = gtHR_median
            else:
                esHR_all = np.concatenate((esHR_all, esHR_median), axis=0)
                gtHR_all = np.concatenate((gtHR_all, gtHR_median), axis=0)
    # UBFC-rPPG dataset. (realistic).
    list_attendant = [1] + list(range(3, 6)) + list(range(8, 19)) + [20] + list(range(22, 27)) + list(range(30, 50))
    # parameter class initialization.
    Params = util_pre_analysis.Params(dir_option=dir_option, name_dataset='UBFC-rPPG')
    # groundtruth class initialization.
    GT = util_pre_analysis.GroundTruth(dir_dataset=Params.dir_dataset, name_dataset='UBFC-rPPG')
    len_window = int(len_time * Params.fps)
    # Loop over all attendants.
    for num_attendant in tqdm(list_attendant):
        # Loop over all rPPG algorithms.
        for i_algorithm in range(0, len(list_algorithm)):
            algorithm = list_algorithm[i_algorithm]
            dir_es = os.path.join(dir_crt, 'data', 'UBFC-rPPG', 'hr', str(num_attendant)+'_'+algorithm+'.csv')
            df_es = pd.read_csv(dir_es)
            if i_algorithm == 0:
                esHR = df_es.loc[df_es['ROI'].values==roi, 'BPM'].values
            else:
                esHR = esHR + df_es.loc[df_es['ROI'].values==roi, 'BPM'].values
        esHR = esHR/len(list_algorithm)
        gtHR = gtHR[:int(np.floor(len(gtHR)/len_window)*len_window)]
        esHR = esHR[:int(np.floor(len(esHR)/len_window)*len_window)]
        gtHR = np.reshape(a=gtHR, newshape=(int(np.floor(len(gtHR)/len_window)), len_window))
        gtHR_median = np.median(gtHR, axis=1)
        esHR = np.reshape(a=esHR, newshape=(int(np.floor(len(esHR)/len_window)), len_window))
        esHR_median = np.median(esHR, axis=1)
        esHR_all = np.concatenate((esHR_all, esHR_median), axis=0)
        gtHR_all = np.concatenate((gtHR_all, gtHR_median), axis=0)
    ratio_3 = np.mean((np.abs(esHR_all-gtHR_all))<=3)
    ratio_5 = np.mean((np.abs(esHR_all-gtHR_all))<=5)
    ratio_10 = np.mean((np.abs(esHR_all-gtHR_all))<=10)
    print(ratio_3, ratio_5, ratio_10)
    res = sm.graphics.mean_diff_plot(esHR_all, gtHR_all, scatter_kwds={'color': '#1E466E'})
    plt.xlim(left=40, right=140)
    plt.ylim(bottom=-80, top=80)
    # visualization and save data.
    figure = plt.gcf()
    figure.set_size_inches(16, 9)
    plt.axhline(y=10, c='green', ls='-')
    plt.axhline(y=-10, c='green', ls='-')
    plt.fill_between(x=[np.min([esHR_all.min(), gtHR_all.min()])-3, np.max([esHR_all.max(), gtHR_all.max()])+3], y1=-10, y2=10, 
                     color='green', alpha=0.5)
    print(roi)
    plt.show()
    pass



if __name__ == "__main__":
    list_roi = ['glabella', 'lower medial forehead', 'left malar', 'right malar', 'left lower lateral forehead', 'right lower lateral forehead', 'upper nasal dorsum']  # top-seven ROIs.
    for roi in list_roi:
        main_cognitive_baplot(roi=roi)