"""
the experiment for examining the effect of ROI selection under different subject's motion types or cognitive conditions.
"""

# Author: Shuo Li
# Date: 2024/09/03

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main_cognitive_stackbar(list_algorithm):
    """main function to visualize the stacked barchart of the cognitive dataset.
    Parameters
    ----------
    list_algorithm: list of select rPPG algorithms.   # ['CHROM', 'POS', 'LGI', 'OMIT'].
    
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
    # loop over all selected algorithms and then compute the average.
    for i_algorithm in range(len(list_algorithm)):
        algorithm = list_algorithm[i_algorithm]
        # metric dataframe.
        # UBFC-rPPG dataset (realistic).
        dir_UBFC_rPPG_realistic = os.path.join(dir_crt, 'result', 'UBFC-rPPG', 'evaluation_'+algorithm+'.csv')
        df_UBFC_rPPG_realistic = pd.read_csv(dir_UBFC_rPPG_realistic, index_col=0)
        df_UBFC_rPPG_realistic = df_UBFC_rPPG_realistic.reset_index()
        # UBFC-Phys dataset (arithmetic).
        dir_UBFC_Phys = os.path.join(dir_crt, 'result', 'UBFC-Phys', 'evaluation_'+algorithm+'.csv')
        df_UBFC_Phys = pd.read_csv(dir_UBFC_Phys, index_col=0)
        df_UBFC_Phys_arithmetic = df_UBFC_Phys.loc[df_UBFC_Phys['condition'].values==3, :].copy()
        df_UBFC_Phys_arithmetic = df_UBFC_Phys_arithmetic.reset_index()
        # collect performance evaluation results.
        if i_algorithm == 0:
            # MAE metric.
            data_roi_mae_UBFC_rPPG_realistic = []
            data_roi_mae_UBFC_Phys_arithmetic = []
            # PCC metric.
            data_roi_pcc_UBFC_rPPG_realistic = []
            data_roi_pcc_UBFC_Phys_arithmetic = []
            # SNR metric.
            data_roi_snr_UBFC_rPPG_realistic = []
            data_roi_snr_UBFC_Phys_arithmetic = []
        for i in range(len(list_roi_name)):
            # MAE metric.
            mae_UBFC_rPPG_realistic = df_UBFC_rPPG_realistic.loc[df_UBFC_rPPG_realistic['ROI'].values==list_roi_name[i], 'MAE'].dropna().values
            mae_UBFC_Phys_arithmetic = df_UBFC_Phys_arithmetic.loc[df_UBFC_Phys_arithmetic['ROI'].values==list_roi_name[i], 'MAE'].dropna().values
            if i_algorithm == 0:
                data_roi_mae_UBFC_rPPG_realistic.append(np.median(mae_UBFC_rPPG_realistic))
                data_roi_mae_UBFC_Phys_arithmetic.append(np.median(mae_UBFC_Phys_arithmetic))
            else:
                data_roi_mae_UBFC_rPPG_realistic[i] = data_roi_mae_UBFC_rPPG_realistic[i] + np.median(mae_UBFC_rPPG_realistic)
                data_roi_mae_UBFC_Phys_arithmetic[i] = data_roi_mae_UBFC_Phys_arithmetic[i] + np.median(mae_UBFC_Phys_arithmetic)
            # PCC metric.
            pcc_UBFC_rPPG_realistic = df_UBFC_rPPG_realistic.loc[df_UBFC_rPPG_realistic['ROI'].values==list_roi_name[i], 'PCC'].dropna().values
            pcc_UBFC_Phys_arithmetic = df_UBFC_Phys_arithmetic.loc[df_UBFC_Phys_arithmetic['ROI'].values==list_roi_name[i], 'PCC'].dropna().values
            if i_algorithm == 0:
                data_roi_pcc_UBFC_rPPG_realistic.append(np.median(pcc_UBFC_rPPG_realistic))
                data_roi_pcc_UBFC_Phys_arithmetic.append(np.median(pcc_UBFC_Phys_arithmetic))
            else:
                data_roi_pcc_UBFC_rPPG_realistic[i] = data_roi_pcc_UBFC_rPPG_realistic[i] + np.median(pcc_UBFC_rPPG_realistic)
                data_roi_pcc_UBFC_Phys_arithmetic[i] = data_roi_pcc_UBFC_Phys_arithmetic[i] + np.median(pcc_UBFC_Phys_arithmetic)
            # SNR metric.
            snr_UBFC_rPPG_realistic = df_UBFC_rPPG_realistic.loc[df_UBFC_rPPG_realistic['ROI'].values==list_roi_name[i], 'SNR'].dropna().values
            snr_UBFC_rPPG_realistic[np.isinf(snr_UBFC_rPPG_realistic)] = np.median(snr_UBFC_rPPG_realistic)
            snr_UBFC_Phys_arithmetic = df_UBFC_Phys_arithmetic.loc[df_UBFC_Phys_arithmetic['ROI'].values==list_roi_name[i], 'SNR'].dropna().values
            snr_UBFC_Phys_arithmetic[np.isinf(snr_UBFC_Phys_arithmetic)] = np.median(snr_UBFC_Phys_arithmetic)
            if i_algorithm == 0:
                data_roi_snr_UBFC_rPPG_realistic.append(np.median(snr_UBFC_rPPG_realistic))
                data_roi_snr_UBFC_Phys_arithmetic.append(np.median(snr_UBFC_Phys_arithmetic))
            else:
                data_roi_snr_UBFC_rPPG_realistic[i] = data_roi_snr_UBFC_rPPG_realistic[i] + np.median(snr_UBFC_rPPG_realistic)
                data_roi_snr_UBFC_Phys_arithmetic[i] = data_roi_snr_UBFC_Phys_arithmetic[i] + np.median(snr_UBFC_Phys_arithmetic)
    # average over all included algorithms.
    # MAE metric.
    data_roi_mae_UBFC_rPPG_realistic = np.array(data_roi_mae_UBFC_rPPG_realistic, dtype=np.float32)/len(list_algorithm)
    data_roi_mae_UBFC_Phys_arithmetic = np.array(data_roi_mae_UBFC_Phys_arithmetic, dtype=np.float32)/len(list_algorithm)
    # PCC metric.
    data_roi_pcc_UBFC_rPPG_realistic = np.array(data_roi_pcc_UBFC_rPPG_realistic, dtype=np.float32)/len(list_algorithm)
    data_roi_pcc_UBFC_Phys_arithmetic = np.array(data_roi_pcc_UBFC_Phys_arithmetic, dtype=np.float32)/len(list_algorithm)
    # SNR metric.
    data_roi_snr_UBFC_rPPG_realistic = np.array(data_roi_snr_UBFC_rPPG_realistic, dtype=np.float32)/len(list_algorithm)
    data_roi_snr_UBFC_Phys_arithmetic = np.array(data_roi_snr_UBFC_Phys_arithmetic, dtype=np.float32)/len(list_algorithm)
    # overall score.
    data_roi_os_UBFC_rPPG_realistic = (1/3)*(np.max(data_roi_mae_UBFC_rPPG_realistic) - data_roi_mae_UBFC_rPPG_realistic)/ \
                                      (np.max(data_roi_mae_UBFC_rPPG_realistic) - np.min(data_roi_mae_UBFC_rPPG_realistic)) + \
                                      (1/3)*(data_roi_pcc_UBFC_rPPG_realistic - np.min(data_roi_pcc_UBFC_rPPG_realistic))/ \
                                      (np.max(data_roi_mae_UBFC_rPPG_realistic) - np.min(data_roi_mae_UBFC_rPPG_realistic)) + \
                                      (1/3)*(data_roi_snr_UBFC_rPPG_realistic - np.min(data_roi_snr_UBFC_rPPG_realistic))/ \
                                      (np.max(data_roi_snr_UBFC_rPPG_realistic) - np.min(data_roi_snr_UBFC_rPPG_realistic))
    data_roi_os_UBFC_Phys_arithmetic = (1/3)*(np.max(data_roi_mae_UBFC_Phys_arithmetic) - data_roi_mae_UBFC_Phys_arithmetic)/ \
                                       (np.max(data_roi_mae_UBFC_Phys_arithmetic) - np.min(data_roi_mae_UBFC_Phys_arithmetic)) + \
                                       (1/3)*(data_roi_pcc_UBFC_Phys_arithmetic - np.min(data_roi_pcc_UBFC_Phys_arithmetic))/ \
                                       (np.max(data_roi_pcc_UBFC_Phys_arithmetic) - np.min(data_roi_pcc_UBFC_Phys_arithmetic)) + \
                                       (1/3)*(data_roi_snr_UBFC_Phys_arithmetic - np.min(data_roi_snr_UBFC_Phys_arithmetic))/ \
                                       (np.max(data_roi_snr_UBFC_Phys_arithmetic) - np.min(data_roi_snr_UBFC_Phys_arithmetic))
    list_stack = [data_roi_os_UBFC_rPPG_realistic, data_roi_os_UBFC_Phys_arithmetic]
    # data visualization.
    plt.cla()
    ind = np.arange(0, 28, 1)
    list_stack_sum = np.array(list_stack[0] + list_stack[1])
    list_sort = np.argsort(list_stack_sum)
    ind = ind[list_sort][::-1]
    # stacked bar chart.
    list_stack[0] = list_stack[0][list_sort]
    list_stack[1] = list_stack[1][list_sort]
    width = 0.72
    p0 = plt.barh(np.arange(1, 29, 1), list_stack[0], width, edgecolor='black', color=np.array([170, 220, 224])/255)
    p1 = plt.barh(np.arange(1, 29, 1), list_stack[1], width, left=list_stack[0], edgecolor='black', color=np.array([30, 70, 110])/255)
    plt.grid(visible=None, which='major', axis='x')
    plt.tight_layout()
    plt.ylim([0.5, 28.5])
    plt.yticks(ticks=np.linspace(0, 29, 30))    
    print('ROI list ranked in the descending order:')
    print(np.array(list_roi_name)[ind])
    print(ind+1)
    figure = plt.gcf()
    figure.set_size_inches(13, 9)
    plt.show()
    pass



if __name__ == "__main__":
    list_algorithm = ['CHROM', 'POS', 'LGI', 'OMIT']   # ['LGI', 'OMIT', 'CHROM', 'POS'].
    # loop over all performance evaluation metrics.
    main_cognitive_stackbar(list_algorithm=list_algorithm)