"""
the experiment for examining the effect of ROI selection under different subject's motion types or cognitive conditions.
"""

# Author: Shuo Li
# Date: 2023/09/11

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main_motion_stackbar(list_algorithm):
    """main function to visualize the stacked barchart of the motion dataset.
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
        # UBFC-Phys dataset (speech).
        dir_UBFC_Phys = os.path.join(dir_crt, 'result', 'UBFC-Phys', 'evaluation_'+algorithm+'.csv')
        df_UBFC_Phys = pd.read_csv(dir_UBFC_Phys, index_col=0)
        df_UBFC_Phys_speech = df_UBFC_Phys.loc[df_UBFC_Phys['condition'].values==2, :].copy()
        df_UBFC_Phys_speech = df_UBFC_Phys_speech.reset_index()
        # LGI-PPGI.
        dir_LGI_PPGI = os.path.join(dir_crt, 'result', 'LGI-PPGI', 'evaluation_'+algorithm+'.csv')
        df_LGI_PPGI = pd.read_csv(dir_LGI_PPGI, index_col=0)
        # rotation.
        df_LGI_PPGI_rotation = df_LGI_PPGI.loc[df_LGI_PPGI['motion'].values=='rotation', :].copy()
        df_LGI_PPGI_rotation = df_LGI_PPGI_rotation.reset_index()
        # gym.
        df_LGI_PPGI_gym = df_LGI_PPGI.loc[df_LGI_PPGI['motion'].values=='gym', :].copy()
        df_LGI_PPGI_gym = df_LGI_PPGI_gym.reset_index()
        # talk.
        df_LGI_PPGI_talk = df_LGI_PPGI.loc[df_LGI_PPGI['motion'].values=='talk', :].copy()
        df_LGI_PPGI_talk = df_LGI_PPGI_talk.reset_index()
        # collect performance evaluation results.
        if i_algorithm == 0:
            # MAE metric.
            data_roi_mae_UBFC_Phys_speech = []
            data_roi_mae_LGI_PPGI_rotation = []
            data_roi_mae_LGI_PPGI_gym = []
            data_roi_mae_LGI_PPGI_talk = []
            # PCC metric.
            data_roi_pcc_UBFC_Phys_speech = []
            data_roi_pcc_LGI_PPGI_rotation = []
            data_roi_pcc_LGI_PPGI_gym = []
            data_roi_pcc_LGI_PPGI_talk = []
            # SNR metric.
            data_roi_snr_UBFC_Phys_speech = []
            data_roi_snr_LGI_PPGI_rotation = []
            data_roi_snr_LGI_PPGI_gym = []
            data_roi_snr_LGI_PPGI_talk = []
        for i in range(len(list_roi_name)):
            # MAE metric.
            mae_UBFC_Phys_speech = df_UBFC_Phys_speech.loc[df_UBFC_Phys_speech['ROI'].values==list_roi_name[i], 'MAE'].dropna().values
            mae_LGI_PPGI_rotation = df_LGI_PPGI_rotation.loc[df_LGI_PPGI_rotation['ROI'].values==list_roi_name[i], 'MAE'].dropna().values
            mae_LGI_PPGI_gym = df_LGI_PPGI_gym.loc[df_LGI_PPGI_gym['ROI'].values==list_roi_name[i], 'MAE'].dropna().values
            mae_LGI_PPGI_talk = df_LGI_PPGI_talk.loc[df_LGI_PPGI_talk['ROI'].values==list_roi_name[i], 'MAE'].dropna().values
            if i_algorithm == 0:
                data_roi_mae_UBFC_Phys_speech.append(np.median(mae_UBFC_Phys_speech))
                data_roi_mae_LGI_PPGI_rotation.append(np.median(mae_LGI_PPGI_rotation))
                data_roi_mae_LGI_PPGI_gym.append(np.median(mae_LGI_PPGI_gym))
                data_roi_mae_LGI_PPGI_talk.append(np.median(mae_LGI_PPGI_talk))
            else:
                data_roi_mae_UBFC_Phys_speech[i] = data_roi_mae_UBFC_Phys_speech[i] + np.median(mae_UBFC_Phys_speech)
                data_roi_mae_LGI_PPGI_rotation[i] = data_roi_mae_LGI_PPGI_rotation[i] + np.median(mae_LGI_PPGI_rotation)
                data_roi_mae_LGI_PPGI_gym[i] = data_roi_mae_LGI_PPGI_gym[i] + np.median(mae_LGI_PPGI_gym)
                data_roi_mae_LGI_PPGI_talk[i] = data_roi_mae_LGI_PPGI_talk[i] + np.median(mae_LGI_PPGI_talk)
            # PCC metric.
            pcc_UBFC_Phys_speech = df_UBFC_Phys_speech.loc[df_UBFC_Phys_speech['ROI'].values==list_roi_name[i], 'PCC'].dropna().values
            pcc_LGI_PPGI_rotation = df_LGI_PPGI_rotation.loc[df_LGI_PPGI_rotation['ROI'].values==list_roi_name[i], 'PCC'].dropna().values
            pcc_LGI_PPGI_gym = df_LGI_PPGI_gym.loc[df_LGI_PPGI_gym['ROI'].values==list_roi_name[i], 'PCC'].dropna().values
            pcc_LGI_PPGI_talk = df_LGI_PPGI_talk.loc[df_LGI_PPGI_talk['ROI'].values==list_roi_name[i], 'PCC'].dropna().values
            if i_algorithm == 0:
                data_roi_pcc_UBFC_Phys_speech.append(np.median(pcc_UBFC_Phys_speech))
                data_roi_pcc_LGI_PPGI_rotation.append(np.median(pcc_LGI_PPGI_rotation))
                data_roi_pcc_LGI_PPGI_gym.append(np.median(pcc_LGI_PPGI_gym))
                data_roi_pcc_LGI_PPGI_talk.append(np.median(pcc_LGI_PPGI_talk))
            else:
                data_roi_pcc_UBFC_Phys_speech[i] = data_roi_pcc_UBFC_Phys_speech[i] + np.median(pcc_UBFC_Phys_speech)
                data_roi_pcc_LGI_PPGI_rotation[i] = data_roi_pcc_LGI_PPGI_rotation[i] + np.median(pcc_LGI_PPGI_rotation)
                data_roi_pcc_LGI_PPGI_gym[i] = data_roi_pcc_LGI_PPGI_gym[i] + np.median(pcc_LGI_PPGI_gym)
                data_roi_pcc_LGI_PPGI_talk[i] = data_roi_pcc_LGI_PPGI_talk[i] + np.median(pcc_LGI_PPGI_talk)
            # SNR metric.
            snr_UBFC_Phys_speech = df_UBFC_Phys_speech.loc[df_UBFC_Phys_speech['ROI'].values==list_roi_name[i], 'SNR'].dropna().values
            snr_LGI_PPGI_rotation = df_LGI_PPGI_rotation.loc[df_LGI_PPGI_rotation['ROI'].values==list_roi_name[i], 'SNR'].dropna().values
            snr_LGI_PPGI_gym = df_LGI_PPGI_gym.loc[df_LGI_PPGI_gym['ROI'].values==list_roi_name[i], 'SNR'].dropna().values
            snr_LGI_PPGI_talk = df_LGI_PPGI_talk.loc[df_LGI_PPGI_talk['ROI'].values==list_roi_name[i], 'SNR'].dropna().values
            if i_algorithm == 0:
                data_roi_snr_UBFC_Phys_speech.append(np.median(snr_UBFC_Phys_speech))
                data_roi_snr_LGI_PPGI_rotation.append(np.median(snr_LGI_PPGI_rotation))
                data_roi_snr_LGI_PPGI_gym.append(np.median(snr_LGI_PPGI_gym))
                data_roi_snr_LGI_PPGI_talk.append(np.median(snr_LGI_PPGI_talk))
            else:
                data_roi_snr_UBFC_Phys_speech[i] = data_roi_snr_UBFC_Phys_speech[i] + np.median(snr_UBFC_Phys_speech)
                data_roi_snr_LGI_PPGI_rotation[i] = data_roi_snr_LGI_PPGI_rotation[i] + np.median(snr_LGI_PPGI_rotation)
                data_roi_snr_LGI_PPGI_gym[i] = data_roi_snr_LGI_PPGI_gym[i] + np.median(snr_LGI_PPGI_gym)
                data_roi_snr_LGI_PPGI_talk[i] = data_roi_snr_LGI_PPGI_talk[i] + np.median(snr_LGI_PPGI_talk)
        pass
    # average over all included algorithms.
    # MAE metric.
    data_roi_mae_UBFC_Phys_speech = np.array(data_roi_mae_UBFC_Phys_speech, dtype=np.float32)/len(list_algorithm)
    data_roi_mae_LGI_PPGI_rotation = np.array(data_roi_mae_LGI_PPGI_rotation, dtype=np.float32)/len(list_algorithm)
    data_roi_mae_LGI_PPGI_gym = np.array(data_roi_mae_LGI_PPGI_gym, dtype=np.float32)/len(list_algorithm)
    data_roi_mae_LGI_PPGI_talk = np.array(data_roi_mae_LGI_PPGI_talk, dtype=np.float32)/len(list_algorithm)
    # PCC metric.
    data_roi_pcc_UBFC_Phys_speech = np.array(data_roi_pcc_UBFC_Phys_speech, dtype=np.float32)/len(list_algorithm)
    data_roi_pcc_LGI_PPGI_rotation = np.array(data_roi_pcc_LGI_PPGI_rotation, dtype=np.float32)/len(list_algorithm)
    data_roi_pcc_LGI_PPGI_gym = np.array(data_roi_pcc_LGI_PPGI_gym, dtype=np.float32)/len(list_algorithm)
    data_roi_pcc_LGI_PPGI_talk = np.array(data_roi_pcc_LGI_PPGI_talk, dtype=np.float32)/len(list_algorithm)
    # SNR metric.
    data_roi_snr_UBFC_Phys_speech = np.array(data_roi_snr_UBFC_Phys_speech, dtype=np.float32)/len(list_algorithm)
    data_roi_snr_LGI_PPGI_rotation = np.array(data_roi_snr_LGI_PPGI_rotation, dtype=np.float32)/len(list_algorithm)
    data_roi_snr_LGI_PPGI_gym = np.array(data_roi_snr_LGI_PPGI_gym, dtype=np.float32)/len(list_algorithm)
    data_roi_snr_LGI_PPGI_talk = np.array(data_roi_snr_LGI_PPGI_talk, dtype=np.float32)/len(list_algorithm)
    # overall score.
    data_roi_os_UBFC_Phys_speech = (1/3)*(np.max(data_roi_mae_UBFC_Phys_speech) - data_roi_mae_UBFC_Phys_speech)/ \
                                      (np.max(data_roi_mae_UBFC_Phys_speech) - np.min(data_roi_mae_UBFC_Phys_speech)) + \
                                      (1/3)*(data_roi_pcc_UBFC_Phys_speech - np.min(data_roi_pcc_UBFC_Phys_speech))/ \
                                      (np.max(data_roi_pcc_UBFC_Phys_speech) - np.min(data_roi_pcc_UBFC_Phys_speech)) + \
                                      (1/3)*(data_roi_snr_UBFC_Phys_speech - np.min(data_roi_snr_UBFC_Phys_speech))/ \
                                      (np.max(data_roi_snr_UBFC_Phys_speech) - np.min(data_roi_snr_UBFC_Phys_speech))
    data_roi_os_LGI_PPGI_rotation = (1/3)*(np.max(data_roi_mae_LGI_PPGI_rotation) - data_roi_mae_LGI_PPGI_rotation)/ \
                                    (np.max(data_roi_mae_LGI_PPGI_rotation) - np.min(data_roi_mae_LGI_PPGI_rotation)) + \
                                    (1/3)*(data_roi_pcc_LGI_PPGI_rotation - np.min(data_roi_pcc_LGI_PPGI_rotation))/ \
                                    (np.max(data_roi_pcc_LGI_PPGI_rotation) - np.min(data_roi_pcc_LGI_PPGI_rotation)) + \
                                    (1/3)*(data_roi_snr_LGI_PPGI_rotation - np.min(data_roi_snr_LGI_PPGI_rotation))/ \
                                    (np.max(data_roi_snr_LGI_PPGI_rotation) - np.min(data_roi_snr_LGI_PPGI_rotation))
    data_roi_os_LGI_PPGI_gym = (1/3)*(np.max(data_roi_mae_LGI_PPGI_gym) - data_roi_mae_LGI_PPGI_gym)/ \
                               (np.max(data_roi_mae_LGI_PPGI_gym) - np.min(data_roi_mae_LGI_PPGI_gym)) + \
                               (1/3)*(data_roi_pcc_LGI_PPGI_gym - np.min(data_roi_pcc_LGI_PPGI_gym))/ \
                               (np.max(data_roi_pcc_LGI_PPGI_gym) - np.min(data_roi_pcc_LGI_PPGI_gym)) + \
                               (1/3)*(data_roi_snr_LGI_PPGI_gym - np.min(data_roi_snr_LGI_PPGI_gym))/ \
                               (np.max(data_roi_snr_LGI_PPGI_gym) - np.min(data_roi_snr_LGI_PPGI_gym))
    data_roi_os_LGI_PPGI_talk = (1/3)*(np.max(data_roi_mae_LGI_PPGI_talk) - data_roi_mae_LGI_PPGI_talk)/ \
                                (np.max(data_roi_mae_LGI_PPGI_talk) - np.min(data_roi_mae_LGI_PPGI_talk)) + \
                                (1/3)*(data_roi_pcc_LGI_PPGI_talk - np.min(data_roi_pcc_LGI_PPGI_talk))/ \
                                (np.max(data_roi_pcc_LGI_PPGI_talk) - np.min(data_roi_pcc_LGI_PPGI_talk)) + \
                                (1/3)*(data_roi_snr_LGI_PPGI_talk - np.min(data_roi_snr_LGI_PPGI_talk))/ \
                                (np.max(data_roi_snr_LGI_PPGI_talk) - np.min(data_roi_snr_LGI_PPGI_talk))
    list_stack = [data_roi_os_UBFC_Phys_speech, data_roi_os_LGI_PPGI_rotation, 
                  data_roi_os_LGI_PPGI_gym, data_roi_os_LGI_PPGI_talk]
    pass
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
    p0 = plt.barh(np.arange(1, 29, 1), list_stack[0], width, edgecolor='black', color=np.array([255, 230, 183])/255)
    p1 = plt.barh(np.arange(1, 29, 1), list_stack[1], width, left=list_stack[0], edgecolor='black', color=np.array([255, 208, 111])/255)
    p2 = plt.barh(np.arange(1, 29, 1), list_stack[2], width, left=list_stack[0]+list_stack[1], edgecolor='black', color=np.array([239, 138, 71])/255)
    p3 = plt.barh(np.arange(1, 29, 1), list_stack[3], width, left=list_stack[0]+list_stack[1]+list_stack[2], edgecolor='black', color=np.array([231, 98, 84])/255)
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
    main_motion_stackbar(list_algorithm=list_algorithm)