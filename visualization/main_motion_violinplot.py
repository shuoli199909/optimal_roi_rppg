"""
the experiment for examining the effect of ROI selection under different subject's motion types and cognitive tasks.
"""

# Author: Shuo Li
# Date: 2024/09/15

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main_motion_violinplot(list_algorithm):
    """main function to visualize the violin plot on the cognitive dataset.
    Parameters
    ----------
    list_algorithm: list of select rPPG algorithms.   # ['CHROM', 'POS', 'LGI', 'OMIT'].
    
    Returns
    -------

    """

    # get project directory.
    dir_crt = os.getcwd()
    # list of ROI names.
    list_roi_name = ['glabella', 'lower medial forehead', 'left lower lateral forehead', 
                     'right lower lateral forehead', 'left malar', 'right malar', 'upper nasal dorsum']
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
            data_roi_mae = []
            # PCC metric.
            data_roi_pcc = []
            # SNR metric.
            data_roi_snr = []
        for i in range(len(list_roi_name)):
            # MAE metric.
            mae_UBFC_Phys_speech = df_UBFC_Phys_speech.loc[df_UBFC_Phys_speech['ROI'].values==list_roi_name[i], 'MAE'].values
            np.nan_to_num(mae_UBFC_Phys_speech, copy=False, nan=np.nanmedian(mae_UBFC_Phys_speech))
            mae_LGI_PPGI_rotation = df_LGI_PPGI_rotation.loc[df_LGI_PPGI_rotation['ROI'].values==list_roi_name[i], 'MAE'].values
            np.nan_to_num(mae_LGI_PPGI_rotation, copy=False, nan=np.nanmedian(mae_LGI_PPGI_rotation))
            mae_LGI_PPGI_gym = df_LGI_PPGI_gym.loc[df_LGI_PPGI_gym['ROI'].values==list_roi_name[i], 'MAE'].values
            np.nan_to_num(mae_LGI_PPGI_gym, copy=False, nan=np.nanmedian(mae_LGI_PPGI_gym))
            mae_LGI_PPGI_talk = df_LGI_PPGI_talk.loc[df_LGI_PPGI_talk['ROI'].values==list_roi_name[i], 'MAE'].values
            np.nan_to_num(mae_LGI_PPGI_talk, copy=False, nan=np.nanmedian(mae_LGI_PPGI_talk))
            if i_algorithm == 0:
                data_roi_mae.append(np.array(mae_UBFC_Phys_speech.tolist() + mae_LGI_PPGI_rotation.tolist() + \
                                             mae_LGI_PPGI_gym.tolist() + mae_LGI_PPGI_talk.tolist()))
            else:
                data_roi_mae[i] = data_roi_mae[i] + np.array(mae_UBFC_Phys_speech.tolist() + mae_LGI_PPGI_rotation.tolist() + \
                                                             mae_LGI_PPGI_gym.tolist() + mae_LGI_PPGI_talk.tolist())
            # PCC metric.
            pcc_UBFC_Phys_speech = df_UBFC_Phys_speech.loc[df_UBFC_Phys_speech['ROI'].values==list_roi_name[i], 'PCC'].values
            np.nan_to_num(pcc_UBFC_Phys_speech, copy=False, nan=np.nanmedian(pcc_UBFC_Phys_speech))
            pcc_LGI_PPGI_rotation = df_LGI_PPGI_rotation.loc[df_LGI_PPGI_rotation['ROI'].values==list_roi_name[i], 'PCC'].values
            np.nan_to_num(pcc_LGI_PPGI_rotation, copy=False, nan=np.nanmedian(pcc_LGI_PPGI_rotation))
            pcc_LGI_PPGI_gym = df_LGI_PPGI_gym.loc[df_LGI_PPGI_gym['ROI'].values==list_roi_name[i], 'PCC'].values
            np.nan_to_num(pcc_LGI_PPGI_gym, copy=False, nan=np.nanmedian(pcc_LGI_PPGI_gym))
            pcc_LGI_PPGI_talk = df_LGI_PPGI_talk.loc[df_LGI_PPGI_talk['ROI'].values==list_roi_name[i], 'PCC'].values
            np.nan_to_num(pcc_LGI_PPGI_talk, copy=False, nan=np.nanmedian(pcc_LGI_PPGI_talk))
            if i_algorithm == 0:
                data_roi_pcc.append(np.array(pcc_UBFC_Phys_speech.tolist() + pcc_LGI_PPGI_rotation.tolist() + \
                                             pcc_LGI_PPGI_gym.tolist() + pcc_LGI_PPGI_talk.tolist()))
            else:
                data_roi_pcc[i] = data_roi_pcc[i] + np.array(pcc_UBFC_Phys_speech.tolist() + pcc_LGI_PPGI_rotation.tolist() + \
                                                             pcc_LGI_PPGI_gym.tolist() + pcc_LGI_PPGI_talk.tolist())
            # SNR metric.
            snr_UBFC_Phys_speech = df_UBFC_Phys_speech.loc[df_UBFC_Phys_speech['ROI'].values==list_roi_name[i], 'SNR'].values
            np.nan_to_num(snr_UBFC_Phys_speech, copy=False, nan=np.nanmedian(snr_UBFC_Phys_speech))
            snr_LGI_PPGI_rotation = df_LGI_PPGI_rotation.loc[df_LGI_PPGI_rotation['ROI'].values==list_roi_name[i], 'SNR'].values
            np.nan_to_num(snr_LGI_PPGI_rotation, copy=False, nan=np.nanmedian(snr_LGI_PPGI_rotation))
            snr_LGI_PPGI_gym = df_LGI_PPGI_gym.loc[df_LGI_PPGI_gym['ROI'].values==list_roi_name[i], 'SNR'].values
            np.nan_to_num(snr_LGI_PPGI_gym, copy=False, nan=np.nanmedian(snr_LGI_PPGI_gym))
            snr_LGI_PPGI_talk = df_LGI_PPGI_talk.loc[df_LGI_PPGI_talk['ROI'].values==list_roi_name[i], 'SNR'].values
            np.nan_to_num(snr_LGI_PPGI_talk, copy=False, nan=np.nanmedian(snr_LGI_PPGI_talk))
            if i_algorithm == 0:
                data_roi_snr.append(np.array(snr_UBFC_Phys_speech.tolist() + snr_LGI_PPGI_rotation.tolist() + \
                                             snr_LGI_PPGI_gym.tolist() + snr_LGI_PPGI_talk.tolist()))
            else:
                data_roi_snr[i] = data_roi_snr[i] + np.array(snr_UBFC_Phys_speech.tolist() + snr_LGI_PPGI_rotation.tolist() + \
                                                             snr_LGI_PPGI_gym.tolist() + snr_LGI_PPGI_talk.tolist())
    # average over all included algorithms.
    # MAE metric.
    data_roi_mae = np.array(data_roi_mae)/len(list_algorithm)
    # PCC metric.
    data_roi_pcc = np.array(data_roi_pcc)/len(list_algorithm)
    # SNR metric.
    data_roi_snr = np.array(data_roi_snr)/len(list_algorithm)
    # overall score.
    # data visualization.
    # MAE.
    plt.cla()
    dir_save = os.path.join(dir_crt, 'plot', 'motion', 'violinplot_mae_motion.png')
    plt.violinplot(data_roi_mae.T, showmedians=True)
    plt.axhline(y=np.min(np.median(data_roi_mae, axis=1)), c='green', ls='dotted')
    figure = plt.gcf()
    figure.set_size_inches(18, 9)
    plt.show()
    # PCC.
    plt.cla()
    dir_save = os.path.join(dir_crt, 'plot', 'motion', 'violinplot_pcc_motion.png')
    plt.violinplot(data_roi_pcc.T, showmedians=True)
    plt.axhline(y=np.max(np.median(data_roi_pcc, axis=1)), c='green', ls='dotted')
    figure = plt.gcf()
    figure.set_size_inches(18, 9)
    plt.show()
    # SNR.
    plt.cla()
    dir_save = os.path.join(dir_crt, 'plot', 'motion', 'violinplot_snr_motion.png')
    plt.violinplot(data_roi_snr.T, showmedians=True)
    plt.axhline(y=np.max(np.median(data_roi_snr, axis=1)), c='green', ls='dotted')
    figure = plt.gcf()
    figure.set_size_inches(18, 9)
    plt.show()
    pass


if __name__ == "__main__":
    list_algorithm = ['CHROM', 'POS', 'LGI', 'OMIT']   # ['LGI', 'OMIT', 'CHROM', 'POS'].
    # loop over all performance evaluation metrics.
    main_motion_violinplot(list_algorithm=list_algorithm)