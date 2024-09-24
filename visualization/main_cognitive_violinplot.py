"""
the experiment for examining the effect of ROI selection under different subject's motion types and cognitive tasks.
"""

# Author: Shuo Li
# Date: 2024/09/15

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main_cognitive_violinplot(list_algorithm):
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
            data_roi_mae = []
            # PCC metric.
            data_roi_pcc = []
            # SNR metric.
            data_roi_snr = []
        for i in range(len(list_roi_name)):
            # MAE metric.
            mae_UBFC_rPPG_realistic = df_UBFC_rPPG_realistic.loc[df_UBFC_rPPG_realistic['ROI'].values==list_roi_name[i], 'MAE'].values
            np.nan_to_num(mae_UBFC_rPPG_realistic, copy=False, nan=np.nanmedian(mae_UBFC_rPPG_realistic))
            mae_UBFC_Phys_arithmetic = df_UBFC_Phys_arithmetic.loc[df_UBFC_Phys_arithmetic['ROI'].values==list_roi_name[i], 'MAE'].values
            np.nan_to_num(mae_UBFC_Phys_arithmetic, copy=False, nan=np.nanmedian(mae_UBFC_Phys_arithmetic))
            if i_algorithm == 0:
                data_roi_mae.append(np.array(mae_UBFC_rPPG_realistic.tolist() + mae_UBFC_Phys_arithmetic.tolist()))
            else:
                data_roi_mae[i] = data_roi_mae[i] + np.array(mae_UBFC_rPPG_realistic.tolist() + mae_UBFC_Phys_arithmetic.tolist())
            # PCC metric.
            pcc_UBFC_rPPG_realistic = df_UBFC_rPPG_realistic.loc[df_UBFC_rPPG_realistic['ROI'].values==list_roi_name[i], 'PCC'].values
            np.nan_to_num(pcc_UBFC_rPPG_realistic, copy=False, nan=np.nanmedian(pcc_UBFC_rPPG_realistic))
            pcc_UBFC_Phys_arithmetic = df_UBFC_Phys_arithmetic.loc[df_UBFC_Phys_arithmetic['ROI'].values==list_roi_name[i], 'PCC'].values
            np.nan_to_num(pcc_UBFC_Phys_arithmetic, copy=False, nan=np.nanmedian(pcc_UBFC_Phys_arithmetic))
            if i_algorithm == 0:
                data_roi_pcc.append(np.array(pcc_UBFC_rPPG_realistic.tolist() + pcc_UBFC_Phys_arithmetic.tolist()))
            else:
                data_roi_pcc[i] = data_roi_pcc[i] + np.array(pcc_UBFC_rPPG_realistic.tolist() + pcc_UBFC_Phys_arithmetic.tolist())
            # SNR metric.
            snr_UBFC_rPPG_realistic = df_UBFC_rPPG_realistic.loc[df_UBFC_rPPG_realistic['ROI'].values==list_roi_name[i], 'SNR'].values
            snr_UBFC_rPPG_realistic[np.isinf(snr_UBFC_rPPG_realistic)] = np.median(snr_UBFC_rPPG_realistic)
            np.nan_to_num(snr_UBFC_rPPG_realistic, copy=False, nan=np.nanmedian(snr_UBFC_rPPG_realistic))
            snr_UBFC_Phys_arithmetic = df_UBFC_Phys_arithmetic.loc[df_UBFC_Phys_arithmetic['ROI'].values==list_roi_name[i], 'SNR'].values
            snr_UBFC_Phys_arithmetic[np.isinf(snr_UBFC_Phys_arithmetic)] = np.median(snr_UBFC_Phys_arithmetic)
            np.nan_to_num(snr_UBFC_Phys_arithmetic, copy=False, nan=np.nanmedian(snr_UBFC_Phys_arithmetic))
            if i_algorithm == 0:
                data_roi_snr.append(np.array(snr_UBFC_rPPG_realistic.tolist() + snr_UBFC_Phys_arithmetic.tolist()))
            else:
                data_roi_snr[i] = data_roi_snr[i] + np.array(snr_UBFC_rPPG_realistic.tolist() + snr_UBFC_Phys_arithmetic.tolist())
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
    k = 6
    df = pd.DataFrame(data=[data_roi_mae[k, :], data_roi_pcc[k, :], data_roi_snr[k, :]]).T
    dir_save = os.path.join(dir_crt, 'plot', 'motion', 'test.csv')
    df.to_csv(dir_save)
    plt.cla()
    dir_save = os.path.join(dir_crt, 'plot', 'motion', 'violinplot_mae_cognitive.png')
    plt.violinplot(data_roi_mae.T, showmedians=True)
    plt.axhline(y=np.min(np.median(data_roi_mae, axis=1)), c='green', ls='dotted')
    figure = plt.gcf()
    figure.set_size_inches(18, 9)
    plt.show()
    # PCC.
    plt.cla()
    dir_save = os.path.join(dir_crt, 'plot', 'motion', 'violinplot_pcc_cognitive.png')
    plt.violinplot(data_roi_pcc.T, showmedians=True)
    plt.axhline(y=np.max(np.median(data_roi_pcc, axis=1)), c='green', ls='dotted')
    figure = plt.gcf()
    figure.set_size_inches(18, 9)
    plt.show()
    # SNR.
    plt.cla()
    dir_save = os.path.join(dir_crt, 'plot', 'motion', 'violinplot_snr_cognitive.png')
    plt.violinplot(data_roi_snr.T, showmedians=True)
    plt.axhline(y=np.max(np.median(data_roi_snr, axis=1)), c='green', ls='dotted')
    figure = plt.gcf()
    figure.set_size_inches(18, 9)
    plt.show()
    pass



if __name__ == "__main__":
    list_algorithm = ['CHROM', 'POS', 'LGI', 'OMIT']   # ['LGI', 'OMIT', 'CHROM', 'POS'].
    # loop over all performance evaluation metrics.
    main_cognitive_violinplot(list_algorithm=list_algorithm)