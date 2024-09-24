"""
performance evaluation of different rPPG methods.
"""

# Author: Shuo Li
# Date: 2023/08/05

import os
import sys
import numpy as np
import pandas as pd
dir_crt = os.getcwd()
sys.path.append(os.path.join(dir_crt, 'util'))
import util_pre_analysis


def main_eval(name_dataset='UBFC-rPPG', algorithm='CHROM'):
    """Evaluation pipeline for a given dataset using a given algorithm.
    Parameters
    ----------
    name_dataset: name of the selected rPPG dataset. 
                  ['UBFC-rPPG', 'UBFC-Phys', 'LGI-PPGI', 'BUAA-MIHR'].
    algorithm: selected rPPG algorithm. ['CHROM', 'LGI', 'OMIT', 'POS'].

    Returns
    -------

    """
    # get current directory.
    dir_crt = os.getcwd()
    dir_option = os.path.join(dir_crt, 'config', 'options.yaml')
    # parameter class initialization.
    Params = util_pre_analysis.Params(dir_option=dir_option, name_dataset=name_dataset)
    # groundtruth class initialization.
    GT = util_pre_analysis.GroundTruth(dir_dataset=Params.dir_dataset, name_dataset=name_dataset)
    # structures for different datasets.
    if name_dataset == 'UBFC-rPPG':  # UBFC-rPPG dataset.
        list_attendant = [1] + list(range(3, 6)) + list(range(8, 19)) + [20] + \
                         list(range(22, 27)) + list(range(30, 50))  # attendant sequence num.
        # dataframe initialization.
        df_eval = pd.DataFrame(columns=['attendant', 'ROI', 'DTW', 'PCC', 'CCC', 'RMSE', 'MAE'])
        # loop over all data points.
        for num_attendant in list_attendant:
            print([name_dataset, algorithm, num_attendant])
            # load BVP and HR signals.
            dir_hr = os.path.join(dir_crt, 'data', name_dataset, 'hr', str(num_attendant)+'_'+algorithm+'.csv')
            df_hr = pd.read_csv(dir_hr, index_col=0)
            # load ground truth.
            gtTime, gtTrace, gtHR = GT.get_GT(specification=['realistic', num_attendant], 
                                              num_frame_interp=int(len(df_hr)/len(Params.list_roi_name)), 
                                              slice=[0, 1])
            # initialization of BVP and BPM arrays.
            sig_bvp = np.zeros(shape=[int(len(df_hr)/len(Params.list_roi_name)), len(Params.list_roi_name)])
            sig_bpm = np.zeros(shape=[int(len(df_hr)/len(Params.list_roi_name)), len(Params.list_roi_name)])
            for i_roi in range(len(Params.list_roi_name)):
                sig_bvp[:, i_roi] = df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BVP']
                sig_bpm[:, i_roi] = df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BPM']
            # metrics calculation.
            df_metric = util_pre_analysis.eval_pipe(sig_bvp, sig_bpm, gtTime, gtTrace, gtHR, Params)
            df_eval = pd.concat([df_eval, df_metric])
            df_eval.reset_index(drop=True, inplace=True)
            df_eval.loc[len(df_eval)-len(Params.list_roi_name):, 'attendant'] = num_attendant
            # dataframe saving.
            df_eval.to_csv(os.path.join(dir_crt, 'result', name_dataset, 'evaluation_'+algorithm+'.csv'))


    elif name_dataset == 'UBFC-Phys':   # UBFC-Phys dataset.
        # name of attendants.
        list_attendant = list(range(1, 57))
        # condition types.
        list_condition = [1]  # [1, 2, 3]
        # dataframe initialization.
        df_eval = pd.DataFrame(columns=['attendant', 'condition', 'ROI', 'DTW', 'PCC', 'CCC', 'RMSE', 'MAE'])
        # loop over all data points.
        for num_attendant in list_attendant:
            for condition in list_condition:
                print([name_dataset, algorithm, num_attendant, condition])
                # load BVP and HR signals.
                dir_hr = os.path.join(dir_crt, 'data', name_dataset, 'hr', str(num_attendant)+'_'+str(condition)+'_'+algorithm+'.csv')
                df_hr = pd.read_csv(dir_hr, index_col=0)
                # load groundtruth.
                gtTime, gtTrace, gtHR = GT.get_GT(specification=[num_attendant, condition], 
                                                  num_frame_interp=int(len(df_hr)/len(Params.list_roi_name)), 
                                                  slice=[0, 1])
                # initialization of BVP and BPM arrays.
                sig_bvp = np.zeros(shape=[int(len(df_hr)/len(Params.list_roi_name)), len(Params.list_roi_name)])
                sig_bpm = np.zeros(shape=[int(len(df_hr)/len(Params.list_roi_name)), len(Params.list_roi_name)])
                for i_roi in range(len(Params.list_roi_name)):
                    sig_bvp[:, i_roi] = df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BVP']
                    sig_bpm[:, i_roi] = df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BPM']
                # metrics calculation.
                df_metric = util_pre_analysis.eval_pipe(sig_bvp, sig_bpm, gtTime, gtTrace, gtHR, Params)
                df_eval = pd.concat([df_eval, df_metric])
                df_eval.reset_index(drop=True, inplace=True)
                df_eval.loc[len(df_eval)-len(Params.list_roi_name):, 'attendant'] = num_attendant
                df_eval.loc[len(df_eval)-len(Params.list_roi_name):, 'condition'] = condition
                # dataframe saving.
                df_eval.to_csv(os.path.join(dir_crt, 'result', name_dataset, 'evaluation_'+algorithm+'.csv'))


    elif name_dataset == 'LGI-PPGI':   # LGI-PPGI dataset.
        list_attendant = ['angelo', 'david', 'alex', 'cpi', 'felix', 'harun']  # attendant name.
        list_motion = ['resting', 'gym', 'rotation', 'talk']  # motion type.
        # dataframe initialization.
        df_eval = pd.DataFrame(columns=['attendant', 'motion', 'ROI', 'DTW', 'PCC', 'CCC', 'RMSE', 'MAE'])
        # loop over all data points.
        for name_attendant in list_attendant:
            for motion in list_motion:
                print([name_dataset, algorithm, name_attendant, motion])
                # load BVP and HR signals.
                dir_hr = os.path.join(dir_crt, 'data', name_dataset, 'hr', name_attendant+'_'+motion+'_'+algorithm+'.csv')
                df_hr = pd.read_csv(dir_hr, index_col=0)
                # load groundtruth.
                gtTime, gtTrace, gtHR = GT.get_GT(specification=[name_attendant, motion], 
                                                  num_frame_interp=int(len(df_hr)/len(Params.list_roi_name)), 
                                                  slice=[0, 1])
                # initialization of BVP and BPM arrays.
                sig_bvp = np.zeros(shape=[int(len(df_hr)/len(Params.list_roi_name)), len(Params.list_roi_name)])
                sig_bpm = np.zeros(shape=[int(len(df_hr)/len(Params.list_roi_name)), len(Params.list_roi_name)])
                for i_roi in range(len(Params.list_roi_name)):
                    sig_bvp[:, i_roi] = df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BVP']
                    sig_bpm[:, i_roi] = df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BPM']
                # metrics calculation.
                df_metric = util_pre_analysis.eval_pipe(sig_bvp, sig_bpm, gtTime, gtTrace, gtHR, Params)
                df_eval = pd.concat([df_eval, df_metric])
                df_eval.reset_index(drop=True, inplace=True)
                df_eval.loc[len(df_eval)-len(Params.list_roi_name):, 'attendant'] = name_attendant
                df_eval.loc[len(df_eval)-len(Params.list_roi_name):, 'motion'] = motion
                # dataframe saving.
                df_eval.to_csv(os.path.join(dir_crt, 'result', name_dataset, 'evaluation_'+algorithm+'.csv'))
        

    elif name_dataset == 'BUAA-MIHR':   # BUAA-MIHR dataset.
        # sequnce num of attendants.
        list_attendant = list(range(1, 14))
        # illumination levels.
        list_lux = ['lux 10.0', 'lux 15.8', 'lux 25.1', 'lux 39.8', 'lux 63.1', 'lux 100.0']
        # attendant names.
        list_name = ['APH', 'GDB', 'HB', 'KLK', 'LW', 'LXR', 'LY', 'LZY', 'LMX', 'MXZ', 'PYC', 'QSH', 'WQT']
        # dataframe initialization.
        df_eval = pd.DataFrame(columns=['attendant', 'lux', 'ROI', 'DTW', 'PCC', 'CCC', 'RMSE', 'MAE'])
        # loop over all data points.
        for num_attendant in list_attendant:
            for lux in list_lux:
                print([name_dataset, algorithm, num_attendant, lux])
                # load BVP and HR signals.
                dir_hr = os.path.join(dir_crt, 'data', name_dataset, 'hr', str(num_attendant).zfill(2) + \
                                      '_' + str(lux).replace(' ', '') + '_' + algorithm+'.csv')
                df_hr = pd.read_csv(dir_hr, index_col=None)
                # load groundtruth.
                name = list_name[num_attendant-1]
                gtTime, gtTrace, gtHR = GT.get_GT(specification=[num_attendant, lux, name], 
                                                  num_frame_interp=int(len(df_hr)/len(Params.list_roi_name)), 
                                                  slice=[0, 1])
                # initialization of BVP and BPM arrays.
                sig_bvp = np.zeros(shape=[int(len(df_hr)/len(Params.list_roi_name)), len(Params.list_roi_name)])
                sig_bpm = np.zeros(shape=[int(len(df_hr)/len(Params.list_roi_name)), len(Params.list_roi_name)])
                for i_roi in range(len(Params.list_roi_name)):
                    sig_bvp[:, i_roi] = df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BVP']
                    sig_bpm[:, i_roi] = df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BPM']
                # metrics calculation.
                df_metric = util_pre_analysis.eval_pipe(sig_bvp, sig_bpm, gtTime, gtTrace, gtHR, Params)
                df_eval = pd.concat([df_eval, df_metric])
                df_eval.reset_index(drop=True, inplace=True)
                # assign data point information.
                df_eval.loc[len(df_eval)-len(Params.list_roi_name):, 'attendant'] = num_attendant
                df_eval.loc[len(df_eval)-len(Params.list_roi_name):, 'lux'] = lux.replace('lux ', '')
                # dataframe saving.
                df_eval.to_csv(os.path.join(dir_crt, 'result', name_dataset, 'evaluation_'+algorithm+'.csv'))


if __name__ == "__main__":
    list_dataset = ['UBFC-Phys']  # ['UBFC-rPPG', 'UBFC-Phys', 'LGI-PPGI', 'BUAA-MIHR'].
    list_algorithm = ['POS']  # ['CHROM', 'LGI', 'OMIT', 'POS'].
    # loop over all selected rPPG datasets.
    for name_dataset in list_dataset:
        # loop over all selected rPPG algorithms.
        for algorithm in list_algorithm:
            print([name_dataset, algorithm])
            main_eval(name_dataset=name_dataset, algorithm=algorithm)