"""
generate ground truth BPM data using existing BVP data for UBFC-Phys dataset.
"""

# Author: Shuo Li
# Date: 2023/08/18

import os
import cv2
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from pyVHR.BPM.utils import Welch


def main_gen_gtHR(dir_dataset):
    """main function for generating ground truth BPM data for UBFC-Phys dataset.
    Parameters
    ----------
    dir_dataset: directory of the dataset (UBFC-Phys).
    Params: a class containing the pre-defined parameters for the preliminary analysis.
    
    Returns
    -------

    """

    len_window = 6   # window length in seconds.
    stride_window = 1   # window stride in seconds.
    nFFT = 2048//1   # freq. resolution for STFTs.
    minHz = 0.65  # minimal frequency in Hz.
    maxHz = 4.0   # maximal frequency in Hz.

    # list of attendants.
    list_attendant = list(range(1, 29)) + list(range(30, 55)) + [56]
    # list of conditions.
    list_conition = [1, 2, 3]
    # loop over all attendants.
    for num_attendant in tqdm(list_attendant):

        # loop over all conditions.
        for num_condition in list_conition:
            # ground truth BVP signal.
            dir_bvp = os.path.join(dir_dataset, 's'+str(num_attendant), 'bvp_s'+str(num_attendant)+'_T'+str(num_condition)+'.csv')
            df_bvp = pd.read_csv(dir_bvp, index_col=None)
            sig_bvp = df_bvp.values
            # calculate frames per second of the BVP signal.
            dir_video = os.path.join(dir_dataset, 's'+str(num_attendant), 'vid_s'+str(num_attendant)+'_T'+str(num_condition)+'.avi')
            cap = cv2.VideoCapture(dir_video)
            duration = cap.get(7)/cap.get(5)   # video duration (sec).
            fps = len(sig_bvp)/duration
            # calculate window length and stride length.
            len_window_frame = int(len_window * fps)
            stride_window_frame = int(stride_window * fps)
            # BPM signal initialization
            sig_bpm = np.zeros_like(sig_bvp)
            # windowing operation for HR estimation.
            idx_crt = 0   # current index.
            while (idx_crt+len_window_frame-1) <= (len(sig_bvp)-1):
                # signal slicing.
                sig_bvp_slice = sig_bvp[idx_crt:(idx_crt+len_window_frame-1)]
                # Welch's method.
                Pfreqs, Power = Welch(np.reshape(sig_bvp_slice, newshape=[1, len(sig_bvp_slice)]), fps, minHz, maxHz, nFFT)
                Pmax = np.argmax(Power, axis=1)  # power max.
                sig_bpm[int(0.5*(2*idx_crt+len_window_frame-1))] = Pfreqs[Pmax.squeeze()]
                idx_crt = idx_crt + stride_window_frame   # move forward.
            # linear interpolation.
            sig_bpm[sig_bpm == 0] = np.nan
            df_sig_bpm = pd.Series(data=sig_bpm[:, 0])
            df_sig_bpm = df_sig_bpm.interpolate(method='nearest')
            df_sig_bpm = df_sig_bpm.fillna(method='ffill')
            df_sig_bpm = df_sig_bpm.fillna(method='bfill')
            # save BPM signal.
            dir_bpm = os.path.join(dir_dataset, 's'+str(num_attendant), 'bpm_s'+str(num_attendant)+'_T'+str(num_condition)+'.csv')
            df_sig_bpm.to_csv(dir_bpm, index=False, header=None)


if __name__ == "__main__":
    # generate ground truth HR for UBFC-Phys.
    dir_crt = os.getcwd()
    dir_option = os.path.join(dir_crt, 'config', 'options.yaml')
    dir_dataset = yaml.safe_load(open(dir_option))['UBFC-Phys']['dir_dataset']
    main_gen_gtHR(dir_dataset=dir_dataset)