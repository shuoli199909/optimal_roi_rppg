"""
Generate ground truth BPM data using existing BVP data for UBFC-Phys dataset.
"""

# Author: Shuo Li
# Date: 2023/08/18

import warnings
warnings.filterwarnings("ignore")  # Ignore unnecessary warnings.
import os
import sys
import cv2
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
dir_crt = os.getcwd()
sys.path.append(os.path.join(dir_crt, 'util'))
import util_pyVHR


def main_gen_gtHR(dir_dataset):
    """Main function for generating ground truth BPM data for UBFC-Phys dataset.
    Parameters
    ----------
    dir_dataset: Directory of the dataset (UBFC-Phys).
    Params: A class containing the pre-defined parameters for the preliminary analysis.
    
    Returns
    -------

    """

    len_window = 6   # Window length in seconds.
    stride_window = 1   # Window stride in seconds.
    nFFT = 2048//1   # Freq. Resolution for STFTs.
    minHz = 0.65  # Minimal frequency in Hz.
    maxHz = 4.0   # Maximal frequency in Hz.

    # List of attendants.
    list_attendant = list(range(1, 29)) + list(range(30, 55)) + [56]
    # List of conditions.
    list_conition = [1, 2, 3]
    # Loop over all attendants.
    for num_attendant in tqdm(list_attendant):

        # Loop over all conditions.
        for num_condition in list_conition:
            # Ground truth BVP signal.
            dir_bvp = os.path.join(dir_dataset, 's'+str(num_attendant), 'bvp_s'+str(num_attendant)+'_T'+str(num_condition)+'.csv')
            df_bvp = pd.read_csv(dir_bvp, index_col=None)
            sig_bvp = df_bvp.values
            # Calculate frames per second of the BVP signal.
            dir_video = os.path.join(dir_dataset, 's'+str(num_attendant), 'vid_s'+str(num_attendant)+'_T'+str(num_condition)+'.avi')
            cap = cv2.VideoCapture(dir_video)
            duration = cap.get(7)/cap.get(5)   # Video duration (sec).
            fps = len(sig_bvp)/duration
            # Calculate window length and stride length.
            len_window_frame = int(len_window * fps)
            stride_window_frame = int(stride_window * fps)
            # BPM signal initialization
            sig_bpm = np.zeros_like(sig_bvp)
            # Windowing operation for HR estimation.
            idx_crt = 0   # current index.
            while (idx_crt+len_window_frame-1) <= (len(sig_bvp)-1):
                # Signal slicing.
                sig_bvp_slice = sig_bvp[idx_crt:(idx_crt+len_window_frame-1)]
                # Welch's method of filtering.
                Pfreqs, Power = util_pyVHR.Welch(np.reshape(sig_bvp_slice, newshape=[1, len(sig_bvp_slice)]), fps, minHz, maxHz, nFFT)
                Pmax = np.argmax(Power, axis=1)  # Power max.
                sig_bpm[int(0.5*(2*idx_crt+len_window_frame-1))] = Pfreqs[Pmax.squeeze()]
                idx_crt = idx_crt + stride_window_frame   # Move forward.
            # Linear interpolation.
            sig_bpm[sig_bpm == 0] = np.nan
            df_sig_bpm = pd.Series(data=sig_bpm[:, 0])
            df_sig_bpm = df_sig_bpm.interpolate(method='linear')
            df_sig_bpm = df_sig_bpm.ffill().bfill()
            # Save BPM signal.
            dir_bpm = os.path.join(dir_dataset, 's'+str(num_attendant), 'bpm_s'+str(num_attendant)+'_T'+str(num_condition)+'.csv')
            df_sig_bpm.to_csv(dir_bpm, index=False, header=None)


if __name__ == "__main__":
    # Generate ground truth HR for UBFC-Phys.
    dir_crt = os.getcwd()
    dir_option = os.path.join(dir_crt, 'config', 'options.yaml')
    dir_dataset = yaml.safe_load(open(dir_option))['UBFC-Phys']['dir_dataset']
    main_gen_gtHR(dir_dataset=dir_dataset)