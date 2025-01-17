"""
Transform raw RGB traces to BVP and HR signals.
"""

# Author: Shuo Li
# Date: 2023/07/18

import warnings
warnings.filterwarnings("ignore")  # Ignore unnecessary warnings.
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Turn off oneDNN custom operations.
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
dir_crt = os.getcwd()
sys.path.append(os.path.join(dir_crt, 'util'))
import util_analysis


def main_rgb2hr(name_dataset, algorithm):
    """Main function for transforming RGB traces to HR-related signals.
    Parameters
    ----------
    name_dataset: Name of the selected dataset.
                  ['UBFC-rPPG', 'UBFC-Phys', 'LGI-PPGI', 'BUAA-MIHR'].
    algorithm: Selected rPPG algorithm. ['CHROM', 'LGI', 'OMIT', 'POS'].
    
    Returns
    -------

    """
    # Get current directory.
    dir_crt = os.getcwd()
    dir_option = os.path.join(dir_crt, 'config', 'options.yaml')
    # Parameter class initialization.
    Params = util_analysis.Params(dir_option=dir_option, name_dataset=name_dataset)

    # RGB signal -> bvp signal.
    if name_dataset == 'UBFC-rPPG':
        # Sequence num of attendants.
        list_attendant = [1] + list(range(3, 6)) + list(range(8, 19)) + [20] + list(range(22, 27)) + list(range(30, 50))
        # Loop over all attendants.
        for num_attendant in tqdm(list_attendant):
            # Parse the RGB signal from the RGB dataframe. Size = [num_frames, num_ROI, rgb_channels(3)].
            dir_sig_rgb = os.path.join(dir_crt, 'data', name_dataset, 'rgb', str(num_attendant)+'.csv')
            df_rgb = pd.read_csv(dir_sig_rgb)
            # RGB signal initialization.
            sig_rgb = np.zeros([df_rgb['frame'].max(), len(np.unique(df_rgb['ROI'].values)), 3])
            # Loop over all ROIs.
            for i_roi in range(len(Params.list_roi_name)):
                sig_rgb[:, i_roi, 0] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'R']  # Red channel.
                sig_rgb[:, i_roi, 1] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'G']  # Green channel.
                sig_rgb[:, i_roi, 2] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'B']  # Blue channel.
            # RGB video information.
            dir_vid = os.path.join(Params.dir_dataset, 'DATASET_2', 'subject'+str(num_attendant), 'vid.avi')
            # Get video fps.
            capture = cv2.VideoCapture(dir_vid)
            Params.fps = capture.get(cv2.CAP_PROP_FPS)
            # RGB signal -> bvp signal & bpm signal.
            sig_bvp, sig_bpm = util_analysis.rppg_hr_pipe(sig_rgb=sig_rgb, method=algorithm, Params=Params)
            # Create the dataframe to save the HR-related data (bvp signal & bpm signal).
            df_hr = pd.DataFrame(columns=['frame', 'time', 'ROI', 'BVP', 'BPM'], index=list(range(len(df_rgb))))
            df_hr.loc[:, ['frame', 'time', 'ROI']] = df_rgb.loc[:, ['frame', 'time', 'ROI']]
            # Loop over all ROIs.
            for i_roi in range(len(Params.list_roi_name)):
                df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BVP'] = sig_bvp[:, i_roi]  # BVP signal.
                df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BPM'] = sig_bpm[:, i_roi]  # BPM signal.
            # Data saving.
            dir_save_data = os.path.join(dir_crt, 'data', name_dataset, 'hr', str(num_attendant)+'_'+algorithm+'.csv')
            df_hr.to_csv(dir_save_data, index=False)

    elif name_dataset == 'UBFC-Phys':
        # List of attendants.
        list_attendant = list(range(1, 57))
        # Condition types.
        list_condition = [1]   # [1, 2, 3]
        for num_attendant in tqdm(list_attendant):
            for condition in list_condition:
                # Parse the RGB signal from the RGB dataframe. Size = [num_frames, num_ROI, rgb_channels(3)].
                dir_sig_rgb = os.path.join(dir_crt, 'data', name_dataset, 'rgb', str(num_attendant) + '_' + str(condition) + '.csv')
                df_rgb = pd.read_csv(dir_sig_rgb)
                # RGB signal initialization.
                sig_rgb = np.zeros([df_rgb['frame'].max(), len(np.unique(df_rgb['ROI'].values)), 3])
                # Loop over all ROIs.
                for i_roi in range(len(Params.list_roi_name)):
                    sig_rgb[:, i_roi, 0] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'R']  # Red channel.
                    sig_rgb[:, i_roi, 1] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'G']  # Green channel.
                    sig_rgb[:, i_roi, 2] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'B']  # Blue channel.
                # RGB video information.
                dir_vid = os.path.join(Params.dir_dataset, 's'+str(num_attendant), 'vid_s'+str(num_attendant)+'_T'+str(condition)+'.avi')
                # Get video fps.
                capture = cv2.VideoCapture(dir_vid)
                Params.fps = capture.get(cv2.CAP_PROP_FPS)
                # RGB signal -> bvp signal & bpm signal.
                sig_bvp, sig_bpm = util_analysis.rppg_hr_pipe(sig_rgb=sig_rgb, method=algorithm, Params=Params)
                # Create the dataframe to save the HR-related data (bvp signal & bpm signal).
                df_hr = pd.DataFrame(columns=['frame', 'time', 'ROI', 'BVP', 'BPM'], index=list(range(len(df_rgb))))
                df_hr.loc[:, ['frame', 'time', 'ROI']] = df_rgb.loc[:, ['frame', 'time', 'ROI']]
                # Loop over all ROIs.
                for i_roi in range(len(Params.list_roi_name)):
                    df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BVP'] = sig_bvp[:, i_roi]  # BVP signal.
                    df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BPM'] = sig_bpm[:, i_roi]  # BPM signal.
                # Data saving.
                dir_save_data = os.path.join(dir_crt, 'data', name_dataset, 'hr', str(num_attendant) + '_' + \
                                             str(condition) + '_' + algorithm + '.csv')
                df_hr.to_csv(dir_save_data, index=False)

    elif name_dataset == 'LGI-PPGI':
        # Name of attendants.
        list_attendant = ['alex', 'angelo', 'cpi', 'david', 'felix', 'harun']
        # Motion types.
        list_motion = ['gym', 'resting', 'talk', 'rotation']
        for attendant in tqdm(list_attendant):
            for motion in list_motion:
                # Parse the RGB signal from the RGB dataframe. Size = [num_frames, num_ROI, rgb_channels(3)].
                dir_sig_rgb = os.path.join(dir_crt, 'data', name_dataset, 'rgb', attendant+'_'+motion+'.csv')
                df_rgb = pd.read_csv(dir_sig_rgb)
                # RGB signal initialization.
                sig_rgb = np.zeros([df_rgb['frame'].max(), len(np.unique(df_rgb['ROI'].values)), 3])
                # Loop over all ROIs.
                for i_roi in range(len(Params.list_roi_name)):
                    sig_rgb[:, i_roi, 0] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'R']   # Red channel.
                    sig_rgb[:, i_roi, 1] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'G']   # Green channel.
                    sig_rgb[:, i_roi, 2] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'B']   # Blue channel.
                # RGB video information.
                dir_vid = os.path.join(Params.dir_dataset, attendant, attendant+'_'+motion, 'cv_camera_sensor_stream_handler.avi')
                # Get video fps.
                capture = cv2.VideoCapture(dir_vid)
                Params.fps = capture.get(cv2.CAP_PROP_FPS)
                # RGB signal -> bvp signal & bpm signal.
                sig_bvp, sig_bpm = util_analysis.rppg_hr_pipe(sig_rgb=sig_rgb, method=algorithm, Params=Params)
                # Create the dataframe to save the HR-related data (bvp signal & bpm signal).
                df_hr = pd.DataFrame(columns=['frame', 'time', 'ROI', 'BVP', 'BPM'], index=list(range(len(df_rgb))))
                df_hr.loc[:, ['frame', 'time', 'ROI']] = df_rgb.loc[:, ['frame', 'time', 'ROI']]
                # Loop over all ROIs.
                for i_roi in range(len(Params.list_roi_name)):
                    df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BVP'] = sig_bvp[:, i_roi]  # BVP signal.
                    df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BPM'] = sig_bpm[:, i_roi]  # BPM signal.
                # Data saving.
                dir_save_data = os.path.join(dir_crt, 'data', name_dataset, 'hr', attendant+'_'+motion+'_'+algorithm+'.csv')
                df_hr.to_csv(dir_save_data, index=False)

    elif name_dataset == 'BUAA-MIHR':
        # Sequnce num of attendants.
        list_attendant = list(range(1, 14))
        # Lux levels.
        list_lux = ['lux 10.0', 'lux 15.8', 'lux 25.1', 'lux 39.8', 'lux 63.1', 'lux 100.0']
        # Attendant names.
        list_name = ['APH', 'GDB', 'HB', 'KLK', 'LW', 'LXR', 'LY', 'LZY', 'LMX', 'MXZ', 'PYC', 'QSH', 'WQT']
        # Loop over all attendants.
        for num_attendant in tqdm(list_attendant):
            # Loop over all illumination levels.
            for lux in list_lux:
                # Parse the RGB signal from the RGB dataframe. Size = [num_frames, num_ROI, rgb_channels(3)].
                dir_sig_rgb = os.path.join(dir_crt, 'data', name_dataset, 'rgb', str(num_attendant).zfill(2)+'_'+lux.replace(' ', '')+'.csv')
                df_rgb = pd.read_csv(dir_sig_rgb, index_col=None)
                # RGB signal initialization.
                sig_rgb = np.zeros([df_rgb['frame'].max(), len(np.unique(df_rgb['ROI'].values)), 3])
                # Loop over all ROIs.
                for i_roi in range(len(Params.list_roi_name)):
                    sig_rgb[:, i_roi, 0] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'R']  # Red channel.
                    sig_rgb[:, i_roi, 1] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'G']  # Green channel.
                    sig_rgb[:, i_roi, 2] = df_rgb.loc[df_rgb['ROI'].values == Params.list_roi_name[i_roi], 'B']  # Blue channel.
                # RGB video information.
                dir_vid = os.path.join(Params.dir_dataset, 'Sub '+str(num_attendant).zfill(2), lux, \
                                       lux.replace(' ', '') + '_' + list_name[num_attendant-1]+'.avi')
                # Get video fps.
                capture = cv2.VideoCapture(dir_vid)
                Params.fps = capture.get(cv2.CAP_PROP_FPS)
                # RGB signal -> bvp signal & bpm signal.
                sig_bvp, sig_bpm = util_analysis.rppg_hr_pipe(sig_rgb=sig_rgb, method=algorithm, Params=Params)
                # Create the dataframe to save the HR-related data (bvp signal & bpm signal).
                df_hr = pd.DataFrame(columns=['frame', 'time', 'ROI', 'BVP', 'BPM'], index=list(range(len(df_rgb))))
                df_hr.loc[:, ['frame', 'time', 'ROI']] = df_rgb.loc[:, ['frame', 'time', 'ROI']]
                # Loop over all ROIs.
                for i_roi in range(len(Params.list_roi_name)):
                    df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BVP'] = sig_bvp[:, i_roi]  # BVP signal.
                    df_hr.loc[df_hr['ROI'].values == Params.list_roi_name[i_roi], 'BPM'] = sig_bpm[:, i_roi]  # BPM signal.
                # Data saving.
                dir_save_data = os.path.join(dir_crt, 'data', name_dataset, 'hr', str(num_attendant).zfill(2) + \
                                             '_' + str(lux).replace(' ', '') + '_' + algorithm+'.csv')
                df_hr.to_csv(dir_save_data, index=False)


if __name__ == "__main__":
    # Available datasets.
    list_dataset = ['UBFC-rPPG', 'UBFC-Phys', 'LGI-PPGI', 'BUAA-MIHR']   # ['UBFC-rPPG', 'UBFC-Phys', 'LGI-PPGI', 'BUAA-MIHR'].
    # Selected rPPG algorithms.
    list_algorithm = ['LGI', 'OMIT', 'CHROM', 'POS']   # ['LGI', 'OMIT', 'CHROM', 'POS'].
    for name_dataset in list_dataset:
        for algorithm in list_algorithm:
            print([name_dataset, algorithm])
            main_rgb2hr(name_dataset=name_dataset, algorithm=algorithm)