"""
extract raw RGB traces from facial videos.
"""

# Author: Shuo Li
# Date: 2023/05/05

import os
import sys
import pyVHR
import pandas as pd
from tqdm import tqdm
dir_crt = os.getcwd()
sys.path.append(os.path.join(dir_crt, 'util'))
import util_pre_analysis


def main_extract_rgb(name_dataset):
    """main function for face detection of videos.
    Parameters
    ----------
    name_dataset: name of the selected dataset.
                  [UBFC-rPPG, UBFC-Phys, LGI-PPGI, BUAA-MIHR].
    
    Returns
    -------

    """
    # get current directory.
    dir_crt = os.getcwd()
    dir_option = os.path.join(dir_crt, 'config', 'options.yaml')
    # parameter class initialization.
    Params = util_pre_analysis.Params(dir_option=dir_option, name_dataset=name_dataset)


    # video -> RGB signal.
    if name_dataset == 'UBFC-rPPG':
        # sequnce num of attendants.
        list_attendant = [1] + list(range(3, 6)) + list(range(8, 19)) + [20] + list(range(22, 27)) + list(range(30, 50))
        # create a log to save num of nan.
        df_nan = pd.DataFrame(columns=['attendant', 'num_nan'])
        for num_attendant in tqdm(list_attendant):
            print([name_dataset, num_attendant])
            # video directory.
            dir_vid = os.path.join(Params.dir_dataset, 'DATASET_2', 'subject'+str(num_attendant), 'vid.avi')
            # video fps.
            Params.fps = pyVHR.extraction.utils.get_fps(dir_vid)
            # RGB signal extraction.
            df_rgb, num_nan = util_pre_analysis.vid_to_sig(dir_vid=dir_vid, Params=Params)
            # save RGB signals.
            dir_save_data = os.path.join(dir_crt, 'data', name_dataset, 'rgb', str(num_attendant)+'.csv')
            df_rgb.to_csv(dir_save_data, index=False)
            # save nan events.
            df_nan.loc[len(df_nan)] = [num_attendant, num_nan] 
            dir_save_nan = os.path.join(dir_crt,'data', name_dataset, 'rgb', 'nan_event.csv')
            df_nan.to_csv(dir_save_nan, index=False)


    elif name_dataset == 'UBFC-Phys':
        # name of attendants.
        list_attendant = list(range(1, 57))
        # condition types.
        list_condition = [1]  # [1, 2, 3]
        # create a log to save num of nan.
        df_nan = pd.DataFrame(columns=['attendant', 'condition', 'num_nan'])
        for num_attendant in tqdm(list_attendant):
            for condition in list_condition:
                print([name_dataset, num_attendant, condition])
                # video directory.
                dir_vid = os.path.join(Params.dir_dataset, 's'+str(num_attendant), 'vid_s'+str(num_attendant)+'_T'+str(condition)+'.avi')
                # video fps.
                Params.fps = pyVHR.extraction.utils.get_fps(dir_vid)
                # RGB signal extraction.
                df_rgb, num_nan = util_pre_analysis.vid_to_sig(dir_vid=dir_vid, Params=Params)
                # save RGB signals.
                dir_save_data = os.path.join(dir_crt, 'data', name_dataset, 'rgb', str(num_attendant)+'_'+str(condition)+'.csv')
                df_rgb.to_csv(dir_save_data, index=False)
                # save nan events.
                df_nan.loc[len(df_nan)] = [num_attendant, condition, num_nan] 
                dir_save_nan = os.path.join(dir_crt,'data', name_dataset, 'rgb', 'nan_event.csv')
                df_nan.to_csv(dir_save_nan, index=False)


    elif name_dataset == 'LGI-PPGI':
        # name of attendants.
        list_attendant = ['alex', 'angelo', 'cpi', 'david', 'felix', 'harun']
        # motion types.
        list_motion = ['talk', 'gym', 'resting', 'rotation']
        # create a log to save num of nan.
        df_nan = pd.DataFrame(columns=['attendant', 'motion', 'num_nan'])
        for attendant in tqdm(list_attendant):
            for motion in list_motion:
                print([name_dataset, attendant, motion])
                # video directory.
                dir_vid = os.path.join(Params.dir_dataset, attendant, attendant+'_'+motion, 'cv_camera_sensor_stream_handler.avi')
                # video fps.
                Params.fps = pyVHR.extraction.utils.get_fps(dir_vid)
                # RGB signal extraction.
                df_rgb, num_nan = util_pre_analysis.vid_to_sig(dir_vid=dir_vid, Params=Params)
                # save RGB signals.
                dir_save_data = os.path.join(dir_crt, 'data', name_dataset, 'rgb', attendant+'_'+motion+'.csv')
                df_rgb.to_csv(dir_save_data, index=False)
                # save nan events.
                df_nan.loc[len(df_nan)] = [attendant, motion, num_nan] 
                dir_save_nan = os.path.join(dir_crt,'data', name_dataset, 'rgb', 'nan_event.csv')
                df_nan.to_csv(dir_save_nan, index=False)
 

    elif name_dataset == 'BUAA-MIHR':
        # sequnce num of attendants.
        list_attendant = list(range(1, 14))
        # illumination levels.
        list_lux = ['lux 10.0', 'lux 15.8', 'lux 25.1', 'lux 39.8', 'lux 63.1', 'lux 100.0']
        # attendant names.
        list_name = ['APH', 'GDB', 'HB', 'KLK', 'LW', 'LXR', 'LY', 'LZY', 'LMX', 'MXZ', 'PYC', 'QSH', 'WQT']
        # create a log to save num of nan.
        df_nan = pd.DataFrame(columns=['attendant', 'lux', 'num_nan'])
        for num_attendant in tqdm(list_attendant):
            for lux in list_lux:
                name = list_name[num_attendant-1]
                print([num_attendant, name, lux])
                # video directory.
                dir_vid = os.path.join(Params.dir_dataset, 'Sub '+str(num_attendant).zfill(2), lux, lux.replace(' ', '')+'_'+name+'.avi')
                # video fps.
                Params.fps = pyVHR.extraction.utils.get_fps(dir_vid)
                # RGB signal extraction.
                df_rgb, num_nan = util_pre_analysis.vid_to_sig(dir_vid=dir_vid, Params=Params)
                # save RGB signals.
                dir_save_data = os.path.join(dir_crt, 'data', name_dataset, 'rgb', str(num_attendant).zfill(2)+'_'+str(lux).replace(' ', '')+'.csv')
                df_rgb.to_csv(dir_save_data, index=False)
                # save nan events.
                df_nan.loc[len(df_nan)] = [num_attendant, lux, num_nan] 
                dir_save_nan = os.path.join(dir_crt,'data', name_dataset, 'rgb', 'nan_event.csv')
                df_nan.to_csv(dir_save_nan, index=False)


if __name__ == "__main__":
    # available datasets.
    list_dataset = ['BUAA-MIHR']
    # extract RGB signal.
    for name_dataset in list_dataset:
        main_extract_rgb(name_dataset=name_dataset)