"""
extract relevant features from videos. included features: ROI size (number of pixels), facial surface orientation.
"""

# Author: Shuo Li
# Date: 2023/05/05

import os
import sys
import pyVHR
from tqdm import tqdm
dir_crt = os.getcwd()
sys.path.append(os.path.join(dir_crt, 'util'))
import util_pre_analysis



def main_extract_feature(name_dataset):
    """main function for feature extraction of facial videos.
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
    # video -> features.
    if name_dataset == 'UBFC-rPPG':
        # sequnce num of attendants.
        list_attendant = [1] + list(range(3, 6)) + list(range(8, 19)) + [20] + list(range(22, 27)) + list(range(30, 50))
        for num_attendant in tqdm(list_attendant):
            print([num_attendant])
            # video directory.
            dir_vid = os.path.join(Params.dir_dataset, 'DATASET_2', 'subject'+str(num_attendant), 'vid.avi')
            Params.fps = pyVHR.extraction.utils.get_fps(dir_vid)
            # feature calculation.
            df_feature = util_pre_analysis.vid_to_feature(dir_vid=dir_vid, Params=Params)
            # save features.
            dir_save_data = os.path.join(dir_crt, 'data', name_dataset, 'feature', str(num_attendant)+'.csv')
            df_feature.to_csv(dir_save_data, index=False)

    elif name_dataset == 'UBFC-Phys':
        # list of attendants.
        list_attendant = list(range(1, 29)) + list(range(30, 55)) + [56]
        # list of conditions.
        list_conition = [1]   # [1, 2, 3]
        # loop over all attendants.
        for num_attendant in tqdm(list_attendant):

            # loop over all conditions.
            for num_condition in list_conition:
                print([num_attendant, num_condition])
                # video directory.
                dir_vid = os.path.join(Params.dir_dataset, 's'+str(num_attendant), 'vid_s'+str(num_attendant)+'_T'+str(num_condition)+'.avi')
                Params.fps = pyVHR.extraction.utils.get_fps(dir_vid)
                # feature calculation.
                df_feature = util_pre_analysis.vid_to_feature(dir_vid=dir_vid, Params=Params)
                # save features.
                dir_save_data = os.path.join(dir_crt, 'data', name_dataset, 'feature', str(num_attendant)+'.csv')
                df_feature.to_csv(dir_save_data, index=False)


if __name__ == "__main__":
    main_extract_feature(name_dataset='UBFC-rPPG')   # ['UBFC-rPPG', 'UBFC-Phys']