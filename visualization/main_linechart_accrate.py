"""
the experiment for examining the effect of ROI selection under different subject's motion types and cognitive tasks.
"""

# Author: Shuo Li
# Date: 2023/09/11

import os
import numpy as np
import matplotlib.pyplot as plt
dir_crt = os.getcwd()

def main_linechart_accrate():
    """linechart of comparing the acceptance rates between different ROIs."""

    list_accratio_motion = [0.492, 0.47, 0.4, 0.388, 0.478, 0.47, 0.434]
    list_accratio_cognitive = [0.704519119, 0.676709154, 0.692931634, 0.681344148, 0.68829664, 0.657010429, 0.682502897]
    plt.plot(list_accratio_motion, marker='o', color=np.array([231, 98, 84])/255)
    plt.axhline(y=0.492, ls=':', color=np.array([231, 98, 84])/255)
    plt.plot(list_accratio_cognitive, marker='o', color=np.array([30, 70, 110])/255)
    plt.axhline(y=0.704519119, ls=':', color=np.array([30, 70, 110])/255)
    plt.ylim([0.35, 0.75])
    plt.show()
    pass
    

if __name__ == "__main__":
    main_linechart_accrate()