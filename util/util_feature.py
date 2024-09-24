"""
utils for feature extraction. included features: ROI size (number of pixels), facial surface orientation.
"""

# Author: Shuo Li
# Date: 2023/07/23

import cv2
import numpy as np
import open3d as o3d


def surface_orientation(loc_landmark, list_roi_num):
    """compute the surface orientation metric.

    Parameters
    ----------
    loc_landmark: detected 3D landmarks. size=[468, 3].
    list_roi_num: the list containing sequence numbers of selected keypoints of different ROIs. size = [num_roi].

    Returns
    -------
    so_mean: mean values of the surface orientation of each ROI. size = [num_roi].
    so_median: median values of the surface orientation of each ROI. size = [num_roi].
    so_std: standard deviation of the surface orientation of each ROI. size = [num_roi].
    """

    # create the point cloud.
    pcd = o3d.open3d.geometry.PointCloud()
    pcd.points = o3d.open3d.utility.Vector3dVector(loc_landmark)
    # normal vector estimation.
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))
    data_normals = np.asarray(pcd.normals)
    # surface orientation.
    data_so = np.arccos((np.matmul(data_normals, [0,0,1]))/(np.linalg.norm(data_normals, axis=1)))
    data_so = (0.5*np.pi - np.abs(0.5*np.pi - data_so))*180/np.pi
    # metric data initialization.
    so_mean = np.zeros(len(list_roi_num))  # mean.
    so_median = np.zeros(len(list_roi_num))  # median.
    so_std = np.zeros(len(list_roi_num))  # std.
    # loop over all ROIs to compute the surface orientation metrics.
    for i_roi in range(len(list_roi_num)):
        data_so_crt = data_so[list_roi_num[i_roi]]
        so_mean[i_roi] = np.mean(data_so_crt)  # mean.
        so_median[i_roi] = np.median(data_so_crt) # median.
        so_std[i_roi] = np.std(data_so_crt)  # std.

    return so_mean, so_median, so_std


def compute_num_pixel(img, loc_landmark, list_roi_num):
    """compute the number of pixels.

    Parameters
    ----------
    img: 2D image. size=[height, width, 3].
    loc_landmark: detected 3D landmarks. size=[468, 3].
    list_roi_num: the list containing sequence numbers of selected keypoints of different ROIs. size = [num_roi].

    Returns
    -------
    num_pixel: number of pixels. size = [num_roi].
    """
    # RGB -> gray.
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # kurtosis metric initialization.
    num_pixel = np.zeros(len(list_roi_num))
    # create masks.
    zeros = np.zeros(img.shape, dtype=np.uint8)
    # ROI-forehead-nose-leftcheek-rightcheek-underlip. Colorization.
    height_img = img.shape[0]
    width_img = img.shape[1]
    # loop over all ROIs to compute the FWHM metric.
    for i_roi in range(len(list_roi_num)):
        # create ROI mask.
        mask_crt = cv2.fillPoly(zeros.copy(), [np.multiply(loc_landmark[list_roi_num[i_roi], :2], [width_img, height_img]).astype(int)], color=(1))
        num_pixel[i_roi] = np.sum(np.sum(mask_crt))

    return num_pixel