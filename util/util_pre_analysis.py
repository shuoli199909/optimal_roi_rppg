"""
utils for the preliminary analysis of the optimal ROI selection under different conditions.
"""

# Author: Shuo Li
# Date: 2023/05/30

import os
import yaml
import pyVHR
import cv2
import util_feature
import pyVHR.BVP.methods as methods
import mediapipe as mp
import pandas as pd
import numpy as np
from xml.dom import minidom
from sklearn import metrics
from tqdm import tqdm
from dtaidistance import dtw
from scipy.signal import resample
from pyVHR.BVP.BVP import RGB_sig_to_BVP
from pyVHR.BPM.BPM import BVP_to_BPM
from pyVHR.utils import errors
from pyVHR.BPM.utils import Welch
from pyVHR.BVP.filters import apply_filter, BPfilter


class Params():
    """load the pre-defined parameters for preliminary analysis from a YAML file. 
       create a class.
    """

    def __init__(self, dir_option, name_dataset) -> None:
        """parameter calss initialization.

        Parameters
        ----------
        dir_option: directory of the YAML file.
        name_dataset: name of datasets. ['UBFC-rPPG', 'UBFC-Phys', 'LGI-PPGI', 'BUAA-MIHR']

        Returns
        -------

        """

        # options.
        self.options = yaml.safe_load(open(dir_option))
        # url.
        self.url = self.options[name_dataset]['url']
        # dataset directory.
        self.dir_dataset = self.options[name_dataset]['dir_dataset']
        # face detection parameters.
        self.max_num_faces = self.options[name_dataset]['max_num_faces']  # number of target faces.
        self.minDetectionCon = self.options[name_dataset]['minDetectionCon']  # minimal detection confidence.
        self.minTrackingCon = self.options[name_dataset]['minTrackingCon']  # minimal tracking confidence.
        # the list containing sequence numbers of selected keypoints of different ROIs. size = [num_roi].
        self.list_roi_num = self.options[name_dataset]['list_roi_num']
        # the list containing names of different ROIs. size = [num_roi].
        self.list_roi_name = self.options[name_dataset]['list_roi_name']
        # RGB signal -> windowed signal.
        self.len_window = self.options[name_dataset]['len_window']  # window length in seconds.
        self.stride_window = self.options[name_dataset]['stride_window']  # window stride in seconds.
        self.fps = self.options[name_dataset]['fps']  # frames per second.


class GroundTruth():
    """load the groundtruth data. (time, PPG waveform, PPG HR). 
       create a class.
    """

    def __init__(self, dir_dataset, name_dataset) -> None:
        """groundtruth class initialization.

        Parameters
        ----------
        dir_option: directory of the YAML file.
        name_dataset: name of datasets. ['UBFC-rPPG', 'UBFC-Phys', 'LGI-PPGI', 'BUAA-MIHR']

        Returns
        -------

        """

        # directory of the dataset.
        self.dir_dataset = dir_dataset
        # dataset name. ['UBFC-rPPG', 'UBFC-Phys', 'LGI-PPGI', 'BUAA-MIHR'].
        self.name_dataset = name_dataset
    
    def get_GT(self, specification, num_frame_interp, slice):
        """get the ground truth data.

        Parameters
        ----------
        specification: specificy the dataset.
                       UBFC-rPPG: [condition, num_attendant]
                                  'simple' ~ [5-8, 10-12].
                                  'realistic' ~ [1, 3-5, 8-18, 20, 22-26, 30-49].
                                  example: ['simple', 6]
                       UBFC-Phys: [num_attendant, num_task].
                                  num_attendant: [1-56].
                                  num_task: [1, 2, 3] - [rest, speech, arithmetic].
                                  example: [2, 2].
                       LGI-PPGI: [name_attendant, motion].
                                 name_attendant: ['alex', 'angelo', 'cpi', 'david', 'felix', 'harun'].
                                 motion: ['gym', 'resting', 'rotation', 'talk'].
                                 example: ['alex', 'gym'].
                       BUAA-MIHR: [num_attendant, lux, name].
                                  num_attendant: [1-14].
                                  lux: ['lux 1.0', 'lux 1.6', 'lux 2.5', 'lux 4.0', 'lux 6.3', 'lux 10.0', 
                                        'lux 15.8', 'lux 25.1', 'lux 39.8', 'lux 63.1', 'lux 100.0'].
                                  name: ['APH', 'GDB', 'HB', 'KLK', 'LW', 'LXR', 'LY', 'LZY', 'LMX', 'MXZ', 'PYC', 'QSH', 'WQT'].
        num_frame_interp: total number of frames after interpolation.
        slice: select a time window of the signal. [start time, end time]. the time is normalized into [0, 1].

        Returns
        -------
        gtTime: ground truth time in numpy array. size = [num_frames].
        gtTrace: ground truth PPG waveform data in numpy array. size = [num_frames].
        gtHR: ground truth HR data in numpy array. size = [num_frames].
        """

        if self.name_dataset == 'UBFC-rPPG':  # UBFC-rPPG dataset.
            
            if specification[0] == 'simple':  # simple. 
                dir_crt = os.path.join(self.dir_dataset, 'DATASET_1', str(specification[1])+'-gt', 'gtdump.xmp')
                df_GT = pd.read_csv(dir_crt, header=None)
                gtTime = df_GT[0].values/1000
                gtTrace = df_GT[3].values
                gtHR = df_GT[1].values
                
            elif specification[0] == 'realistic':  # realistic.
                dir_crt = os.path.join(self.dir_dataset, 'DATASET_2', 'subject'+str(specification[1]), 'ground_truth.txt')
                npy_GT = np.loadtxt(dir_crt)
                gtTime = npy_GT[2, :]
                gtTrace = npy_GT[0, :]
                gtHR = npy_GT[1, :]


        elif self.name_dataset == 'UBFC-Phys':  # UBFC-Phys dataset.
            # groundtruth BVP trace.
            dir_bvp = os.path.join(self.dir_dataset, 's'+str(specification[0]), 'bvp_s'+str(specification[0])+'_T'+str(specification[1])+'.csv')
            gtTrace = np.loadtxt(dir_bvp)
            # groundtruth video.
            dir_vid = os.path.join(self.dir_dataset, 's'+str(specification[0]), 'vid_s'+str(specification[0])+'_T'+str(specification[1])+'.avi')
            capture = cv2.VideoCapture(dir_vid)
            fps = capture.get(cv2.CAP_PROP_FPS)   # frame rate.
            num_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))   # number of frames.
            duration = num_frame/fps  # video duration. (sec).
            # groundtruth time.
            gtTime = np.linspace(start=0, stop=duration, num=num_frame)
            # groundtruth hr.
            dir_bpm = os.path.join(self.dir_dataset, 's'+str(specification[0]), 'bpm_s'+str(specification[0])+'_T'+str(specification[1])+'.csv')
            gtHR = np.loadtxt(dir_bpm)


        elif self.name_dataset == 'LGI-PPGI':  # LGI-PPGI dataset.
            dir_vid = os.path.join(self.dir_dataset, str(specification[0]), specification[0]+'_'+specification[1], 'cv_camera_sensor_stream_handler.avi')
            dir_xml = os.path.join(self.dir_dataset, specification[0], specification[0]+'_'+specification[1], 'cms50_stream_handler.xml')
            dom = minidom.parse(dir_xml)
            # ground truth heart rate.
            value_HR = dom.getElementsByTagName('value1')
            # ground truth trace.
            value_Trace = dom.getElementsByTagName('value2')
            gtHR = []
            gtTrace = []
            for i in range(len(value_HR)):
                HR_tmp = value_HR[i].firstChild.data
                if '\n' not in HR_tmp:  # exclude invalid data.
                    gtHR.append(int(HR_tmp))
                Trace_tmp = value_Trace[i].firstChild.data
                if '\n' not in Trace_tmp:  # exclude invalid data.
                    gtTrace.append(int(Trace_tmp))
            # ground truth time.
            capture = cv2.VideoCapture(dir_vid)
            fps = capture.get(cv2.CAP_PROP_FPS)   # frame rate.
            num_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))   # number of frames.
            duration = num_frame/fps   # video duration. (sec).
            gtTime = np.linspace(start=0, stop=duration, num=num_frame)
            # list -> numpy array.
            gtHR = np.array(gtHR)
            gtTrace = np.array(gtTrace)

        
        elif self.name_dataset == 'BUAA-MIHR':  # BUAA-MIHR dataset.
            dir_crt = os.path.join(self.dir_dataset, 'Sub '+str(specification[0]).zfill(2), specification[1])
            # PPG trace wave.
            gtTrace = np.loadtxt(os.path.join(dir_crt, specification[1].replace(' ', '')+'_'+specification[2]+'_wave.csv'))
            # time stamp.
            # RGB video information.
            dir_vid = os.path.join(dir_crt, specification[1].replace(' ', '')+'_'+specification[2]+'.avi')
            # get video fps.
            capture = cv2.VideoCapture(dir_vid)
            fps = capture.get(cv2.CAP_PROP_FPS)
            num_frame = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = num_frame/fps
            gtTime = np.linspace(start=0, stop=duration, num=int(num_frame))
            # HR data.
            df_HR = pd.read_csv(os.path.join(dir_crt, specification[1].replace(' ', '')+'_'+specification[2]+'.csv'))
            gtHR = df_HR['PULSE'].values
            # HR signal resampling.
        
        # resampling according to gtTime.
        gtTrace = resample(x=gtTrace, num=num_frame_interp)
        gtHR = resample(x=gtHR, num=num_frame_interp)
        # time windowing.
        frame_start = round(slice[0] * len(gtTime))
        frame_end = round(slice[1] * len(gtTime))
        gtTime = gtTime[frame_start:frame_end]
        gtTrace = gtTrace[frame_start:frame_end]
        gtTrace = (gtTrace - np.min(gtTrace))/(np.max(gtTrace) - np.min(gtTrace))  # normalize into [0, 1].
        gtHR = gtHR[frame_start:frame_end]

        return gtTime, gtTrace, gtHR


class FaceDetector():
    """a class for face detection, segmentation and RGB signal extraction."""

    def __init__(self, Params):
        """class initialization.
        Parameters
        ----------
        Params: a class containing the pre-defined parameters.

        Returns
        -------

        """

        # Confidence
        self.minDetectionCon = Params.minDetectionCon  # minimal detection confidence.
        self.minTrackingCon = Params.minTrackingCon  # minimal tracking confidence.
        # Mediapipe utils
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)  # face detection.
        self.mpDraw = mp.solutions.drawing_utils  # drawing utils.
        self.faceMesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=Params.max_num_faces,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackingCon
            )  # Face mesh.
        # ROI params.
        # the list containing sequence numbers of selected keypoints of different ROIs. size = [num_roi].
        self.list_roi_num = np.array(Params.list_roi_num, dtype=object)
        # the list containing names of different ROIs. size = [num_roi].
        self.list_roi_name = np.array(Params.list_roi_name, dtype=object)


    def extract_landmark(self, img):
        """extract 2D keypoint locations.
        Parameters
        ----------
        img: the input image of the current frame. channel = [B, G, R].

        Returns
        -------
        loc_landmark: detected normalized 3D landmarks. size=[468, 3].
        """
        
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB.
        results = self.faceMesh.process(img_RGB)  # apply face mesh.
        # draw landmarks on the image.
        if results.multi_face_landmarks:
            # if the face is detected.
            # loop over all detected faces.
            # in this experiment, we only detect one face in one video.
            for face_landmark in results.multi_face_landmarks:
                # decompose the 3D face landmarks without resizing into the image size.
                loc_landmark = np.zeros([len(face_landmark.landmark), 3], dtype=np.float32)  # coordinates of 3D landmarks.
                for i in range(len(face_landmark.landmark)):
                    loc_landmark[i, 0] = face_landmark.landmark[i].x
                    loc_landmark[i, 1] = face_landmark.landmark[i].y
                    loc_landmark[i, 2] = face_landmark.landmark[i].z
        else:
            # if no face is detected.
            loc_landmark = np.nan
        
        return loc_landmark


    def extract_RGB(self, img, loc_landmark):
        """extract RGB signals from the given image and ROI.
        Parameters
        ----------
        img: 2D image. default in BGR style. size=[height, width, 3]
        loc_landmark: detected normalized (0-1) 3D landmarks. size=[468, 3].

        Returns
        -------
        sig_rgb: RGB signal of the current frame as a numpy array. size=[num_roi, 3].
        """

        if (np.isnan(loc_landmark)).any() == True:
            # if no face is detected.
            sig_rgb = np.nan
        else:
            # if the face is detected.
            # BGR -> RGB.
            img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # rescale the input landmarks location.
            height_img = img.shape[0]
            width_img = img.shape[1]
            loc_landmark[:, 0] = loc_landmark[:, 0] * width_img
            loc_landmark[:, 1] = loc_landmark[:, 1] * height_img
            # RGB signal initialization.
            sig_rgb = np.zeros(shape=[self.list_roi_num.shape[0], 3])
            # loop over all ROIs.
            zeros = np.zeros(img.shape, dtype=np.uint8)
            for i_roi in range(0, self.list_roi_num.shape[0]):
                # create the current ROI mask.
                roi_name = self.list_roi_name[i_roi]
                mask = cv2.fillPoly(zeros.copy(), [loc_landmark[self.list_roi_num[self.list_roi_name==roi_name][0], :2].astype(int)], color=(1, 1, 1))
                # only compute on a specific ROI.
                img_masked = np.multiply(img_RGB, mask)
                # compute the RGB signal.
                sig_rgb[i_roi, :] = 3*img_masked.sum(0).sum(0)/(mask.sum())

        return sig_rgb


    def extract_feature(self, img, loc_landmark):
        """extract features for deeper understanding of rPPG techniques.
        Parameters
        ----------
        img: 2D image. default in BGR style. size=[height, width, 3].
        loc_landmark: detected 3D landmarks. size=[468, 3].

        Returns
        -------
        mean_so: mean keypoint surface orientations. size = [num_roi].
        median_so: median keypoint surface orientations. size = [num_roi].
        std_so: standard deviation of keypoint surface orientations. size = [num_roi].
        num_pixel: number of pixels. size = [num_roi].
        """

        # create a dataframe.
        df_feature = pd.DataFrame(columns=['ROI', 'mean_so', 'median_so', 'std_so', 'num_pixel'], index=list(range(len(self.list_roi_name))))
        # BGR -> RGB.
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # surface orientation metric.
        mean_so, median_so, std_so = util_feature.surface_orientation(loc_landmark=loc_landmark, list_roi_num=self.list_roi_num)
        # number of pixels.
        num_pixel = util_feature.compute_num_pixel(img=img_RGB, loc_landmark=loc_landmark, list_roi_num=self.list_roi_num)
        # save data.
        df_feature.loc[:, 'ROI'] = self.list_roi_name
        df_feature.loc[:, 'mean_so'] = mean_so
        df_feature.loc[:, 'median_so'] = median_so
        df_feature.loc[:, 'std_so'] = std_so
        df_feature.loc[:, 'num_pixel'] = num_pixel

        return df_feature


    def faceMeshDraw(self, img, roi_name):
        """draw a face mesh annotations on the input image.
        Parameters
        ----------
        img: the input image of the current frame.
        roi_name: name of the roi. the name should be in the name list.

        Returns
        -------
        img_draw: the output image after drawing the ROI of the current frame. 
        """
        
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB.
        results = self.faceMesh.process(img_RGB)  # apply face mesh.
        mp_face_mesh = mp.solutions.face_mesh_connections
        # draw landmarks on the image.
        if results.multi_face_landmarks:
            # loop over all detected faces.
            # in this experiment, we only detect one face in one video.
            for face_landmark in results.multi_face_landmarks:
                # landmark points.
                self.mpDraw.draw_landmarks(
                    image=img,
                    landmark_list=face_landmark,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.mpDraw.DrawingSpec(thickness=1, circle_radius=1),
                    connection_drawing_spec=self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)
                )
                # decompose the 3D face landmarks.
                height_img = img.shape[0]
                width_img = img.shape[1]
                loc_landmark = np.zeros([len(face_landmark.landmark), 2], dtype=np.int32)  # coordinates of 2D landmarks.
                for i in range(len(face_landmark.landmark)):
                    loc_landmark[i, 0] = face_landmark.landmark[i].x * width_img
                    loc_landmark[i, 1] = face_landmark.landmark[i].y * height_img
                # create a zero vector for mask construction.
                zeros = np.zeros(img.shape, dtype=np.uint8)
                # ROI-forehead-nose-leftcheek-rightcheek-underlip. Colorization.
                mask = cv2.fillPoly(zeros.copy(), [loc_landmark[self.list_roi_num[self.list_roi_name==roi_name][0], :]], color=(1, 1, 1))
                img_draw = img + mask * 50
            
        return img_draw


def vid_to_sig(dir_vid, Params):
    """transform the input video into RGB signals. 
       return the signals as pandas dataframe.

    Parameters
    ----------
    dir_vid: directory of the input video.
    Params: a class containing the pre-defined parameters for the preliminary analysis.

    Returns
    -------
    df_rgb: dataframe containing the RGB signal of the input video.
    num_nan: number of nan values of the extracted RGB signal.
    """

    # input video
    video_crt = cv2.VideoCapture(dir_vid)
    # create the face detection object.
    Detector_crt = FaceDetector(Params=Params)
    # create the dataframe containing the RGB signals and other necessary data.
    df_rgb = pd.DataFrame(columns=['frame', 'time', 'ROI', 'R', 'G', 'B'])
    # start processing each frame.
    num_frame = 0
    while(video_crt.isOpened()):
        ret, img_frame = video_crt.read()
        if (ret == False) or (cv2.waitKey(1) & 0xFF == ord('q')):
            # terminate in the end.
            break
        # detect facial landmark keypoints. the locations are normalized into [0, 1].
        loc_landmark = Detector_crt.extract_landmark(img=img_frame)  # size = [468, 3]
        # extract RGB signal.
        sig_rgb = Detector_crt.extract_RGB(img=img_frame, loc_landmark=loc_landmark)  # size = [num_roi, 3].
        # loop over all ROIs and save the RGB data.
        df_rgb_tmp = pd.DataFrame(columns=['frame', 'time', 'ROI', 'R', 'G', 'B'], index=list(range(0, len(Params.list_roi_name))))
        for i_roi in range(len(Params.list_roi_name)):
            # ROI name.
            df_rgb_tmp.loc[i_roi, 'ROI'] = Params.list_roi_name[i_roi]
            if (np.isnan(sig_rgb)).any() == True:
                # if no face is detected.
                df_rgb_tmp.loc[i_roi, ['R', 'G', 'B']] = np.nan
            else:
                # if the face is detected.
                # RGB channels.
                df_rgb_tmp.loc[i_roi, ['R', 'G', 'B']] = sig_rgb[i_roi, :]
        # sequence number of frame.
        num_frame = num_frame + 1
        df_rgb_tmp.loc[:, 'frame'] = num_frame
        # time of the current frame.
        df_rgb_tmp.loc[:, 'time'] = num_frame * Params.fps
        # attach to the main dataframe.
        df_rgb = df_rgb.append(df_rgb_tmp)
    
    # dataframe reindex.
    df_rgb = df_rgb.reset_index(drop=True)
    # for frames with nan values, use time interpolation. 
    num_nan = df_rgb.isnull().sum().sum()
    for roi_name in Params.list_roi_name:
        df_rgb.loc[df_rgb['ROI'].values==roi_name, :] = df_rgb.loc[df_rgb['ROI'].values==roi_name, :].interpolate(method='time')
        df_rgb.loc[df_rgb['ROI'].values==roi_name, :] = df_rgb.loc[df_rgb['ROI'].values==roi_name, :].fillna(method='ffill')
        df_rgb.loc[df_rgb['ROI'].values==roi_name, :] = df_rgb.loc[df_rgb['ROI'].values==roi_name, :].fillna(method='bfill')

    return df_rgb, num_nan


def vid_to_feature(dir_vid, Params):
    """extract features from the input video. 
       store the features in a pandas dataframe.

    Parameters
    ----------
    dir_vid: directory of the input video.
    Params: a class containing the pre-defined parameters for the preliminary analysis.

    Returns
    -------
    df_feature: dataframe containing the features of the input video.
    """

    # input video
    video_crt = cv2.VideoCapture(dir_vid)
    # create the face detection object.
    Detector_crt = FaceDetector(Params=Params)
    # create the dataframe containing the RGB signals and other necessary data.
    df_feature = pd.DataFrame(columns=['frame', 'time', 'ROI', 'mean_so', 'median_so', 'std_so', 'num_pixel'])
    # start processing each frame.
    num_frame = 0
    while((video_crt.isOpened()) & (num_frame < 20)):  # for efficient computing, only selected a sequence as the input.
        ret, img_frame = video_crt.read()
        if (ret == False) or (cv2.waitKey(1) & 0xFF == ord('q')):
            # terminate in the end.
            break
        # detect facial landmark keypoints. the locations are normalized into [0, 1].
        loc_landmark = Detector_crt.extract_landmark(img=img_frame)  # size = [468, 3]
        # extract RGB signal.
        df_feature_crt = Detector_crt.extract_feature(img=img_frame, loc_landmark=loc_landmark)
        # sequence number of frame.
        num_frame = num_frame + 1
        df_feature_crt.loc[:, 'frame'] = num_frame
        # time of the current frame.
        df_feature_crt.loc[:, 'time'] = num_frame * Params.fps
        # attach to the main dataframe.
        df_feature = df_feature.append(df_feature_crt)
    
    # dataframe reindex.
    df_feature = df_feature.reset_index(drop=True)
    # for frames with nan values, use time interpolation. 
    for roi_name in Params.list_roi_name:
        df_feature.loc[df_feature['ROI'].values==roi_name, :] = df_feature.loc[df_feature['ROI'].values==roi_name, :].interpolate(method='time')
        df_feature.loc[df_feature['ROI'].values==roi_name, :] = df_feature.loc[df_feature['ROI'].values==roi_name, :].fillna(method='ffill')
        df_feature.loc[df_feature['ROI'].values==roi_name, :] = df_feature.loc[df_feature['ROI'].values==roi_name, :].fillna(method='bfill')

    return df_feature


def sig_to_windowed(sig_rgb, Params):
    """transform the original RGB signals into windowed RGB signals.

    Parameters
    ----------
    sig_rgb: the extracted RGB signal of different ROIs. size: [num_frames, num_ROI, rgb_channels].
    Params: a class containing the pre-defined parameters for the preliminary analysis.

    Returns
    -------
    sig_rgb_win: the windowed rgb signals. size: [num_estimators, rgb_channels, window_frames].
    timesES: an array of times in seconds.
    """

    # parameter parsing.
    len_window = Params.len_window  # window length in seconds.
    stride_window = Params.stride_window  # window overlap in seconds.
    fps = Params.fps  # frames per second.
    # signal windowing.
    sig_rgb_win, timesES = pyVHR.extraction.utils.sig_windowing(sig_rgb , len_window , stride_window , fps)

    return sig_rgb_win, timesES


def sig_windowed_to_bvp(sig_rgb_win, method, Params):
    """transform the windowed RGB signals into blood volume pulse (BVP) signals.

    Parameters
    ----------
    sig_rgb_win: the windowed rgb signals. size: [num_estimators, rgb_channels, window_frames].
    method: selected rPPG algorithm. ['CHROM', 'GREEN', 'ICA', 'LGI', 'OMIT', 'PBV', 'POS', 'OMIT'].
    Params: pre-defined parameter structure.

    Returns
    -------
    sig_bvp_win: the windowed bvp(Blood Volume Pulse) signal.
    """

    # selected rPPG algorithms. windowed signal -> bvp signal.
    if method == 'CHROM':
        sig_bvp_win = RGB_sig_to_BVP(windowed_sig=sig_rgb_win, fps=Params.fps, device_type='cuda', method=methods.cupy_CHROM)
    elif method == 'GREEN':
        sig_bvp_win = RGB_sig_to_BVP(windowed_sig=sig_rgb_win, fps=Params.fps, device_type='cpu', method=methods.cpu_GREEN)
    elif method == 'ICA':
        sig_bvp_win = RGB_sig_to_BVP(windowed_sig=sig_rgb_win, fps=Params.fps, device_type='cpu', method=methods.cpu_ICA, params={'component': 'second_comp'})
    elif method == 'LGI':
        sig_bvp_win = RGB_sig_to_BVP(windowed_sig=sig_rgb_win, fps=Params.fps, device_type='cpu', method=methods.cpu_LGI)
    elif method == 'OMIT':
        sig_bvp_win = RGB_sig_to_BVP(windowed_sig=sig_rgb_win, fps=Params.fps, device_type='cpu', method=methods.cpu_OMIT)
    elif method == 'PBV':
        sig_bvp_win = RGB_sig_to_BVP(windowed_sig=sig_rgb_win, fps=Params.fps, device_type='cpu', method=methods.cpu_PBV)
    elif method == 'PCA':
        sig_bvp_win = RGB_sig_to_BVP(windowed_sig=sig_rgb_win, fps=Params.fps, device_type='cpu', method=methods.cpu_PCA, params={'component': 'second_comp'})
    elif method == 'POS':
        sig_bvp_win = RGB_sig_to_BVP(windowed_sig=sig_rgb_win, fps=Params.fps, device_type='cuda', method=methods.cupy_POS, params={'fps':Params.fps})
    
    return sig_bvp_win


def rppg_hr_pipe(sig_rgb, method, Params):
    """the complete pipeline of transforming raw RGB traces into BVP & HR signals.

    Parameters
    ----------
    sig_rgb: the extracted RGB signal of different ROIs. size: [num_frames, num_ROI, rgb_channels].
    method: selected rPPG algorithm. ['CHROM', 'GREEN', 'ICA', 'LGI', 'OMIT', 'PBV', 'POS', 'OMIT'].
    Params: a class containing the pre-defined parameters for the preliminary analysis.

    Returns
    -------
    sig_bvp: blood volume pulse (BVP) signal of different ROI without windowing. size=[num_frames, num_ROI].
    sig_bpm: beats per minute (BPM) signal of different ROI. size=[num_frames, num_ROI].
    """

    # RGB signal -> windowed RGB signal.
    sig_rgb_win, timeES = sig_to_windowed(sig_rgb=sig_rgb, Params=Params)
    # windowed RGB signal -> windowed raw bvp signal.
    sig_bvp_win = sig_windowed_to_bvp(sig_rgb_win=sig_rgb_win, method=method, Params=Params)
    # windowed raw bvp signal -> windowed filtered bvp signal.
    sig_bvp_win_filtered = apply_filter(sig_bvp_win, BPfilter, params={'order':6, 'minHz':0.65, 'maxHz':4.0, 'fps':Params.fps})
    # fill nan values.
    for i_window in range(len(sig_bvp_win_filtered)):
        is_nan = np.any(np.isnan(sig_bvp_win_filtered[i_window]))
        if is_nan == False:
            continue
        elif i_window == 0:
            sig_bvp_win_filtered[i_window] = np.ones(28, np.shape(sig_bvp_win_filtered[i_window])[1])
        else:
            sig_bvp_win_filtered[i_window] = sig_bvp_win_filtered[i_window-1]
    # de-windowing bvp signal.
    for i in range(len(sig_bvp_win_filtered)):
        if i == 0:
            sig_bvp = (sig_bvp_win_filtered[i])[:, :round(Params.fps*Params.stride_window)]
        else:
            sig_bvp = np.concatenate((sig_bvp, (sig_bvp_win_filtered[i])[:, :round(Params.fps*Params.stride_window)]), axis=1)
    sig_bvp = np.concatenate((sig_bvp, (sig_bvp_win_filtered[i])[:, round(Params.fps*Params.stride_window):]), axis=1)
    # windowed filtered bvp signal -> bpm(Beats Per Minute) signal.
    multi_sig_bpm = BVP_to_BPM(bvps=sig_bvp_win_filtered, fps=Params.fps, minHz=0.65, maxHz=4.0)
    # remove nan values.
    for i in range(len(multi_sig_bpm)):
        if len(multi_sig_bpm[i]) != len(Params.list_roi_name):
            multi_sig_bpm[i] = multi_sig_bpm[i-1]
    # list -> numpy array.
    sig_bpm = np.array(multi_sig_bpm)
    # reshaping.
    sig_bvp_old = np.transpose(sig_bvp, [1, 0])
    sig_bpm_old = np.transpose(sig_bpm, [0, 1])
    # resampling.
    sig_bvp = np.zeros_like(sig_rgb[:, :, 0])
    sig_bpm = np.zeros_like(sig_rgb[:, :, 0])
    # across different ROIs.
    for i_roi in range(sig_bvp_old.shape[1]):
        # BVP signal.
        sig_bvp[:, i_roi] = np.interp(x=np.linspace(0, len(sig_bvp), len(sig_bvp)), 
                                      xp=np.linspace(0, len(sig_bvp), len(sig_bvp_old)), 
                                      fp=sig_bvp_old[:, i_roi])
        # HR signal.
        sig_bpm[:, i_roi] = np.interp(x=np.linspace(0, len(sig_bpm), len(sig_bpm)), 
                                      xp=np.linspace(0, len(sig_bpm), len(sig_bpm_old)), 
                                      fp=sig_bpm_old[:, i_roi])
    

    return sig_bvp, sig_bpm


def eval_pipe(sig_bvp, sig_bpm, gtTime, gtTrace, gtHR, Params):
    """the complete pipeline for rPPG algorithm evaluation.
       this evaluation is based on BVP & BPM signals.
       the selected metrics are: [PCC, CCC, RMSE, MAE, DTW].

    Parameters
    ----------
    sig_bvp: BVP signal of different ROIs after de-windowing. size=[num_frames, num_ROI].
    sig_bpm: BPM signal of different ROI. size=[num_frames, num_ROI].
    gtTime: ground truth time in numpy array. size = [num_frames].
    gtTrace: ground truth PPG waveform data in numpy array. size = [num_frames].
    gtHR: ground truth HR data in numpy array. size = [num_frames].
    Params: a class containing the pre-defined parameters for the preliminary analysis.

    Returns
    -------
    list_DTW: DTW metric. size = [num_roi].
    list_PCC = Pearson's Correlation Coefficient (PCC). size = [num_roi].
    list_CCC = Concordance Correlation Coefficient (CCC). size = [num_roi].
    list_RMSE = Root Mean Square Error (RMSE). size = [num_roi].
    list_MAE = Mean Absolute Error (MAE). size = [num_roi].
    """

    # metrics initialization of different ROIs.
    list_DTW = np.zeros(len(Params.list_roi_name))
    list_PCC = np.zeros(len(Params.list_roi_name))
    list_CCC = np.zeros(len(Params.list_roi_name))
    list_RMSE = np.zeros(len(Params.list_roi_name))
    list_MAE = np.zeros(len(Params.list_roi_name))
    list_SNR = np.zeros(len(Params.list_roi_name))
    # process different ROI respectively.
    for i in tqdm(range(len(sig_bpm[0, :]))):
        # BVP signal of each ROI.
        sig_bvp_crt = sig_bvp[:, i]
        # BPM signal of each ROI.
        sig_bpm_crt = sig_bpm[:, i]
        # windowing. this process helps stabilize the evaluation results. len_win = 10s.
        for slice_start in np.arange(0, len(gtTrace), round(len(gtTrace)*10/np.max(gtTime))):
            if slice_start+round(len(gtTrace)*10/np.max(gtTime)) >= len(gtTrace):
                gtTrace_crt = gtTrace[len(gtTrace)-round(len(gtTrace)*10/np.max(gtTime)):]
                gtHR_crt = gtHR[len(gtTrace)-round(len(gtTrace)*10/np.max(gtTime)):]
                sig_rppg_crt_slice = sig_bvp_crt[len(gtTrace)-round(len(gtTrace)*10/np.max(gtTime)):]
                sig_bpm_crt_slice = sig_bpm_crt[len(gtTrace)-round(len(gtTrace)*10/np.max(gtTime)):]
            else:
                gtTrace_crt = gtTrace[slice_start:slice_start+round(len(gtTrace)*10/np.max(gtTime))]
                gtHR_crt = gtHR[slice_start:slice_start+round(len(gtTrace)*10/np.max(gtTime))]
                sig_rppg_crt_slice = sig_bvp_crt[slice_start:slice_start+round(len(gtTrace)*10/np.max(gtTime))]
                sig_bpm_crt_slice = sig_bpm_crt[slice_start:slice_start+round(len(gtTrace)*10/np.max(gtTime))]
            # BVP signal normalization. 
            sig_rppg_crt_slice = (sig_rppg_crt_slice-np.min(sig_rppg_crt_slice))/(np.max(sig_rppg_crt_slice)-np.min(sig_rppg_crt_slice))
            gtTrace_crt = (gtTrace_crt-np.min(gtTrace_crt))/(np.max(gtTrace_crt)-np.min(gtTrace_crt))
            # DTW.
            dist_dtw = dtw.distance(gtTrace_crt, sig_rppg_crt_slice)
            # PCC.
            PCC = np.abs(np.corrcoef(gtTrace_crt, sig_rppg_crt_slice)[0, 1])
            # CCC.
            CCC = np.abs(errors.concordance_correlation_coefficient(bpm_true=gtTrace_crt, bpm_pred=sig_rppg_crt_slice))
            # RMSE.
            RMSE = np.sqrt(metrics.mean_absolute_error(gtHR_crt, sig_bpm_crt_slice))
            # MAE.
            MAE = metrics.mean_absolute_error(gtHR_crt, sig_bpm_crt_slice)
            # SNR.
            SNR = signal_to_noise_ratio(sig_rppg_crt_slice, Params.fps, gtHR_crt.mean())

            list_DTW[i] = list_DTW[i] + dist_dtw
            list_PCC[i] = list_PCC[i] + PCC
            list_CCC[i] = list_CCC[i] + CCC
            list_RMSE[i] = list_RMSE[i] + RMSE
            list_MAE[i] = list_MAE[i] + MAE
            list_SNR[i] = list_SNR[i] + SNR

    # averaging.
    num_win = len(np.arange(0, len(gtTrace), round(len(gtTrace)*10/np.max(gtTime))))
    list_DTW = list_DTW/num_win
    list_PCC = list_PCC/num_win
    list_CCC = list_CCC/num_win
    list_RMSE = list_RMSE/num_win
    list_MAE = list_MAE/num_win
    list_SNR = list_SNR/num_win
    # dataframe initialization.
    df_metric = pd.DataFrame(columns=['ROI', 'DTW', 'PCC', 'CCC', 'RMSE', 'MAE', 'SNR'])
    df_metric.loc[:, 'ROI'] = Params.list_roi_name
    df_metric.loc[:, 'DTW'] = list_DTW
    df_metric.loc[:, 'PCC'] = list_PCC
    df_metric.loc[:, 'CCC'] = list_CCC
    df_metric.loc[:, 'RMSE'] = list_RMSE
    df_metric.loc[:, 'MAE'] = list_MAE
    df_metric.loc[:, 'SNR'] = list_SNR

    return df_metric


def signal_to_noise_ratio(sig_bvp, fps, gtHR):

    interv1 = 0.2*60
    interv2 = 0.2*60
    NyquistF = fps/2.
    FResBPM = 0.5
    nfft = np.ceil((60*2*NyquistF)/FResBPM)

    pfreqs, power = Welch(np.reshape(sig_bvp, (1, len(sig_bvp))), fps, nfft=nfft)
    GTMask1 = np.logical_and(pfreqs>=gtHR-interv1, pfreqs<=gtHR+interv1)
    GTMask2 = np.logical_and(pfreqs>=(gtHR*2)-interv2, pfreqs<=(gtHR*2)+interv2)
    GTMask = np.logical_or(GTMask1, GTMask2)
    FMask = np.logical_not(GTMask)
    win_snr = []
    for i in range(len(power)):
        p = power[i,:]
        SPower = np.sum(p[GTMask])
        allPower = np.sum(p[FMask])
        snr = 10*np.log10(SPower/allPower)
        win_snr.append(snr)

    return np.median(win_snr)