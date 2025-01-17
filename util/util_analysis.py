"""
Utils for the analysis of the optimal ROI selection under different conditions.
"""

# Author: Shuo Li
# Date: 2023/05/30

import warnings
warnings.filterwarnings("ignore")  # Ignore unnecessary warnings.
import os
import cv2
import yaml
import util_pyVHR
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm
from xml.dom import minidom
from sklearn import metrics
from dtaidistance import dtw
from scipy.signal import resample


class Params():
    """Load the pre-defined parameters for preliminary analysis from a YAML file. 
       Create a class.
    """

    def __init__(self, dir_option, name_dataset) -> None:
        """Parameter calss initialization.

        Parameters
        ----------
        dir_option: Directory of the YAML file.
        name_dataset: Name of datasets. ['UBFC-rPPG', 'UBFC-Phys', 'LGI-PPGI', 'BUAA-MIHR']

        Returns
        -------

        """

        # Options.
        self.options = yaml.safe_load(open(dir_option))
        # Url.
        self.url = self.options[name_dataset]['url']
        # Dataset directory.
        self.dir_dataset = self.options[name_dataset]['dir_dataset']
        # Face detection parameters.
        self.max_num_faces = self.options[name_dataset]['max_num_faces']  # Number of target faces.
        self.minDetectionCon = self.options[name_dataset]['minDetectionCon']  # Minimal detection confidence.
        self.minTrackingCon = self.options[name_dataset]['minTrackingCon']  # Minimal tracking confidence.
        # The list containing sequence numbers of selected keypoints of different ROIs. Size = [num_roi].
        self.list_roi_num = self.options[name_dataset]['list_roi_num']
        # The list containing names of different ROIs. Size = [num_roi].
        self.list_roi_name = self.options[name_dataset]['list_roi_name']
        # RGB signal -> windowed signal.
        self.len_window = self.options[name_dataset]['len_window']  # Window length in seconds.
        self.stride_window = self.options[name_dataset]['stride_window']  # Window stride in seconds.
        self.fps = self.options[name_dataset]['fps']  # Frames per second.


class GroundTruth():
    """Load the groundtruth data. (time, PPG waveform, PPG HR). 
       Create a class.
    """

    def __init__(self, dir_dataset, name_dataset) -> None:
        """Groundtruth class initialization.

        Parameters
        ----------
        dir_option: Directory of the YAML file.
        name_dataset: Name of datasets. ['UBFC-rPPG', 'UBFC-Phys', 'LGI-PPGI', 'BUAA-MIHR']

        Returns
        -------

        """

        # Directory of the dataset.
        self.dir_dataset = dir_dataset
        # Dataset name. ['UBFC-rPPG', 'UBFC-Phys', 'LGI-PPGI', 'BUAA-MIHR'].
        self.name_dataset = name_dataset
    
    def get_GT(self, specification, num_frame_interp, slice):
        """Get the ground truth data.

        Parameters
        ----------
        specification: Specificy the dataset.
                       UBFC-rPPG: [condition, num_attendant]
                                  'simple' ~ [5-8, 10-12].
                                  'realistic' ~ [1, 3-5, 8-18, 20, 22-26, 30-49].
                                  Example: ['simple', 6]
                       UBFC-Phys: [num_attendant, num_task].
                                  num_attendant: [1-56].
                                  num_task: [1, 2, 3] - [rest, speech, arithmetic].
                                  Example: [2, 2].
                       LGI-PPGI: [name_attendant, motion].
                                 name_attendant: ['alex', 'angelo', 'cpi', 'david', 'felix', 'harun'].
                                 motion: ['gym', 'resting', 'rotation', 'talk'].
                                 Example: ['alex', 'gym'].
                       BUAA-MIHR: [num_attendant, lux, name].
                                  num_attendant: [1-14].
                                  lux: ['lux 1.0', 'lux 1.6', 'lux 2.5', 'lux 4.0', 'lux 6.3', 'lux 10.0', 
                                        'lux 15.8', 'lux 25.1', 'lux 39.8', 'lux 63.1', 'lux 100.0'].
                                  name: ['APH', 'GDB', 'HB', 'KLK', 'LW', 'LXR', 'LY', 'LZY', 'LMX', 'MXZ', 'PYC', 'QSH', 'WQT'].
        num_frame_interp: Total number of frames after interpolation.
        slice: Select a time window of the signal. [start time, end time]. The time is normalized into [0, 1].

        Returns
        -------
        gtTime: Ground truth time in numpy array. Size = [num_frames].
        gtTrace: Ground truth PPG waveform data in numpy array. Size = [num_frames].
        gtHR: Ground truth HR data in numpy array. Size = [num_frames].
        """

        if self.name_dataset == 'UBFC-rPPG':  # UBFC-rPPG dataset.
            
            if specification[0] == 'simple':  # Simple. 
                dir_crt = os.path.join(self.dir_dataset, 'DATASET_1', str(specification[1])+'-gt', 'gtdump.xmp')
                df_GT = pd.read_csv(dir_crt, header=None)
                gtTime = df_GT[0].values/1000
                gtTrace = df_GT[3].values
                gtHR = df_GT[1].values
                
            elif specification[0] == 'realistic':  # Realistic.
                dir_crt = os.path.join(self.dir_dataset, 'DATASET_2', 'subject'+str(specification[1]), 'ground_truth.txt')
                npy_GT = np.loadtxt(dir_crt)
                gtTime = npy_GT[2, :]
                gtTrace = npy_GT[0, :]
                gtHR = npy_GT[1, :]


        elif self.name_dataset == 'UBFC-Phys':  # UBFC-Phys dataset.
            # Groundtruth BVP trace.
            dir_bvp = os.path.join(self.dir_dataset, 's'+str(specification[0]), 'bvp_s'+str(specification[0])+'_T'+str(specification[1])+'.csv')
            gtTrace = np.loadtxt(dir_bvp)
            # Groundtruth video.
            dir_vid = os.path.join(self.dir_dataset, 's'+str(specification[0]), 'vid_s'+str(specification[0])+'_T'+str(specification[1])+'.avi')
            capture = cv2.VideoCapture(dir_vid)
            fps = capture.get(cv2.CAP_PROP_FPS)  # Frame rate.
            num_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))   # Number of frames.
            duration = num_frame/fps  # Video duration. (sec).
            # Groundtruth time.
            gtTime = np.linspace(start=0, stop=duration, num=num_frame)
            # Groundtruth hr.
            dir_bpm = os.path.join(self.dir_dataset, 's'+str(specification[0]), 'bpm_s'+str(specification[0])+'_T'+str(specification[1])+'.csv')
            gtHR = np.loadtxt(dir_bpm)


        elif self.name_dataset == 'LGI-PPGI':  # LGI-PPGI dataset.
            dir_vid = os.path.join(self.dir_dataset, str(specification[0]), specification[0]+'_'+specification[1], 'cv_camera_sensor_stream_handler.avi')
            dir_xml = os.path.join(self.dir_dataset, specification[0], specification[0]+'_'+specification[1], 'cms50_stream_handler.xml')
            dom = minidom.parse(dir_xml)
            # Ground truth heart rate.
            value_HR = dom.getElementsByTagName('value1')
            # Ground truth trace.
            value_Trace = dom.getElementsByTagName('value2')
            gtHR = []
            gtTrace = []
            for i in range(len(value_HR)):
                HR_tmp = value_HR[i].firstChild.data
                if '\n' not in HR_tmp:  # Exclude invalid data.
                    gtHR.append(int(HR_tmp))
                Trace_tmp = value_Trace[i].firstChild.data
                if '\n' not in Trace_tmp:  # Exclude invalid data.
                    gtTrace.append(int(Trace_tmp))
            # Ground truth time.
            capture = cv2.VideoCapture(dir_vid)
            fps = capture.get(cv2.CAP_PROP_FPS)  # Frame rate.
            num_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # Number of frames.
            duration = num_frame/fps  # Video duration. (sec).
            gtTime = np.linspace(start=0, stop=duration, num=num_frame)
            # list -> numpy array.
            gtHR = np.array(gtHR)
            gtTrace = np.array(gtTrace)

        
        elif self.name_dataset == 'BUAA-MIHR':  # BUAA-MIHR dataset.
            dir_crt = os.path.join(self.dir_dataset, 'Sub '+str(specification[0]).zfill(2), specification[1])
            # PPG trace wave.
            gtTrace = np.loadtxt(os.path.join(dir_crt, specification[1].replace(' ', '')+'_'+specification[2]+'_wave.csv'))
            # Time stamp.
            # RGB video information.
            dir_vid = os.path.join(dir_crt, specification[1].replace(' ', '')+'_'+specification[2]+'.avi')
            # Get video fps.
            capture = cv2.VideoCapture(dir_vid)
            fps = capture.get(cv2.CAP_PROP_FPS)
            num_frame = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = num_frame/fps
            gtTime = np.linspace(start=0, stop=duration, num=int(num_frame))
            # HR data.
            df_HR = pd.read_csv(os.path.join(dir_crt, specification[1].replace(' ', '')+'_'+specification[2]+'.csv'))
            gtHR = df_HR['PULSE'].values
            # HR signal resampling.
        
        # Resampling according to gtTime.
        gtTrace = resample(x=gtTrace, num=num_frame_interp)
        gtHR = resample(x=gtHR, num=num_frame_interp)
        # Time windowing.
        frame_start = round(slice[0] * len(gtTime))
        frame_end = round(slice[1] * len(gtTime))
        gtTime = gtTime[frame_start:frame_end]
        gtTrace = gtTrace[frame_start:frame_end]
        gtTrace = (gtTrace - np.min(gtTrace))/(np.max(gtTrace) - np.min(gtTrace))  # Normalize into [0, 1].
        gtHR = gtHR[frame_start:frame_end]

        return gtTime, gtTrace, gtHR


class FaceDetector():
    """A class for face detection, segmentation and RGB signal extraction."""

    def __init__(self, Params):
        """Class initialization.
        Parameters
        ----------
        Params: A class containing the pre-defined parameters.

        Returns
        -------

        """

        # Confidence.
        self.minDetectionCon = Params.minDetectionCon  # Minimal detection confidence.
        self.minTrackingCon = Params.minTrackingCon  # Minimal tracking confidence.
        # Mediapipe utils.
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)  # Face detection.
        self.mpDraw = mp.solutions.drawing_utils  # Drawing utils.
        self.faceMesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=Params.max_num_faces,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackingCon
            )  # Face mesh.
        # ROI params.
        # The list containing sequence numbers of selected keypoints of different ROIs. Size = [num_roi].
        self.list_roi_num = np.array(Params.list_roi_num, dtype=object)
        # The list containing names of different ROIs. Size = [num_roi].
        self.list_roi_name = np.array(Params.list_roi_name, dtype=object)


    def extract_landmark(self, img):
        """Extract 2D keypoint locations.
        Parameters
        ----------
        img: The input image of the current frame. Channel = [B, G, R].

        Returns
        -------
        loc_landmark: Detected normalized 3D landmarks. Size=[468, 3].
        """
        
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB.
        results = self.faceMesh.process(img_RGB)  # Apply face mesh.
        # Draw landmarks on the image.
        if results.multi_face_landmarks:
            # If the face is detected.
            # Loop over all detected faces.
            # In this experiment, we only detect one face in one video.
            for face_landmark in results.multi_face_landmarks:
                # Decompose the 3D face landmarks without resizing into the image size.
                loc_landmark = np.zeros([len(face_landmark.landmark), 3], dtype=np.float32)  # Coordinates of 3D landmarks.
                for i in range(len(face_landmark.landmark)):
                    loc_landmark[i, 0] = face_landmark.landmark[i].x
                    loc_landmark[i, 1] = face_landmark.landmark[i].y
                    loc_landmark[i, 2] = face_landmark.landmark[i].z
        else:
            # If no face is detected.
            loc_landmark = np.nan
        
        return loc_landmark


    def extract_RGB(self, img, loc_landmark):
        """Extract RGB signals from the given image and ROI.
        Parameters
        ----------
        img: 2D image. Default in BGR style. Size=[height, width, 3]
        loc_landmark: Detected normalized (0-1) 3D landmarks. Size=[468, 3].

        Returns
        -------
        sig_rgb: RGB signal of the current frame as a numpy array. Size=[num_roi, 3].
        """

        if (np.isnan(loc_landmark)).any() == True:
            # If no face is detected.
            sig_rgb = np.nan
        else:
            # If the face is detected.
            # BGR -> RGB.
            img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Rescale the input landmarks location.
            height_img = img.shape[0]
            width_img = img.shape[1]
            loc_landmark[:, 0] = loc_landmark[:, 0] * width_img
            loc_landmark[:, 1] = loc_landmark[:, 1] * height_img
            # RGB signal initialization.
            sig_rgb = np.zeros(shape=[self.list_roi_num.shape[0], 3])
            # Loop over all ROIs.
            zeros = np.zeros(img.shape, dtype=np.uint8)
            for i_roi in range(0, self.list_roi_num.shape[0]):
                # Create the current ROI mask.
                roi_name = self.list_roi_name[i_roi]
                mask = cv2.fillPoly(zeros.copy(), [loc_landmark[self.list_roi_num[self.list_roi_name==roi_name][0], :2].astype(int)], color=(1, 1, 1))
                # Only compute on a specific ROI.
                img_masked = np.multiply(img_RGB, mask)
                # Compute the RGB signal.
                sig_rgb[i_roi, :] = 3*img_masked.sum(0).sum(0)/(mask.sum())

        return sig_rgb


    def faceMeshDraw(self, img, roi_name):
        """Draw a face mesh annotations on the input image.
        Parameters
        ----------
        img: The input image of the current frame.
        roi_name: Name of the roi. The name should be in the name list.

        Returns
        -------
        img_draw: The output image after drawing the ROI of the current frame. 
        """
        
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB.
        results = self.faceMesh.process(img_RGB)  # Apply face mesh.
        mp_face_mesh = mp.solutions.face_mesh_connections
        # Draw landmarks on the image.
        if results.multi_face_landmarks:
            # Loop over all detected faces.
            # In this experiment, we only detect one face in one video.
            for face_landmark in results.multi_face_landmarks:
                # Landmark points.
                self.mpDraw.draw_landmarks(
                    image=img,
                    landmark_list=face_landmark,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.mpDraw.DrawingSpec(thickness=1, circle_radius=1),
                    connection_drawing_spec=self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)
                )
                # Decompose the 3D face landmarks.
                height_img = img.shape[0]
                width_img = img.shape[1]
                loc_landmark = np.zeros([len(face_landmark.landmark), 2], dtype=np.int32)  # Coordinates of 2D landmarks.
                for i in range(len(face_landmark.landmark)):
                    loc_landmark[i, 0] = face_landmark.landmark[i].x * width_img
                    loc_landmark[i, 1] = face_landmark.landmark[i].y * height_img
                # Create a zero vector for mask construction.
                zeros = np.zeros(img.shape, dtype=np.uint8)
                # ROI-forehead-nose-leftcheek-rightcheek-underlip. Colorization.
                mask = cv2.fillPoly(zeros.copy(), [loc_landmark[self.list_roi_num[self.list_roi_name==roi_name][0], :]], color=(1, 1, 1))
                img_draw = img + mask * 50
            
        return img_draw


def vid_to_sig(dir_vid, Params):
    """Transform the input video into RGB signals. 
       Return the signals as pandas dataframe.

    Parameters
    ----------
    dir_vid: Directory of the input video.
    Params: A class containing the pre-defined parameters for the preliminary analysis.

    Returns
    -------
    df_rgb: Dataframe containing the RGB signal of the input video.
    num_nan: Number of nan values of the extracted RGB signal.
    """

    # Input video.
    video_crt = cv2.VideoCapture(dir_vid)
    # Create the face detection object.
    Detector_crt = FaceDetector(Params=Params)
    # Create the dataframe containing the RGB signals and other necessary data.
    df_rgb = pd.DataFrame(columns=['frame', 'time', 'ROI', 'R', 'G', 'B'])
    # Start processing each frame.
    num_frame = 0
    while(video_crt.isOpened()):
        ret, img_frame = video_crt.read()
        if (ret == False) or (cv2.waitKey(1) & 0xFF == ord('q')):
            # Terminate in the end.
            break
        # Detect facial landmark keypoints. The locations are normalized into [0, 1].
        loc_landmark = Detector_crt.extract_landmark(img=img_frame)  # Size = [468, 3]
        # Extract RGB signal.
        sig_rgb = Detector_crt.extract_RGB(img=img_frame, loc_landmark=loc_landmark)  # Size = [num_roi, 3].
        # Loop over all ROIs and save the RGB data.
        df_rgb_tmp = pd.DataFrame(columns=['frame', 'time', 'ROI', 'R', 'G', 'B'], index=list(range(0, len(Params.list_roi_name))))
        for i_roi in range(len(Params.list_roi_name)):
            # ROI name.
            df_rgb_tmp.loc[i_roi, 'ROI'] = Params.list_roi_name[i_roi]
            if (np.isnan(sig_rgb)).any() == True:
                # If no face is detected.
                df_rgb_tmp.loc[i_roi, ['R', 'G', 'B']] = np.nan
            else:
                # If the face is detected.
                # RGB channels.
                df_rgb_tmp.loc[i_roi, ['R', 'G', 'B']] = sig_rgb[i_roi, :]
        # Sequence number of frame.
        num_frame = num_frame + 1
        df_rgb_tmp.loc[:, 'frame'] = num_frame
        # Time of the current frame.
        df_rgb_tmp.loc[:, 'time'] = num_frame * Params.fps
        # Change data format into numeric.
        df_rgb_tmp[['frame']] = df_rgb_tmp[['frame']].astype('int')
        df_rgb_tmp[['time', 'R', 'G', 'B']] = df_rgb_tmp[['time', 'R', 'G', 'B']].astype('float')
        # Attach to the main dataframe.
        df_rgb = pd.concat([df_rgb, df_rgb_tmp])
    # Dataframe reindex.
    df_rgb = df_rgb.reset_index(drop=True)
    # For frames with nan values, use time interpolation. 
    num_nan = df_rgb.isnull().sum().sum()
    for roi_name in Params.list_roi_name:
        df_rgb.loc[df_rgb['ROI'].values==roi_name, :] = df_rgb.loc[df_rgb['ROI'].values==roi_name, :].interpolate(method='linear')
        df_rgb.loc[df_rgb['ROI'].values==roi_name, :] = df_rgb.loc[df_rgb['ROI'].values==roi_name, :].fillna(method='ffill')
        df_rgb.loc[df_rgb['ROI'].values==roi_name, :] = df_rgb.loc[df_rgb['ROI'].values==roi_name, :].fillna(method='bfill')

    return df_rgb, num_nan


def sig_to_windowed(sig_rgb, Params):
    """Transform the original RGB signals into windowed RGB signals.

    Parameters
    ----------
    sig_rgb: The extracted RGB signal of different ROIs. Size: [num_frames, num_ROI, rgb_channels].
    Params: A class containing the pre-defined parameters for the preliminary analysis.

    Returns
    -------
    sig_rgb_win: The windowed rgb signals. Size: [num_estimators, rgb_channels, window_frames].
    timesES: An array of times in seconds.
    """

    # Parameter parsing.
    len_window = Params.len_window  # Window length in seconds.
    stride_window = Params.stride_window  # Window overlap in seconds.
    fps = Params.fps  # Frames per second.
    # Signal windowing.
    sig_rgb_win , timesES = util_pyVHR.sig_windowing(sig_rgb , len_window , stride_window , fps)

    return sig_rgb_win, timesES


def sig_windowed_to_bvp(sig_rgb_win, method, Params):
    """Transform the windowed RGB signals into blood volume pulse (BVP) signals.

    Parameters
    ----------
    sig_rgb_win: The windowed rgb signals. Size: [num_estimators, rgb_channels, window_frames].
    method: Selected rPPG algorithm. ['CHROM', 'GREEN', 'ICA', 'LGI', 'OMIT', 'PBV', 'POS', 'OMIT'].
    Params: Pre-defined parameter structure.

    Returns
    -------
    sig_bvp_win: The windowed bvp(Blood Volume Pulse) signal.
    """

    # Selected rPPG algorithms. Windowed signal -> bvp signal.
    if method == 'CHROM':
        sig_bvp_win = util_pyVHR.RGB_sig_to_BVP(windowed_sig=sig_rgb_win, fps=Params.fps, method=util_pyVHR.cpu_CHROM)
    elif method == 'GREEN':
        sig_bvp_win = util_pyVHR.RGB_sig_to_BVP(windowed_sig=sig_rgb_win, fps=Params.fps, method=util_pyVHR.cpu_GREEN)
    elif method == 'ICA':
        sig_bvp_win = util_pyVHR.RGB_sig_to_BVP(windowed_sig=sig_rgb_win, fps=Params.fps, method=util_pyVHR.cpu_ICA, params={'component': 'second_comp'})
    elif method == 'LGI':
        sig_bvp_win = util_pyVHR.RGB_sig_to_BVP(windowed_sig=sig_rgb_win, fps=Params.fps, method=util_pyVHR.cpu_LGI)
    elif method == 'OMIT':
        sig_bvp_win = util_pyVHR.RGB_sig_to_BVP(windowed_sig=sig_rgb_win, fps=Params.fps, method=util_pyVHR.cpu_OMIT)
    elif method == 'PBV':
        sig_bvp_win = util_pyVHR.RGB_sig_to_BVP(windowed_sig=sig_rgb_win, fps=Params.fps, method=util_pyVHR.cpu_PBV)
    elif method == 'PCA':
        sig_bvp_win = util_pyVHR.RGB_sig_to_BVP(windowed_sig=sig_rgb_win, fps=Params.fps, method=util_pyVHR.cpu_PCA, params={'component': 'second_comp'})
    elif method == 'POS':
        sig_bvp_win = util_pyVHR.RGB_sig_to_BVP(windowed_sig=sig_rgb_win, fps=Params.fps, method=util_pyVHR.cpu_POS, params={'fps':Params.fps})
    
    return sig_bvp_win


def rppg_hr_pipe(sig_rgb, method, Params):
    """The complete pipeline of transforming raw RGB traces into BVP & HR signals.

    Parameters
    ----------
    sig_rgb: The extracted RGB signal of different ROIs. Size: [num_frames, num_ROI, rgb_channels].
    method: Selected rPPG algorithm. ['CHROM', 'GREEN', 'ICA', 'LGI', 'OMIT', 'PBV', 'POS', 'OMIT'].
    Params: A class containing the pre-defined parameters for the preliminary analysis.

    Returns
    -------
    sig_bvp: Blood volume pulse (BVP) signal of different ROI without windowing. Size=[num_frames, num_ROI].
    sig_bpm: Beats per minute (BPM) signal of different ROI. Size=[num_frames, num_ROI].
    """

    # RGB signal -> windowed RGB signal.
    sig_rgb_win, timeES = sig_to_windowed(sig_rgb=sig_rgb, Params=Params)
    # Windowed RGB signal -> windowed raw bvp signal.
    sig_bvp_win = sig_windowed_to_bvp(sig_rgb_win=sig_rgb_win, method=method, Params=Params)
    # Windowed raw bvp signal -> windowed filtered bvp signal.
    sig_bvp_win_filtered = util_pyVHR.apply_filter(sig_bvp_win, util_pyVHR.BPfilter, params={'order':6, 'minHz':0.65, 'maxHz':4.0, 'fps':Params.fps})
    # Fill nan values.
    for i_window in range(len(sig_bvp_win_filtered)):
        is_nan = np.any(np.isnan(sig_bvp_win_filtered[i_window]))
        if is_nan == False:
            continue
        elif i_window == 0:
            sig_bvp_win_filtered[i_window] = np.ones(28, np.shape(sig_bvp_win_filtered[i_window])[1])
        else:
            sig_bvp_win_filtered[i_window] = sig_bvp_win_filtered[i_window-1]
    # De-windowing bvp signal.
    for i in range(len(sig_bvp_win_filtered)):
        if i == 0:
            sig_bvp = (sig_bvp_win_filtered[i])[:, :round(Params.fps*Params.stride_window)]
        else:
            sig_bvp = np.concatenate((sig_bvp, (sig_bvp_win_filtered[i])[:, :round(Params.fps*Params.stride_window)]), axis=1)
    sig_bvp = np.concatenate((sig_bvp, (sig_bvp_win_filtered[i])[:, round(Params.fps*Params.stride_window):]), axis=1)
    # Windowed filtered bvp signal -> bpm(Beats Per Minute) signal.
    multi_sig_bpm = util_pyVHR.BVP_to_BPM(bvps=sig_bvp_win_filtered, fps=Params.fps, minHz=0.65, maxHz=4.0)
    # Remove nan values.
    for i in range(len(multi_sig_bpm)):
        if len(multi_sig_bpm[i]) != len(Params.list_roi_name):
            multi_sig_bpm[i] = multi_sig_bpm[i-1]
    # List -> numpy array.
    sig_bpm = np.array(multi_sig_bpm)
    # Reshaping.
    sig_bvp_old = np.transpose(sig_bvp, [1, 0])
    sig_bpm_old = np.transpose(sig_bpm, [0, 1])
    # Resampling.
    sig_bvp = np.zeros_like(sig_rgb[:, :, 0])
    sig_bpm = np.zeros_like(sig_rgb[:, :, 0])
    # Across different ROIs.
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
    """The complete pipeline for rPPG algorithm evaluation.
       This evaluation is based on BVP & BPM signals.
       The selected metrics are: [PCC, CCC, RMSE, MAE, DTW].

    Parameters
    ----------
    sig_bvp: BVP signal of different ROIs after de-windowing. Size=[num_frames, num_ROI].
    sig_bpm: BPM signal of different ROI. Size=[num_frames, num_ROI].
    gtTime: Ground truth time in numpy array. Size = [num_frames].
    gtTrace: Ground truth PPG waveform data in numpy array. Size = [num_frames].
    gtHR: Ground truth HR data in numpy array. Size = [num_frames].
    Params: A class containing the pre-defined parameters for the preliminary analysis.

    Returns
    -------
    list_DTW: DTW metric. Size = [num_roi].
    list_PCC = Pearson's Correlation Coefficient (PCC). Size = [num_roi].
    list_CCC = Concordance Correlation Coefficient (CCC). Size = [num_roi].
    list_RMSE = Root Mean Square Error (RMSE). Size = [num_roi].
    list_MAE = Mean Absolute Error (MAE). Size = [num_roi].
    """

    # Metrics initialization of different ROIs.
    list_DTW = np.zeros(len(Params.list_roi_name))
    list_PCC = np.zeros(len(Params.list_roi_name))
    list_CCC = np.zeros(len(Params.list_roi_name))
    list_RMSE = np.zeros(len(Params.list_roi_name))
    list_MAE = np.zeros(len(Params.list_roi_name))
    list_MAPE = np.zeros(len(Params.list_roi_name))
    # Process different ROI respectively.
    for i in tqdm(range(len(sig_bpm[0, :]))):
        # BVP signal of each ROI.
        sig_bvp_crt = sig_bvp[:, i]
        # BPM signal of each ROI.
        sig_bpm_crt = sig_bpm[:, i]
        # Windowing. This process helps stabilize the evaluation results. Len_win = 10s.
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
            CCC = np.abs(util_pyVHR.concordance_correlation_coefficient(bpm_true=gtTrace_crt, bpm_pred=sig_rppg_crt_slice))
            # RMSE.
            RMSE = np.sqrt(metrics.mean_absolute_error(gtHR_crt, sig_bpm_crt_slice))
            # MAE.
            MAE = metrics.mean_absolute_error(gtHR_crt, sig_bpm_crt_slice)
            # MAPE.
            MAPE = metrics.mean_absolute_percentage_error(gtHR_crt, sig_bpm_crt_slice)

            list_DTW[i] = list_DTW[i] + dist_dtw
            list_PCC[i] = list_PCC[i] + PCC
            list_CCC[i] = list_CCC[i] + CCC
            list_RMSE[i] = list_RMSE[i] + RMSE
            list_MAE[i] = list_MAE[i] + MAE
            list_MAPE[i] = list_MAPE[i] + MAPE

    # Averaging.
    num_win = len(np.arange(0, len(gtTrace), round(len(gtTrace)*10/np.max(gtTime))))
    list_DTW = list_DTW/num_win
    list_PCC = list_PCC/num_win
    list_CCC = list_CCC/num_win
    list_RMSE = list_RMSE/num_win
    list_MAE = list_MAE/num_win
    list_MAPE = list_MAPE/num_win
    # Dataframe initialization.
    df_metric = pd.DataFrame(columns=['ROI', 'DTW', 'PCC', 'CCC', 'RMSE', 'MAE'])
    df_metric.loc[:, 'ROI'] = Params.list_roi_name
    df_metric.loc[:, 'DTW'] = list_DTW
    df_metric.loc[:, 'PCC'] = list_PCC
    df_metric.loc[:, 'CCC'] = list_CCC
    df_metric.loc[:, 'RMSE'] = list_RMSE
    df_metric.loc[:, 'MAE'] = list_MAE
    df_metric.loc[:, 'MAPE'] = list_MAPE

    return df_metric