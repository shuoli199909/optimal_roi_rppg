"""
Utils for the usage of functions in pyVHR. 
We write the util to settle the problem of unstable Python environments in pyVHR.
Most of the implementation of included functions can be sourced from pyVHR.methods.
GitHub link of pyVHR: https://github.com/phuselab/pyVHR.
"""

# Author: Shuo Li
# Email: shuoli199909@outlook.com
# Date: 2025/01/15


import cv2
import numpy as np
from scipy.signal import welch, butter, filtfilt
from numpy import abs, append, arange, arctan2, argsort, array, concatenate, \
    cos, diag, dot, eye, float64, matrix, multiply, ndarray, sign, sin, sqrt, zeros


## General functions of transforming RGB signals into BVP signals.
# Transform a list of windowed signals into a list of BVP signals.
def signals_to_bvps_cpu(sig, cpu_method, params={}):
    """
    Transform an input RGB signal in a BVP signal using a rPPG 
    method (see pyVHR.BVP.methods).
    This method must use and execute on CPU.
    You can pass also non-RGB signal but the method used must handle its shape.

    Args:
        sig (float32 ndarray): RGB Signal as float32 ndarray with shape  [num_estimators, rgb_channels, num_frames].
            You can pass also a generic signal but the method used must handle its shape and type.
        cpu_method: a method that comply with the fucntion signature documented 
            in pyVHR.BVP.methods. This method must use Numpy.
        params (dict): dictionary of usefull parameters that will be passed to the method.
    
    Returns:
        float32 ndarray: BVP signal as float32 ndarray with shape [num_estimators, num_frames].
    """
    if sig.shape[0] == 0:
        return np.zeros((0, sig.shape[2]), dtype=sig.dtype)
    cpu_sig = np.array(sig)
    if len(params) > 0:
        bvps = cpu_method(cpu_sig, **params)
    else:
        bvps = cpu_method(cpu_sig)
    return bvps

# Transform one single RGB signal into one signal BVP signal.
def RGB_sig_to_BVP(windowed_sig, fps, method=None, params={}):
    """
    Transform an input RGB windowed signal in a BVP windowed signal using a rPPG method (see pyVHR.BVP.methods).
    You can pass also non-RGB signal but the method used must handle its shape.

    Args:
        windowed_sig (list): RGB windowed signal as a list of length num_windows of np.ndarray with shape [num_estimators, rgb_channels, num_frames].
        fps (float): frames per seconds. You can pass also a generic signal but the method used must handle its shape and type.
        method: a method that comply with the fucntion signature documented 
            in pyVHR.BVP.methods. This method must use Numpy if the 'device_type' is 'cpu', Torch if the 'device_type' is 'torch', and Cupy 
            if the 'device_type' is 'cuda'.
        params(dict): dictionary of usefull parameters that will be passed to the method. If the method needs fps you can set {'fps':'adaptive'}
            in the dictionary to use the 'fps' input variable.

    Returns:
        a list of lenght num_windows of BVP signals as np.ndarray with shape [num_estimators, num_frames];
        if no BVP can be found in a window, then the np.ndarray has num_estimators == 0.
    """

    if 'fps' in params and params['fps'] == 'adaptive':
        params['fps'] = np.float32(fps)

    bvps = []
    for sig in windowed_sig:
        copy_signal = np.copy(sig)
        bvp = np.zeros((0, 1), dtype=np.float32)
        bvp = signals_to_bvps_cpu(copy_signal, method, params)

	# check for nan  
        bvp_nonan = []
        for i in range(bvp.shape[0]):
           if not np.isnan(bvp[i]).any():
              bvp_nonan.append(bvp[i])
        if len(bvp_nonan) == 0:            # if empty
           bvps.append(np.zeros((0, 1), dtype=np.float32))
        else:
           bvps.append(np.array(bvp_nonan, dtype=np.float32))

    return bvps


## General functions of transforming BVP signals into HR values.
# BPM class.
class BPM:
    """
    Provides BPMs estimate from BVP signals using CPU.

    BVP signal must be a float32 numpy.ndarray with shape [num_estimators, num_frames].
    """

    def __init__(self, data, fps, startTime=0, minHz=0.65, maxHz=4., verb=False):
        """
        Input 'data' is a BVP signal defined as a float32 Numpy.ndarray with shape [num_estimators, num_frames]
        """
        self.nFFT = 2048//1  # freq. resolution for STFTs
        if len(data.shape) == 1:
            self.data = data.reshape(1, -1)  # 2D array raw-wise
        else:
            self.data = data
        self.fps = fps                       # sample rate
        self.startTime = startTime
        self.verb = verb
        self.minHz = minHz
        self.maxHz = maxHz
    
    def BVP_to_BPM(self):
        """
        Return the BPM signal as a float32 Numpy.ndarray with shape [num_estimators, ].

        This method use the Welch's method to estimate the spectral density of the BVP signal,
        then it chooses as BPM the maximum Amplitude frequency.
        """
        if self.data.shape[0] == 0:
            return np.float32(0.0)
        Pfreqs, Power = Welch(self.data, self.fps, self.minHz, self.maxHz, self.nFFT)
        # -- BPM estimate
        Pmax = np.argmax(Power, axis=1)  # power max
        return Pfreqs[Pmax.squeeze()]

# Transform BVP signals into HR values.
def BVP_to_BPM(bvps, fps, minHz=0.65, maxHz=4.):
    """
    Computes BPMs from multiple BVPs (window) using PSDs maxima (CPU version)

    Args:
        bvps (list): list of length num_windows of BVP signal defined as float32 Numpy.ndarray with shape [num_estimators, num_frames].
        fps (float): frames per seconds.
        minHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
        maxHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).

    Returns:
        A list of length num_windows of BPM signals defined as a float32 Numpy.ndarray with shape [num_estimators, ].
        If any BPM can't be found in a window, then the ndarray has num_estimators == 0.
        
    """
    bpms = []
    obj = None
    for bvp in bvps:
        if obj is None:
            obj = BPM(bvp, fps, minHz=minHz, maxHz=maxHz)
        else:
            obj.data = bvp
        bpm_es = obj.BVP_to_BPM()
        bpms.append(bpm_es)
    return bpms


## RPPG algorithms.
# CHROM.
def cpu_CHROM(signal):
    """
    CHROM method on CPU using Numpy.

    De Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based rPPG. 
    IEEE Transactions on Biomedical Engineering, 60(10), 2878-2886.
    """
    X = signal
    Xcomp = 3*X[:, 0] - 2*X[:, 1]
    Ycomp = (1.5*X[:, 0])+X[:, 1]-(1.5*X[:, 2])
    sX = np.std(Xcomp, axis=1)
    sY = np.std(Ycomp, axis=1)
    alpha = (sX/sY).reshape(-1, 1)
    alpha = np.repeat(alpha, Xcomp.shape[1], 1)
    bvp = Xcomp - np.multiply(alpha, Ycomp)
    return bvp

# LGI.
def cpu_LGI(signal):
    """
    LGI method on CPU using Numpy.

    Pilz, C. S., Zaunseder, S., Krajewski, J., & Blazek, V. (2018). Local group invariance for heart rate estimation from face videos in the wild. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (pp. 1254-1262).
    """
    X = signal
    U, _, _ = np.linalg.svd(X)
    S = U[:, :, 0]
    S = np.expand_dims(S, 2)
    sst = np.matmul(S, np.swapaxes(S, 1, 2))
    p = np.tile(np.identity(3), (S.shape[0], 1, 1))
    P = p - sst
    Y = np.matmul(P, X)
    bvp = Y[:, 1, :]
    return bvp

# POS.
def cpu_POS(signal, **kargs):
    """
    POS method on CPU using Numpy.

    The dictionary parameters are: {'fps':float}.

    Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2016). Algorithmic principles of remote PPG. IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491. 
    """
    # Run the pos algorithm on the RGB color signal c with sliding window length wlen
    # Recommended value for wlen is 32 for a 20 fps camera (1.6 s)
    eps = 10**-9
    X = signal
    e, c, f = X.shape            # e = #estimators, c = 3 rgb ch., f = #frames
    w = int(1.6 * kargs['fps'])   # window length

    # stack e times fixed mat P
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    Q = np.stack([P for _ in range(e)], axis=0)

    # Initialize (1)
    H = np.zeros((e, f))
    for n in np.arange(w, f):
        # Start index of sliding window (4)
        m = n - w + 1
        # Temporal normalization (5)
        Cn = X[:, :, m:(n + 1)]
        M = 1.0 / (np.mean(Cn, axis=2)+eps)
        M = np.expand_dims(M, axis=2)  # shape [e, c, w]
        Cn = np.multiply(M, Cn)

        # Projection (6)
        S = np.dot(Q, Cn)
        S = S[0, :, :, :]
        S = np.swapaxes(S, 0, 1)    # remove 3-th dim

        # Tuning (7)
        S1 = S[:, 0, :]
        S2 = S[:, 1, :]
        alpha = np.std(S1, axis=1) / (eps + np.std(S2, axis=1))
        alpha = np.expand_dims(alpha, axis=1)
        Hn = np.add(S1, alpha * S2)
        Hnm = Hn - np.expand_dims(np.mean(Hn, axis=1), axis=1)
        # Overlap-adding (8)
        H[:, m:(n + 1)] = np.add(H[:, m:(n + 1)], Hnm)

    return H

# GREEN.
def cpu_GREEN(signal):
    """
    GREEN method on CPU using Numpy

    Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008). Remote plethysmographic imaging using ambient light. Optics express, 16(26), 21434-21445.
    """
    return signal[:,1,:]

# OMIT.
def cpu_OMIT(signal):
    """
    OMIT method on CPU using Numpy.

    Álvarez Casado, C., Bordallo López, M. (2022). Face2PPG: An unsupervised pipeline for blood volume pulse extraction from faces. arXiv (eprint 2202.04101).
    """

    bvp = []
    for i in range(signal.shape[0]):
        X = signal[i]
        Q, R = np.linalg.qr(X)
        S = Q[:, 0].reshape(1, -1)
        P = np.identity(3) - np.matmul(S.T, S)
        Y = np.dot(P, X)
        bvp.append(Y[1, :])
    bvp = np.array(bvp)
    return bvp

# ICA.
# JadeR implementation for the ICA algorithm.
def jadeR(X, m=None, verbose=True):
    """
    Blind separation of real signals with JADE.
    jadeR implements JADE, an Independent Component Analysis (ICA) algorithm
    developed by Jean-Francois Cardoso. See http://www.tsi.enst.fr/~cardoso/guidesepsou.html , and papers cited
    at the end of the source file.
    
    Translated into NumPy from the original Matlab Version 1.8 (May 2005) by
    Gabriel Beckers, http://gbeckers.nl .

    Parameters:
        X -- an nxT data matrix (n sensors, T samples). May be a numpy array or
             matrix.
        m -- output matrix B has size mxn so that only m sources are
             extracted.  This is done by restricting the operation of jadeR
             to the m first principal components. Defaults to None, in which
             case m=n.
        verbose -- print info on progress. Default is True.

    Returns:
        An m*n matrix B (NumPy matrix type), such that Y=B*X are separated
        sources extracted from the n*T data matrix X. If m is omitted, B is a
        square n*n matrix (as many sources as sensors). The rows of B are
        ordered such that the columns of pinv(B) are in order of decreasing
        norm; this has the effect that the `most energetically significant`
        components appear first in the rows of Y=B*X.

    Quick notes (more at the end of this file):
    o This code is for REAL-valued signals.  A MATLAB implementation of JADE
    for both real and complex signals is also available from
    http://sig.enst.fr/~cardoso/stuff.html
    o This algorithm differs from the first released implementations of
    JADE in that it has been optimized to deal more efficiently
    1) with real signals (as opposed to complex)
    2) with the case when the ICA model does not necessarily hold.
    o There is a practical limit to the number of independent
    components that can be extracted with this implementation.  Note
    that the first step of JADE amounts to a PCA with dimensionality
    reduction from n to m (which defaults to n).  In practice m
    cannot be `very large` (more than 40, 50, 60... depending on
    available memory)
    o See more notes, references and revision history at the end of
    this file and more stuff on the WEB
    http://sig.enst.fr/~cardoso/stuff.html
    o For more info on NumPy translation, see the end of this file.
    o This code is supposed to do a good job!  Please report any
    problem relating to the NumPY code gabriel@gbeckers.nl

    Copyright original Matlab code : Jean-Francois Cardoso <cardoso@sig.enst.fr>
    Copyright Numpy translation : Gabriel Beckers <gabriel@gbeckers.nl>
    """

    # GB: we do some checking of the input arguments and copy data to new
    # variables to avoid messing with the original input. We also require double
    # precision (float64) and a numpy matrix type for X.

    assert isinstance(X, ndarray),\
        "X (input data matrix) is of the wrong type (%s)" % type(X)
    origtype = X.dtype # remember to return matrix B of the same type
    X = matrix(X.astype(float64))
    assert X.ndim == 2, "X has %d dimensions, should be 2" % X.ndim
    assert (verbose == True) or (verbose == False), \
        "verbose parameter should be either True or False"

    [n,T] = X.shape # GB: n is number of input signals, T is number of samples

    if m==None:
        m=n   # Number of sources defaults to # of sensors
    assert m<=n,\
        "jade -> Do not ask more sources (%d) than sensors (%d )here!!!" % (m,n)

    if verbose:
        print ("jade -> Looking for %d sources" % m)
        print ( "jade -> Removing the mean value")
    X -= X.mean(1)

    # whitening & projection onto signal subspace
    # ===========================================
    if verbose:
        print ("jade -> Whitening the data")
    [D,U] = np.linalg.eig((X * X.T) / float(T)) # An eigen basis for the sample covariance matrix
    k = D.argsort()
    Ds = D[k] # Sort by increasing variances
    PCs = arange(n-1, n-m-1, -1)    # The m most significant princip. comp. by decreasing variance

    # --- PCA  ----------------------------------------------------------
    B = U[:,k[PCs]].T    # % At this stage, B does the PCA on m components

    # --- Scaling  ------------------------------------------------------
    scales = sqrt(Ds[PCs]) # The scales of the principal components .
    B = diag(1./scales) * B  # Now, B does PCA followed by a rescaling = sphering
    #B[-1,:] = -B[-1,:] # GB: to make it compatible with octave
    # --- Sphering ------------------------------------------------------
    X = B * X # %% We have done the easy part: B is a whitening matrix and X is white.

    del U, D, Ds, k, PCs, scales

    # NOTE: At this stage, X is a PCA analysis in m components of the real data, except that
    # all its entries now have unit variance.  Any further rotation of X will preserve the
    # property that X is a vector of uncorrelated components.  It remains to find the
    # rotation matrix such that the entries of X are not only uncorrelated but also `as
    # independent as possible".  This independence is measured by correlations of order
    # higher than 2.  We have defined such a measure of independence which
    #   1) is a reasonable approximation of the mutual information
    #   2) can be optimized by a `fast algorithm"
    # This measure of independence also corresponds to the `diagonality" of a set of
    # cumulant matrices.  The code below finds the `missing rotation " as the matrix which
    # best diagonalizes a particular set of cumulant matrices.


    # Estimation of the cumulant matrices.
    # ====================================
    if verbose:
        print ("jade -> Estimating cumulant matrices")

    # Reshaping of the data, hoping to speed up things a little bit...
    X = X.T
    dimsymm = int((m * ( m + 1)) / 2) # Dim. of the space of real symm matrices
    nbcm = dimsymm  # number of cumulant matrices
    CM = matrix(zeros([m,m*nbcm], dtype=float64)) # Storage for cumulant matrices
    R = matrix(eye(m, dtype=float64))
    Qij = matrix(zeros([m,m], dtype=float64)) # Temp for a cum. matrix
    Xim = zeros(m, dtype=float64) # Temp
    Xijm = zeros(m, dtype=float64) # Temp
    #Uns = numpy.ones([1,m], dtype=numpy.uint32)    # for convenience
    # GB: we don't translate that one because NumPy doesn't need Tony's rule

    # I am using a symmetry trick to save storage.  I should write a short note one of these
    # days explaining what is going on here.
    Range = arange(m) # will index the columns of CM where to store the cum. mats.

    for im in range(m):
        Xim = X[:,im]
        Xijm = multiply(Xim, Xim)
        # Note to myself: the -R on next line can be removed: it does not affect
        # the joint diagonalization criterion
        Qij = multiply(Xijm, X).T * X / float(T)\
            - R - 2 * dot(R[:,im], R[:,im].T)
        CM[:,Range] = Qij
        Range = Range  + m
        for jm in range(im):
            Xijm = multiply(Xim, X[:,jm])
            Qij = sqrt(2) * multiply(Xijm, X).T * X / float(T) \
                - R[:,im] * R[:,jm].T - R[:,jm] * R[:,im].T
            CM[:,Range] = Qij
            Range = Range + m

    # Now we have nbcm = m(m+1)/2 cumulants matrices stored in a big m x m*nbcm array.

    V = matrix(eye(m, dtype=float64))

    Diag = zeros(m, dtype=float64)
    On = 0.0
    Range = arange(m)
    for im in range(nbcm):
        Diag = diag(CM[:,Range])
        On = On + (Diag*Diag).sum(axis=0)
        Range = Range + m
    Off = (multiply(CM,CM).sum(axis=0)).sum(axis=0) - On

    seuil = 1.0e-6 / sqrt(T) # % A statistically scaled threshold on `small" angles
    encore = True
    sweep = 0 # % sweep number
    updates = 0 # % Total number of rotations
    upds = 0 # % Number of rotations in a given seep
    g = zeros([2,nbcm], dtype=float64)
    gg = zeros([2,2], dtype=float64)
    G = zeros([2,2], dtype=float64)
    c = 0
    s = 0
    ton = 0
    toff = 0
    theta = 0
    Gain = 0

    # Joint diagonalization proper

    if verbose:
        print ( "jade -> Contrast optimization by joint diagonalization")

    while encore:
        encore = False
        if verbose:
            print("jade -> Sweep #%3d" % sweep)
        sweep = sweep + 1
        upds  = 0
        Vkeep = V

        for p in range(m-1):
            for q in range(p+1, m):

                Ip = arange(p, m*nbcm, m)
                Iq = arange(q, m*nbcm, m)

                # computation of Givens angle
                g = concatenate([CM[p,Ip] - CM[q,Iq], CM[p,Iq] + CM[q,Ip]])
                gg = dot(g, g.T)
                ton = gg[0,0] - gg[1,1]
                toff = gg[0,1] + gg[1,0]
                theta = 0.5 * arctan2(toff, ton + sqrt(ton * ton + toff * toff))
                Gain = (sqrt(ton * ton + toff * toff) - ton) / 4.0

                # Givens update
                if abs(theta) > seuil:
                    encore = True
                    upds = upds + 1
                    c = cos(theta)
                    s = sin(theta)
                    G = matrix([[c, -s] , [s, c] ])
                    pair = array([p,q])
                    V[:,pair] = V[:,pair] * G
                    CM[pair,:] = G.T * CM[pair,:]
                    CM[:,concatenate([Ip,Iq])] = \
                        append( c*CM[:,Ip]+s*CM[:,Iq], -s*CM[:,Ip]+c*CM[:,Iq], \
                               axis=1)
                    On = On + Gain
                    Off = Off - Gain

        if verbose:
            print ( "completed in %d rotations" % upds)
        updates = updates + upds
    if verbose:
        print ("jade -> Total of %d Givens rotations" % updates)

    # A separating matrix
    # ===================

    B = V.T * B

    # Permute the rows of the separating matrix B to get the most energetic components first.
    # Here the **signals** are normalized to unit variance.  Therefore, the sort is
    # according to the norm of the columns of A = pinv(B)

    if verbose:
        print("jade -> Sorting the components")

    A = np.linalg.pinv(B)
    keys =  array(argsort(multiply(A,A).sum(axis=0)[0]))[0]
    B = B[keys,:]
    B = B[::-1,:]     # % Is this smart ?


    if verbose:
        print ("jade -> Fixing the signs")
    b = B[:,0]
    signs = array(sign(sign(b)+0.1).T)[0] # just a trick to deal with sign=0
    B = diag(signs) * B

    return B.astype(origtype)

def cpu_ICA(signal, **kargs):
    """
    ICA method on CPU using Numpy.

    The dictionary parameters are {'component':str}. Where 'component' can be 'second_comp' or 'all_comp'.

    Poh, M. Z., McDuff, D. J., & Picard, R. W. (2010). Non-contact, automated cardiac pulse measurements using video imaging and blind source separation. Optics express, 18(10), 10762-10774.    
    """
    bvp = []
    for X in signal:
        W = jadeR(X, verbose=False)  
        bvp.append(np.dot(W,X))
    
    # selector
    bvp = np.array(bvp)
    l, c, f = bvp.shape     # l=#landmks c=#3chs, f=#frames
    if kargs['component']=='all_comp':
        bvp = np.reshape(bvp, (l*c, f))  # compact into 2D matrix 
    elif kargs['component']=='second_comp':
        bvp = np.reshape(bvp[:,1,:], (l, f))
    
    # collect
    return bvp


## Other utils.
# Get video frames per second (fps).
def get_fps(videoFileName):
    """
    This method returns the fps of a video file name or path.
    """
    vidcap = cv2.VideoCapture(videoFileName)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    vidcap.release()
    return fps

# Welch method of filtering.
def Welch(bvps, fps, minHz, maxHz, nfft):
        """
        This function computes Welch'method for spectral density estimation.

        Args:
            bvps(flaot32 numpy.ndarray): BVP signal as float32 Numpy.ndarray with shape [num_estimators, num_frames].
            fps (float): frames per seconds.
            minHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
            maxHz (float): frequency in Hz used to isolate a specific subband [minHz, maxHz] (esclusive).
            nfft (int): number of DFT points, specified as a positive integer.
        Returns:
            Sample frequencies as float32 numpy.ndarray, and Power spectral density or power spectrum as float32 numpy.ndarray.
        """
        _, n = bvps.shape
        if n < 256:
            seglength = n
            overlap = int(0.8*n)  # fixed overlapping
        else:
            seglength = 256
            overlap = 200
        # -- periodogram by Welch
        F, P = welch(bvps, nperseg=seglength, noverlap=overlap, fs=fps, nfft=nfft)
        F = F.astype(np.float32)
        P = P.astype(np.float32)
        # -- freq subband (0.65 Hz - 4.0 Hz)
        band = np.argwhere((F > minHz) & (F < maxHz)).flatten()
        Pfreqs = 60*F[band]
        Power = P[:, band]

        return Pfreqs, Power

# Concordance Correlation Coefficient (CCC).
def concordance_correlation_coefficient(bpm_true, bpm_pred):
    
    cor=np.corrcoef(bpm_true, bpm_pred)[0][1]
    mean_true = np.mean(bpm_true)
    mean_pred = np.mean(bpm_pred)
    
    var_true = np.var(bpm_true)
    var_pred = np.var(bpm_pred)
    
    sd_true = np.std(bpm_true)
    sd_pred = np.std(bpm_pred)
    
    numerator = 2*cor*sd_true*sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2

    return numerator/denominator

# Signal windowing operation.
def sig_windowing(sig, wsize, stride, fps):
    """
    This method is used to divide a RGB signal into overlapping windows.

    Args:
        sig (float32 ndarray): ndarray with shape [num_frames, num_estimators, rgb_channels].
        wsize (float): window size in seconds.
        stride (float): stride between overlapping windows in seconds.
        fps (float): frames per seconds.

    Returns:
        A list of ndarray (float32) with shape [num_estimators, rgb_channels, window_frames],
        an array (float32) of times in seconds (win centers)
    """
    N = sig.shape[0]
    block_idx, timesES = sliding_straded_win_idx(N, wsize, stride, fps)
    block_signals = []
    for e in block_idx:
        st_frame = int(e[0])
        end_frame = int(e[-1])
        wind_signal = np.copy(sig[st_frame: end_frame+1])
        wind_signal = np.swapaxes(wind_signal, 0, 1)
        wind_signal = np.swapaxes(wind_signal, 1, 2)
        block_signals.append(wind_signal)
    return block_signals, timesES

# Index computation for overlapping windowed signal.
def sliding_straded_win_idx(N, wsize, stride, fps):
    """
    This method is used to compute the indices for creating an overlapping windows signal.

    Args:
        N (int): length of the signal.
        wsize (float): window size in seconds.
        stride (float): stride between overlapping windows in seconds.
        fps (float): frames per seconds.

    Returns:
        List of ranges, each one contains the indices of a window, and a 1D ndarray of times in seconds, where each one is the center of a window.
    """
    wsize_fr = wsize*fps
    stride_fr = stride*fps
    idx = []
    timesES = []
    num_win = int((N-wsize_fr)/stride_fr)+1
    s = 0
    for i in range(num_win):
        idx.append(np.arange(s, s+wsize_fr))
        s += stride_fr
        timesES.append(wsize/2+stride*i)
    return idx, np.array(timesES, dtype=np.float32)

# Apply filter to the input windowed signal.
def apply_filter(windowed_sig, filter_func, fps=None, params={}):
    """
    Apply a filter method to a windowed RGB signal or BVP signal. 

    Args:
        windowed_sig: list of length num_window of RGB signal as float32 ndarray with shape [num_estimators, rgb_channels, num_frames],
                      or BVP signal as float32 ndarray with shape [num_estimators, num_frames].
        filter_func: filter method that accept a 'windowed_sig' (pyVHR implements some filters in pyVHR.BVP.filters).
        params (dict): usefull parameters passed to the filter method.
    
    Returns:
        A filtered signal with the same shape as the input signal.
    """
    
    if 'fps' in params and params['fps'] == 'adaptive' and fps is not None:
        params['fps'] = np.float32(fps)
    filtered_windowed_sig = []
    for idx in range(len(windowed_sig)):
        transform = False
        sig = np.copy(windowed_sig[idx])
        if len(sig.shape) == 2:
            transform = True
            sig = np.expand_dims(sig, axis=1)
        if params == {}:
            filt_temp = filter_func(sig)
        else:
            filt_temp = filter_func(sig, **params)
        if transform:
            filt_temp = np.squeeze(filt_temp, axis=1)
        
        filtered_windowed_sig.append(filt_temp)

    return filtered_windowed_sig

# Band-pass filter.
def BPfilter(sig, **kargs):
    """
    Band Pass filter (using BPM band) for RGB signal and BVP signal.

    The dictionary parameters are: {'minHz':float, 'maxHz':float, 'fps':float, 'order':int}
    """
    x = np.array(np.swapaxes(sig, 1, 2))
    b, a = butter(kargs['order'], Wn=[kargs['minHz'],
                                      kargs['maxHz']], fs=kargs['fps'], btype='bandpass')
    y = filtfilt(b, a, x, axis=1)
    y = np.swapaxes(y, 1, 2)
    return y