import numpy as np

from scipy import signal #for the scipy.signal.convolve2d function
from scipy import ndimage #for the scipy.ndimage.maximum_filter

import cv2 as cv

# Harris corner detector
def extract_harris(img, sigma = 1.0, k = 0.05, thresh = 1e-5):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength
    '''
    # Convert to float
    img = img.astype(float) / 255.0

    # 1. Compute image gradients in x and y direction
    # TODO: implement the computation of the image gradients Ix and Iy here.
    # You may refer to scipy.signal.convolve2d for the convolution.
    # Do not forget to use the mode "same" to keep the image size unchanged.
    Ix = signal.convolve2d(img, np.array([-1, 0, 1]).reshape(1, -1) * 0.5, mode="same")
    Iy = signal.convolve2d(img, np.array([-1, 0, 1]).reshape(-1, 1) * 0.5, mode="same")
    
    # 2. Blur the computed gradients
    # TODO: compute the blurred image gradients
    # You may refer to cv2.GaussianBlur for the gaussian filtering (border_type=cv2.BORDER_REPLICATE)
    # 3. Compute elements of the local auto-correlation matrix "M"
    # TODO: compute the auto-correlation matrix here
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    Mxx = cv.GaussianBlur(Ixx, (7, 7), sigma, borderType=cv.BORDER_REPLICATE)
    Myy = cv.GaussianBlur(Iyy, (7, 7), sigma, borderType=cv.BORDER_REPLICATE) 
    Mxy = cv.GaussianBlur(Ixy, (7, 7), sigma, borderType=cv.BORDER_REPLICATE) 

    M = np.array([[Mxx,Mxy],[Mxy,Myy]])

    # 4. Compute Harris re!sponse function C
    # TODO: compute the Harris response function C here
    det = Mxx * Myy - Mxy * Mxy
    trace = Mxx + Myy       
    C = det - k * (trace ** 2)

    # 5. Detection with threshold and non-maximum suppression
    # TODO: detection and find the corners here
    # For the non-maximum suppression, you may refer to scipy.ndimage.maximum_filter to check a 3x3 neighborhood.
    # You may refer to np.where to find coordinates of points that fulfill some condition; Please, pay attention to the order of the coordinates.
    # You may refer to np.stack to stack the coordinates to the correct output format
    condition1 = C > thresh
    C_max_filter = ndimage.maximum_filter(C, 3)
    condition2 = (C == C_max_filter)

    corners = np.argwhere((condition1 & condition2))
    corners[:,[0,1]] = corners[:, [1,0]]

    return corners, C

