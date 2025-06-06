import numpy as np

def filter_keypoints(img, keypoints, patch_size = 9):
    '''
    Inputs:
    - keypoints:    (q, 2) numpy array of keypoint locations [x, y]
    Returns:
    - keypoints:    (q', 2) numpy array of keypoint locations [x, y] that are far enough from edges
    '''
    # TODO: Filter out keypoints that are too close to the edges
    img_h, img_w = img.shape[0], img.shape[1]
    d = patch_size // 2

    x_coord = keypoints[:,0].flatten()
    y_coord = keypoints[:,1].flatten()

    x_filter1 = x_coord >= d
    x_filter2 = x_coord <= img_w - d
    y_filter1 = y_coord >= d
    y_filter2 = y_coord <= img_h - d

    x_filter = np.logical_and(x_filter1, x_filter2)
    y_filter = np.logical_and(y_filter1, y_filter2)
    total_filter = np.logical_and(x_filter, y_filter)
    
    return keypoints[total_filter]

# The implementation of the patch extraction is already provided here
def extract_patches(img, keypoints, patch_size = 9):
    '''
    Extract local patches for each keypoint
    Inputs:
    - img:          (h, w) gray-scaled images
    - keypoints:    (q, 2) numpy array of keypoint locations [x, y]
    - patch_size:   size of each patch (with each keypoint as its center)
    Returns:
    - desc:         (q, patch_size * patch_size) numpy array. patch descriptors for each keypoint
    '''
    h, w = img.shape[0], img.shape[1]
    img = img.astype(float) / 255.0
    offset = int(np.floor(patch_size / 2.0))
    ranges = np.arange(-offset, offset + 1)
    desc = np.take(img, ranges[:,None] * w + ranges + (keypoints[:, 1] * w + keypoints[:, 0])[:, None, None]) # (q, patch_size, patch_size)
    desc = desc.reshape(keypoints.shape[0], -1) # (q, patch_size * patch_size)
    return desc

