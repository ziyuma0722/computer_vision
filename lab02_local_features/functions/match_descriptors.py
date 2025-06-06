import numpy as np

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    # TODO: implement this function please
    #adjust shape for broadcasting, result in shape(q1, q2, feature_dim).
    desc1_reshape = np.reshape(desc1, (-1, 1, desc1.shape[-1]))
    desc2_reshape = np.reshape(desc2, (1, -1, desc2.shape[-1]))   

    distances = np.sum((desc1_reshape - desc2_reshape) ** 2, axis=-1)
    return distances

def match_descriptors(desc1, desc2, method = "one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    matches = None

    indices_c = np.argmin(distances, axis=1)
    row_num = np.arange(desc1.shape[0])
    matches_forward = np.column_stack((row_num,indices_c))

    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1
        # TODO: implement the one-way nearest neighbor matching here
        # You may refer to np.argmin to find the index of the minimum over any axis
        matches = matches_forward
    elif method == "mutual":
        # TODO: implement the mutual nearest neighbor matching here
        # You may refer to np.min to find the minimum over any axis
        indices_r = np.argmin(distances, axis=0)
        col_num = np.arange(desc2.shape[0])
        matches_backward = np.column_stack((indices_r,col_num))
        matches = np.array([x for x in set(tuple(x) for x in matches_forward) & set(tuple(x) for x in matches_backward)])
    elif method == "ratio":
        # TODO: implement the ratio test matching here
        # You may use np.partition(distances,2,axis=0)[:,1] to find the second smallest value over a row
        two_least_distant = np.partition(distances, 2)[:,:2]
        ratio_check = two_least_distant[:,0] / two_least_distant[:,1] < ratio_thresh
        matches = matches_forward[ratio_check]
    else:
        raise NotImplementedError
    return matches

