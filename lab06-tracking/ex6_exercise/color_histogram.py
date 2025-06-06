import numpy as np


def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin):

    bb_frame = frame[ymin:ymax, xmin:xmax, :]

    color_hist = np.zeros((3, hist_bin))

    color_hist[0] = np.histogram(bb_frame[:,:,0], bins=hist_bin)[0]
    color_hist[1] = np.histogram(bb_frame[:,:,1], bins=hist_bin)[0]
    color_hist[2] = np.histogram(bb_frame[:,:,2], bins=hist_bin)[0]

    color_hist = color_hist/np.sum(color_hist)   

    return color_hist




