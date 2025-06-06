import numpy as np
from color_histogram import color_histogram
from chi2_cost import chi2_cost


def observe(particles, frame, bbox_height, bbox_width, hist_bin, hist, sigma_observe):

    particles_w = np.zeros((particles.shape[0],1))

    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    for i in range(particles.shape[0]):
        xmin = min(max(0, round(particles[i,0] - 0.5*bbox_width)), frame_width-1)
        xmax = min(max(0, round(particles[i,0] + 0.5*bbox_width)), frame_width-1)
        ymin = min(max(0, round(particles[i,1] - 0.5*bbox_height)), frame_height-1)
        ymax = min(max(0, round(particles[i,1] + 0.5*bbox_height)), frame_height-1)
        hist_i = color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin)
        chi2_i = chi2_cost(hist_i, hist)
        particles_w[i,:] = 1/(np.sqrt(2*np.pi)*sigma_observe) * np.exp(-chi2_i**2/(2*sigma_observe**2))
   
    particles_w = particles_w/np.sum(particles_w)

    return particles_w
