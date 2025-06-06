import numpy as np


def propagate(particles, frame_height, frame_width, params):

    #position noise
    w = np.random.normal(loc=0, scale=params["sigma_position"], size=(2, particles.shape[0]))

    #A matrices
    if params["model"] == 0: 
        A = np.array([[1, 0], [0, 1]])
    
    if params["model"] == 1:    
        A = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        #velocity noise
        w_vel = np.random.normal(loc=0, scale=params["sigma_velocity"], size=(2, particles.shape[0]))
        w = np.vstack((w,w_vel))
        
    #propagate    
    propa_particles = (np.matmul(A, particles.T) + w).T

    #make sure the center inside the frame
    propa_particles[:,0] = np.clip(propa_particles[:,0], 0, frame_width-1)
    propa_particles[:,1] = np.clip(propa_particles[:,1], 0, frame_height-1)


    return propa_particles
