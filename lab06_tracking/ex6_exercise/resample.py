import numpy as np

def resample(particles, particles_w):

    sample_indices = np.random.choice(particles.shape[0], size=particles.shape[0], replace=True, p=particles_w.flatten())

    sample_particles = particles[sample_indices]
    sample_particles_w = particles_w[sample_indices] / np.sum(particles_w[sample_indices])

    return sample_particles, sample_particles_w

