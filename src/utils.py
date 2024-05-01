import numpy as np
from scipy.stats import multivariate_normal

def create_gaussian_particles(mean, std, N):
    particles = np.random.normal(loc=mean, scale=std, size=(N, 3))
    return particles

def state_transition_function(x, dt):
    F = np.array([[1, dt, 0,  0, 0,  0],
                  [0,  1, 0,  0, 0,  0],
                  [0,  0, 1, dt, 0,  0],
                  [0,  0, 0,  1, 0,  0],
                  [0,  0, 0,  0, 1, dt],
                  [0,  0, 0,  0, 0,  1]])
    return np.dot(F, x)

def measurement_function(x):
    return x[:3]

def update_particle_weights(particles, weights, z, R):
    for i, particle in enumerate(particles):
        weights[i] *= multivariate_normal(mean=particle[:3], cov=R).pdf(z)
    weights += 1.e-300
    weights /= np.sum(weights)

def predict_particles(particles, std):
    particles += np.random.normal(0, std, size=particles.shape)

def calculate_rmse(estimations, true_values):
    return np.sqrt(((estimations - true_values) ** 2).mean(axis=0))

def systematic_resample(weights):
    N = len(weights)
    positions = (np.arange(N) + np.random.random()) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes

def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights.fill(1.0 / len(weights))