import numpy as np
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
from scipy.stats import multivariate_normal

def create_kalman_filter(dim_x, dim_z, dt):
    kf = KalmanFilter(dim_x=6, dim_z=3)
    kf.F = np.array([[1, 0, 0, 1, 0, 0],
                     [0, 1, 0, 0, 1, 0],
                     [0, 0, 1, 0, 0, 1],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0]])
    kf.x = np.array([0., 0., 0., 0., 0., 0.])
    kf.P *= 500.
    kf.R *= np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])
    kf.Q[-1, -1] *= 0.01
    kf.Q[3:, 3:] = Q_discrete_white_noise(dim=3, dt=1, var=0.01)
    return kf

def create_particles(mean, std, N):
    particles = np.random.normal(loc=mean, scale=std, size=(N, 3))
    weights = np.ones(N) / N
    return particles, weights

def predict_particles(particles, std):
    particles[:, :3] += std * np.random.randn(particles.shape[0], 3)

def update_particle_weights(particles, weights, measurement, R):
    for i, particle in enumerate(particles):
        weights[i] *= multivariate_normal(mean=particle[:3], cov=R).pdf(measurement)
    weights += 1.e-300
    weights /= weights.sum()

def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights.fill(1.0 / len(weights))


def create_ukf(dim_x, dim_z, dt, hx, fx):
    points = MerweScaledSigmaPoints(n=dim_x, alpha=.1, beta=2., kappa=0)
    ukf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z, dt=dt, hx=hx, fx=fx, points=points)
    ukf.x = np.zeros(dim_x)
    ukf.P *= 10
    ukf.R = np.diag([1.0, 1.0, 1.0]) * 0.01
    ukf.Q = Q_discrete_white_noise(dim=3, dt=dt, var=0.01, block_size=2)

    return ukf