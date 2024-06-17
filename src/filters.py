import numpy as np
import scipy
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
from filterpy.monte_carlo import systematic_resample
from scipy.stats import multivariate_normal


def configure_kalman_filter(gps_data, vo_data):
    gps_data = np.array(gps_data)
    vo_data = np.array(vo_data)

    initial_state = gps_data[0]

    initial_velocity = np.array([0, 0, 0])

    kf = KalmanFilter(dim_x=6, dim_z=3)

    kf.R = np.eye(3) * 0.1

    dt = 1
    kf.Q = np.eye(6) * 0.1

    kf.F = np.array([[1, 0, 0, dt, 0, 0],
                     [0, 1, 0, 0, dt, 0],
                     [0, 0, 1, 0, 0, dt],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1]])

    kf.H = np.array([[1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0]])

    kf.x[:3] = initial_state
    kf.x[3:] = initial_velocity

    kf.P = np.eye(6) * 1000

    return kf


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
    kf.R *= np.array([[.5, 0.0, 0.0],
                      [0.0, .5, 0.0],
                      [0.0, 0.0, .5]])
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


def create_ukf(dim_x, dim_z, dt, hx, fx, gps_data):
    points = MerweScaledSigmaPoints(n=dim_x, alpha=.01, beta=.01, kappa=0)

    ukf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z, dt=dt, hx=hx, fx=fx, points=points)

    ukf.x = np.concatenate((gps_data[1], np.zeros(3)))

    ukf.P *= 0.1

    ukf.R = np.eye(3) * 0.1

    q = Q_discrete_white_noise(dim=3, dt=1, var=0.1)
    ukf.Q = np.eye(6)
    ukf.Q[3:, 3:] = q

    return ukf