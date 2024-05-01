import numpy as np
import filters
import utils


class Simulation:
    def __init__(self):
        self.true_xyz = None
        self.gps_xyz = None
        self.vo_xyz = None
        self.kf_results = None
        self.pf_results = None
        self.ukf_results = None
        self.kf_rmse = None
        self.pf_rmse = None
        self.ukf_rmse = None

    def run(self, num_points, gps_noise, vo_noise, use_kf, use_pf, use_ukf):
        true_xyz = np.vstack((10 * np.sin(np.linspace(0, 2 * np.pi, num_points)),
                              10 * np.cos(np.linspace(0, 2 * np.pi, num_points)),
                              np.linspace(0, 2 * np.pi, num_points))).T

        # true_xyz = data_loader.load_true_coordinates('../media/data_coord/true_coordinates.txt')
        gps_xyz = true_xyz + np.random.normal(0, gps_noise, size=(len(true_xyz), 3))
        vo_xyz = true_xyz + np.random.normal(0, vo_noise, size=(len(true_xyz), 3))

        # vo_xyz = data_loader.load_vo_coordinates('vo_coordinates.txt', scale=0.3)

        num_points = len(true_xyz)

        kf = filters.create_kalman_filter(dim_x=6, dim_z=3, dt=1)
        ukf = filters.create_ukf(dim_x=6, dim_z=3, dt=1, hx=utils.measurement_function,
                                 fx=utils.state_transition_function)
        particles, weights = filters.create_particles(mean=np.array([0, 0, 0]), std=np.array([1, 1, 1]), N=1000)

        kf_results = np.zeros((num_points, 3))
        ukf_results = np.zeros((num_points, 3))
        pf_results = np.zeros((num_points, 3))

        for t in range(num_points):
            z_gps = gps_xyz[t]
            z_vo = vo_xyz[t]
            z = (z_gps + z_vo) / 2

            if use_kf:
                kf.predict()
                kf.update(z)
                kf_results[t, :] = kf.x[:3]

            if use_ukf:
                ukf.predict()
                ukf.update(z)
                ukf_results[t, :] = ukf.x[:3]


            if use_pf:
                R_gps = np.diag([gps_noise ** 2] * 3)
                R_vo = np.diag([vo_noise ** 2] * 3)
                utils.predict_particles(particles, std=np.array([gps_noise, gps_noise, gps_noise]))
                utils.update_particle_weights(particles, weights, z_gps, R_gps)
                utils.predict_particles(particles, std=np.array([vo_noise, vo_noise, vo_noise]))
                utils.update_particle_weights(particles, weights, z_vo, R_vo)
                indexes = utils.systematic_resample(weights)
                utils.resample_from_index(particles, weights, indexes)
                pf_results[t, :] = np.average(particles, weights=weights, axis=0)

        self.true_xyz = true_xyz
        self.gps_xyz = gps_xyz
        self.vo_xyz = vo_xyz
        self.kf_results = kf_results
        self.pf_results = pf_results
        self.ukf_results = ukf_results

        self.kf_rmse = utils.calculate_rmse(kf_results, true_xyz)
        self.pf_rmse = utils.calculate_rmse(pf_results, true_xyz)
        self.ukf_rmse = utils.calculate_rmse(ukf_results, true_xyz)

        return {'kf': self.kf_rmse if use_kf else None,
                'pf': self.pf_rmse if use_pf else None,
                'ukf': self.ukf_rmse if use_ukf else None}
