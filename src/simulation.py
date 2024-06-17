import numpy as np
import filters
import utils


class Simulation:
    def __init__(self):
        self.mode = None
        self.true_xyz = None
        self.gps_xyz = None
        self.vo_xyz = None
        self.use_kf = None
        self.use_pf = None
        self.use_ukf = None
        self.kf_results = None
        self.pf_results = None
        self.ukf_results = None
        self.kf_rmse = None
        self.pf_rmse = None
        self.ukf_rmse = None
        self.gps_rmse = None
        self.vo_rmse = None

    def set_trajectory(self, trajectory_name):

        trajectory_generators = {
            'Random Walk': self.generate_random_walk,
            'Sinusoidal': self.generate_sinusoidal,
            'Circular': self.generate_circular,
            'Figure-Eight': self.generate_figure_eight
        }
        self.trajectory_generator = trajectory_generators.get(trajectory_name)

    def generate_random_walk(self, num_points, step_size=0.5):

        points = np.cumsum(np.random.randn(num_points, 3) * step_size, axis=0)
        return points

    def generate_sinusoidal(self, num_points):

        t = np.linspace(0, 2 * np.pi, num_points)
        points = np.vstack((np.sin(t), np.cos(t), t)).T
        return points

    def generate_circular(self, num_points):

        t = np.linspace(0, 2 * np.pi, num_points)
        points = np.vstack((np.sin(t), np.cos(t), np.zeros(num_points))).T
        return points

    def generate_figure_eight(self, num_points):

        t = np.linspace(0, 2 * np.pi, num_points)
        points = np.vstack((np.sin(t), np.sin(t) * np.cos(t), t)).T
        return points


    def run(self, num_points, gps_noise, vo_noise, use_kf, use_pf, use_ukf, mode='generate'):
        self.mode = mode
        self.use_kf = use_kf
        self.use_pf = use_pf
        self.use_ukf = use_ukf

        if mode == 'generate':
            if hasattr(self, 'trajectory_generator'):
                true_xyz = self.trajectory_generator(num_points)
            else:
                raise ValueError('Не выбрана траектория для генерации данных.')

            gps_xyz = true_xyz + np.random.normal(0, gps_noise, size=(len(true_xyz), 3))
            vo_xyz = true_xyz + np.random.normal(0, vo_noise, size=(len(true_xyz), 3))
            num_points = len(true_xyz)
        else:
            gps_data_path = 'coordinates_HOME.txt'
            vo_data_path = 'vo_scaled_for_filter.txt'

            gps_data = utils.read_data(gps_data_path)
            vo_data = utils.read_data(vo_data_path)

            gps_xyz = np.array(gps_data)
            vo_xyz = np.array(vo_data)

            num_points = len(gps_xyz)



        kf = filters.create_kalman_filter(dim_x=6, dim_z=3, dt=1)
        ukf = filters.create_ukf(dim_x=6, dim_z=3, dt=1, hx=utils.measurement_function,
                                 fx=utils.state_transition_function, gps_data=gps_xyz)
        particles, weights = filters.create_particles(mean=gps_xyz[0], std=np.array([0.001, 0.001, 0.001]), N=1000)


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

                R_gps = np.diag([gps_noise] * 3)
                R_vo = np.diag([vo_noise] * 3)


                utils.predict_particles(particles, std=np.array([gps_noise, gps_noise, gps_noise]))

                utils.update_particle_weights(particles, weights, z_gps, R_gps)

                utils.predict_particles(particles, std=np.array([vo_noise, vo_noise, vo_noise]))

                utils.update_particle_weights(particles, weights, z_vo, R_vo)

                indexes = utils.systematic_resample(weights)
                utils.resample_from_index(particles, weights, indexes)

                pf_results[t, :] = np.average(particles, weights=weights, axis=0)


        if mode == 'generate':
            self.true_xyz = true_xyz
        self.gps_xyz = gps_xyz
        self.vo_xyz = vo_xyz
        self.kf_results = kf_results
        self.pf_results = pf_results
        self.ukf_results = ukf_results


        if mode == 'generate':
            self.kf_rmse = utils.calculate_rmse(kf_results, true_xyz)
            self.pf_rmse = utils.calculate_rmse(pf_results, true_xyz)
            self.ukf_rmse = utils.calculate_rmse(ukf_results, true_xyz)
            self.gps_rmse = utils.calculate_rmse(gps_xyz, true_xyz)
            self.vo_rmse = utils.calculate_rmse(vo_xyz, true_xyz)
        else:
            self.kf_rmse = utils.calculate_rmse(kf_results, gps_data)
            self.pf_rmse = utils.calculate_rmse(pf_results, gps_data)
            self.ukf_rmse = utils.calculate_rmse(ukf_results, gps_data)



        return {'kf': self.kf_rmse if use_kf else None,
                'pf': self.pf_rmse if use_pf else None,
                'ukf': self.ukf_rmse if use_ukf else None,
                'gps': self.gps_rmse if mode == 'generate' else None,
                'vo': self.vo_rmse if mode == 'generate' else None,
                }