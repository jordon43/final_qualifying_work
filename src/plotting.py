import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_trajectory(vo_xyz, gps_xyz, use_kf, use_pf, use_ukf, mode, true_xyz=None, kf_results=None, pf_results=None,
                    ukf_results=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    label_size = 15
    if mode == 'generate':
        ax.plot3D(true_xyz[:, 0], true_xyz[:, 1], true_xyz[:, 2], '-o', label='Ground Truth')
    if use_kf:
        ax.plot3D(kf_results[:, 0], kf_results[:, 1], kf_results[:, 2], '-o', label='Kalman Filter')
    if use_pf:
        ax.plot3D(pf_results[:, 0], pf_results[:, 1], pf_results[:, 2], '-o', label='Particle Filter')
    if use_ukf:
        ax.plot3D(ukf_results[:, 0], ukf_results[:, 1], ukf_results[:, 2], '-o', label='Unscented Kalman Filter')
    ax.plot3D(gps_xyz[:, 0], gps_xyz[:, 1], gps_xyz[:, 2], '-o', label='GPS')
    ax.plot3D(vo_xyz[:, 0], vo_xyz[:, 1], vo_xyz[:, 2], '-o', label='VO')
    ax.set_xlabel('X', fontsize=label_size)
    ax.set_ylabel('Y', fontsize=label_size)
    ax.set_zlabel('Z', fontsize=label_size)

    legend = ax.legend(prop={'size': label_size})
    for label in legend.get_texts():
        label.set_fontsize(label_size)

    ax.tick_params(axis='x', labelsize=label_size)
    ax.tick_params(axis='y', labelsize=label_size)
    ax.tick_params(axis='z', labelsize=label_size)

    plt.show()