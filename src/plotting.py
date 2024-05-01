import matplotlib.pyplot as plt

def plot_trajectory(true_xyz, vo_xyz, gps_xyz, kf_results=None, pf_results=None, ukf_results=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(true_xyz[:, 0], true_xyz[:, 1], true_xyz[:, 2], '-o', label='True')
    ax.plot3D(kf_results[:, 0], kf_results[:, 1], kf_results[:, 2], '-o', label='Kalman Filter')
    ax.plot3D(pf_results[:, 0], pf_results[:, 1], pf_results[:, 2], '-o', label='Particle Filter')
    ax.plot3D(ukf_results[:, 0], ukf_results[:, 1], ukf_results[:, 2], '-o', label='Unscented Kalman Filter')
    ax.plot3D(gps_xyz[:, 0], gps_xyz[:, 1], gps_xyz[:, 2], '-o', label='GPS')
    ax.plot3D(vo_xyz[:, 0], vo_xyz[:, 1], vo_xyz[:, 2], '-o', label='VO')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()