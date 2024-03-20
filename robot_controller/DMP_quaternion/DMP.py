from __future__ import division, print_function
from dmp_position import PositionDMP
from dmp_rotation import RotationDMP
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

def main():
    # Load a demonstration file containing robot positions.
    demo = np.loadtxt("trajectory_2.dat", delimiter=" ", skiprows=1)

    tau = 0.002 * len(demo)
    t = np.arange(0, tau, 0.002)
    demo_p = demo[:, 0:3]

    demo_o = demo[:,3:demo.shape[-1]-1]

    # Check for sign flips
    for i in range(len(demo_o)-1):
        if demo_o[i].dot(demo_o[i+1]) < 0:
            demo_o[i+1] *= -1

    theta = [np.linalg.norm(v) for v in demo_o]
    axis = [v/np.linalg.norm(v) for v in demo_o]

    demo_q = np.array([Quaternion(axis=a,radians=t) for (a,t) in zip(axis,theta)])

    for i in range(len(demo_q)-1):
        if np.array([demo_q[i][0], demo_q[i][1], demo_q[i][2], demo_q[i][3]]).dot(np.array([demo_q[i+1][0], demo_q[i+1][1], demo_q[i+1][2], demo_q[i+1][3]])) < 0:
            demo_q[i+1] *= -1

    demo_quat_array = np.empty((len(demo_q),4))
    for n, d in enumerate(demo_q):
        demo_quat_array[n] = [d[0],d[1],d[2],d[3]]

    # In both canonical_system.py and dmp_position.py you will find some lines missing implementation.
    # Fix those first.

    N = 100  # TODO: Try changing the number of basis functions to see how it affects the output.
    dmp = PositionDMP(n_bfs=N, alpha=48.0)
    dmp.train(demo_p, t, tau)

    # Rotation...
    dmp_rotation = RotationDMP(n_bfs=N, alpha=48.0)
    dmp_rotation.train(demo_q, t, tau)

    # Generate an output trajectory from the trained DMP
    dmp_p, dmp_dp, dmp_ddp = dmp.rollout(t, tau)

    dmp_r, dmp_dr, dmp_ddr = dmp_rotation.rollout(t, tau)
    result_quat_array = np.empty((len(dmp_r),4))
    for n, d in enumerate(dmp_r):
        result_quat_array[n] = [d[0],d[1],d[2],d[3]]

    fig3, axs = plt.subplots(5, 1, sharex=True)
    axs[0].plot(t, demo_quat_array[:, 0], label='Demonstration')
    axs[0].plot(t, result_quat_array[:, 0], label='DMP')
    #axs[0].set_xlabel('t (s)')
    axs[0].set_ylabel('w')
    axs[0].legend()

    axs[1].plot(t, demo_quat_array[:, 1], label='Demonstration')
    axs[1].plot(t, result_quat_array[:, 1], label='DMP')
    #axs[1].set_xlabel('t (s)')
    axs[1].set_ylabel('i')

    axs[2].plot(t, demo_quat_array[:, 2], label='Demonstration')
    axs[2].plot(t, result_quat_array[:, 2], label='DMP')
    #axs[2].set_xlabel('t (s)')
    axs[2].set_ylabel('j')

    axs[3].plot(t, demo_quat_array[:, 3], label='Demonstration')
    axs[3].plot(t, result_quat_array[:, 3], label='DMP')
    #axs[3].set_xlabel('t (s)')
    axs[3].set_ylabel('k')

    quat_error = [Quaternion.distance(Quaternion(q1),Quaternion(q2)) for (q1,q2) in zip(demo_quat_array,result_quat_array)]

    axs[4].plot(t, quat_error, label='Error',c='r')
    axs[4].set_xlabel('t (s)')
    axs[4].set_ylabel('e')
    axs[4].legend()

    def update(i, x, y, z, x2, y2, z2):
        x_data = Quaternion(demo_quat_array[i]).conjugate * Quaternion([0,1,0,0]) * demo_quat_array[i]
        x.set_data([0,x_data[1]],[0,x_data[2]])
        x.set_3d_properties([0,x_data[3]])
        x_data = Quaternion(result_quat_array[i]).conjugate * Quaternion([0,1,0,0]) * result_quat_array[i]
        x2.set_data([0,x_data[1]],[0,x_data[2]])
        x2.set_3d_properties([0,x_data[3]])

        y_data = Quaternion(demo_quat_array[i]).conjugate * Quaternion([0,0,1,0]) * demo_quat_array[i]
        y.set_data([0,y_data[1]],[0,y_data[2]])
        y.set_3d_properties([0,y_data[3]])
        y_data = Quaternion(result_quat_array[i]).conjugate * Quaternion([0,0,1,0]) * result_quat_array[i]
        y2.set_data([0,y_data[1]],[0,y_data[2]])
        y2.set_3d_properties([0,y_data[3]])
        
        z_data = Quaternion(demo_quat_array[i]).conjugate * Quaternion([0,0,0,1]) * demo_quat_array[i]
        z.set_data([0,z_data[1]],[0,z_data[2]])
        z.set_3d_properties([0,z_data[3]])
        z_data = Quaternion(result_quat_array[i]).conjugate * Quaternion([0,0,0,1]) * result_quat_array[i]
        z2.set_data([0,z_data[1]],[0,z_data[2]])
        z2.set_3d_properties([0,z_data[3]])
        print(i)
        return x, y, z, x2, y2, z2,

    fig4 = plt.figure()
    ax = p3.Axes3D(fig4)
    ax.plot3D(demo_p[:, 0], demo_p[:, 1], demo_p[:, 2], label='Demonstration')
    ax.plot3D(dmp_p[:, 0], dmp_p[:, 1], dmp_p[:, 2], label='DMP')
    ax.legend()

    n_steps = len(t)
    print(n_steps)

    def animate3D(i, plot_dmp, dmp_points, update_bar):
        # Updating progress bar
        update_bar(1)
        # Plotting DMP
        plot_dmp.set_data(dmp_points[0:i+1,0], dmp_points[0:i+1,1])
        plot_dmp.set_3d_properties(dmp_points[0:i+1,2])

    def animate2D(i, t, plt_dmp_x, plt_dmp_y, plt_dmp_z, plt_error, dmp_points, update_bar):
        update_bar(1)
        plt_dmp_x.set_data(t[0:i+1], dmp_points[0:i+1, 0])
        plt_dmp_y.set_data(t[0:i+1], dmp_points[0:i+1, 1])
        plt_dmp_z.set_data(t[0:i+1], dmp_points[0:i+1, 2])
        
        plt_error.set_data(t[0:i+1], np.linalg.norm(demo_p[0:i+1,:] - dmp_points[0:i+1,:], axis=1) )


    with tqdm(total=n_steps) as tbar:
        # Animation of 3D path
        fig = plt.figure()
        ax = p3.Axes3D(fig)

        # Static plots and settings
        ax.view_init(elev=12, azim=-160)
        ax.plot3D(demo_p[:, 0], demo_p[:, 1], demo_p[:, 2], label='Demonstration')

        plot_dmp = ax.plot3D([], [], [], label='DMP')[0]
        ax.legend() 
        
        ani = animation.FuncAnimation(fig, animate3D, fargs=[plot_dmp, dmp_p, tbar.update], frames=n_steps-1, interval=25, blit=False, repeat=False)
        
        # Set up formatting for the movie files
        print(animation.writers.list())
        Writer = animation.writers['pillow']
        writer = Writer(fps=60, bitrate=2000)
        ani.save('3D_ori_param.gif', writer=writer)

    with tqdm(total=n_steps) as tbar:
        max_error = np.max(np.linalg.norm(demo_p - dmp_p, axis=1))
        print("Max error",max_error)
        # Animation of 2D path
        fig, axs = plt.subplots(4, 1, sharex=True)
        t = np.arange(0, tau, 0.002)

        axs[0].set_xlabel('t (s)')
        axs[0].set_ylabel('X (m)')
        axs[0].plot(t, demo_p[:, 0], label='Demonstration')
        plt_dmp_x, = axs[0].plot([], [], label='DMP')
        
        axs[1].set_xlabel('t (s)')
        axs[1].set_ylabel('Y (m)')
        axs[1].plot(t, demo_p[:, 1], label='Demonstration')
        plt_dmp_y, = axs[1].plot([], [], label='DMP')
        
        axs[2].set_xlabel('t (s)')
        axs[2].set_ylabel('Z (m)')
        axs[2].set_ylim(0.25, 0.5)
        axs[2].legend()
        axs[2].plot(t, demo_p[:, 2], label='Demonstration')
        plt_dmp_z, = axs[2].plot([], [], label='DMP')
        
        axs[3].set_xlabel('t (s)')
        axs[3].set_ylabel('|| Demo - DMP ||')
        axs[3].set_ylim(0, round(max_error,2))
        plt_error, = axs[3].plot([], [], label='Euclidian diff of Demo and DMP')

        ani = animation.FuncAnimation(fig, animate2D, fargs=[t, plt_dmp_x, plt_dmp_y, plt_dmp_z, plt_error, dmp_p, tbar.update], frames=n_steps-1, interval=25, blit=False, repeat=False)
        Writer = animation.writers['pillow']
        writer = Writer(fps=60, bitrate=2000)
        ani.save('2D_ori_param.gif', writer=writer)

    plt.show()

    return dmp_p, dmp_dp, dmp_ddp

if __name__ == '__main__':
    main()
