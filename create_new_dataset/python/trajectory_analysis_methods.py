# importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import transformations
import json
import pandas as pd
import scipy.signal as sign


class Traj(object):
    def __init__(self, analysis_type, n_elem, tot_angle):
        # type of analysis: "rot" or "transl"
        self.analysis_type = analysis_type
        # number of elements: number of frames of the trajectory
        self.n_elem_traj = n_elem
        # tot_angle: total rotation angle or angle of the circular translation angle
        self.tot_angle = tot_angle
        # X,Y,Z coordinates of the trajectory
        self.X = np.array([])
        self.Y = np.array([])
        self.Z = np.array([])
        # X_rot,Y_rot,Z_rot: rotation angles of the camera around X,Y,Z axes
        self.X_rot = np.array([])
        self.Y_rot = np.array([])
        self.Z_rot = np.array([])
        # length of the trajectory
        self.traj_length = 0
        # rotation_angle: angle of rotation for the rotation trajectory
        self.rotation_angle = 0
        # quaternions of the trajectory for each frame
        self.quat = np.array([])
        self.poses = []
        # methods of the class Traj

    def read_open3d_traj(self, trajectory_file):
        '''
        this function reads the trajectory coordinates and rotation angles from trajectory.log files.
        This function is called only for reconstructed trajectories
        '''
        with open(trajectory_file, "r") as f:
            data = f.readlines()
        transform_list = []
        for i in range(0, len(data), 5):
            transform_list.append(np.array([[float(string) for string in data[i + 1].split()],
                                            [float(string) for string in data[i + 2].split()],
                                            [float(string) for string in data[i + 3].split()],
                                            [float(string) for string in data[i + 4].split()]]))
        self.X = np.array([transform[0][3] for transform in transform_list])
        self.Y = np.array([transform[1][3] for transform in transform_list])
        self.Z = np.array([transform[2][3] for transform in transform_list])
        angles = np.array([transformations.euler_from_matrix(rot, 'rxyz') for rot in transform_list])
        self.X_rot = angles[:, 0] * 180 / np.pi
        self.Y_rot = angles[:, 1] * 180 / np.pi
        self.Z_rot = angles[:, 2] * 180 / np.pi

    def ideal_circular_traj(self, R, center_point):
        '''
        This function generates coordinates and rotation for circular translation trajectory for scene 1
        '''
        step_theta = self.tot_angle / self.n_elem_traj
        eps = 1.1102230246251565e-16
        angles = np.arange(0, self.tot_angle + 10 ** 3 * eps, step_theta)
        self.X = R * np.sin(angles * np.pi / 180) + center_point[0]
        self.Z = R * np.cos(angles * np.pi / 180) + center_point[1]
        self.Y = np.zeros(self.X.shape)
        self.X_rot = np.zeros(angles.shape)
        self.Y_rot = np.zeros(angles.shape)
        self.Z_rot = np.zeros(angles.shape)

    def evaluate_circular_traj_length(self, R):
        '''
        For scene 1, the trajectory length is computed for each frame of the trajectory
        '''
        self.traj_length = np.linspace(0, 2 * np.pi * R, self.n_elem_traj + 1)

    def delete_360_degrees_phase_shifts(self):
        '''
        This function filter possible rotation angles shift of the trajectory of 360°. These shifts are
        corrected and they are filtered from the trajectory. This function is useful for scene 2
        '''
        for i in range(1, len(self.X_rot)):
            if (self.X_rot[i] - self.X_rot[i - 1]) > 340:
                self.X_rot[i:len(self.X_rot)] -= 360
            elif (self.X_rot[i] - self.X_rot[i - 1]) < -340:
                self.X_rot[i:len(self.X_rot)] += 360
        for i in range(1, len(self.Z_rot)):
            if (self.Z_rot[i] - self.Z_rot[i - 1]) > 340:
                self.Z_rot[i:len(self.Z_rot)] -= 360
            elif (self.Z_rot[i] - self.Z_rot[i - 1]) < -340:
                self.Z_rot[i:len(self.Z_rot)] += 360

    def detect_shifts(self, delta_rot, rot_array):
        '''
        This fuction detects the presence of shifts of 90° for rotation angles of trajectories.
        In this case, the ground-truth trajectory is shifted of 90° when the reconstructed trajectory shifts of 90°.
        This is done to compute the error respect to rotation angles.

        delta_rot: rotation angle delta at which the shift is checked
        rot_array: rotation array around one axis of reconstructed trajectory
        '''
        shift_types = []
        shift_found = False
        up_shift = False
        down_shift = False
        rot_idx_cv_init = 0.0
        rot_idx_cv_end = 0.0
        shift = np.array([])
        for i in range(self.n_elem_traj - delta_rot + 1):
            if rot_array[i + delta_rot] - rot_array[i] >= 90:
                if not shift_found:
                    rot_idx_cv_init = i + np.float(delta_rot)
                    shift_found = True
                    up_shift = True
                else:
                    rot_idx_cv_end = np.float(i)
            elif rot_array[i + delta_rot] - rot_array[i] <= -90:
                if not shift_found:
                    rot_idx_cv_init = i + np.float(delta_rot)
                    shift_found = True
                    down_shift = True
                else:
                    rot_idx_cv_end = np.float(i)
            else:
                if shift_found:
                    if up_shift:
                        shift = np.append(shift, (rot_idx_cv_init + rot_idx_cv_end) / 2)
                        up_shift = False
                        shift_types.append("up")
                    if down_shift:
                        shift = np.append(shift, (rot_idx_cv_init + rot_idx_cv_end) / 2)
                        down_shift = False
                        shift_types.append("down")
                    shift_found = False
        if up_shift:
            shift = np.append(shift, rot_idx_cv_init)
            shift_types.append("up")
        if down_shift:
            shift = np.append(shift, rot_idx_cv_init)
            shift_types.append("down")
        return shift, shift_types

    def generate_fitting_phase_fun(self, rot_array, delta_rot):
        '''
        This function creates the ground-truth rotation X or Z angles for for the scene 2.
        This angles should be arrays with a number of elements equal to the number of frames and equal to zero.
        In practice, these angles are shifted of 180°, to consider possible sudden variations of shift of
        reconstructed trajectories
        '''
        rot_ideal = np.zeros([len(rot_array)])
        init_shifts, init_shift_types = self.detect_shifts(delta_rot, rot_array)
        flipped_rot_array = np.flip(rot_array)
        final_shifts, final_shift_types = self.detect_shifts(delta_rot, flipped_rot_array)

        shift_idx = np.array(np.int32((init_shifts + np.flip(-final_shifts + len(rot_array) - 1)) / 2))
        shift_idx_float = np.array((init_shifts + np.flip(-final_shifts + len(rot_array) - 1)) / 2)
        for i in range(len(init_shift_types)):
            if init_shift_types[i] == "up":
                rot_ideal[shift_idx[i]:] += 180
            elif init_shift_types[i] == "down":
                rot_ideal[shift_idx[i]:] -= 180
        return rot_ideal, shift_idx_float

    def generate_angle_rot_phase_fun(self, X_shift_idx, Z_shift_idx, delta_rot):
        '''
        This function creates the ground-truth rotation Y angle for for the scene 2.
        The Y angle changes with constant angular velocity. When there is a shift of 180° of the X and Z axes,
        the Y angular velocity changes its versus in the reconstructed trajectory.
        This phenomenon is considered in the ground-truth trajectory.
        '''
        Y_shift_idx = np.int32(np.array([]))
        for X_idx in X_shift_idx:
            X_idx_low = X_idx - delta_rot / 2
            X_idx_high = X_idx + delta_rot / 2
            for Z_idx in Z_shift_idx:
                if X_idx_low < Z_idx < X_idx_high:
                    Y_shift_idx = np.append(Y_shift_idx, (X_idx + Z_idx) / 2)
        Y_shift_idx = np.int32(Y_shift_idx)
        Y_rot_ideal = np.zeros(self.X_rot.shape)
        delta_angle = -self.tot_angle / self.n_elem_traj
        for i in range(1, len(Y_rot_ideal)):
            Y_rot_ideal[i] = Y_rot_ideal[i - 1] + delta_angle
            if i in Y_shift_idx:
                delta_angle = -delta_angle
        return Y_rot_ideal

    def ideal_rotation_traj(self, X_rot, Z_rot):
        '''
        This function creates the ground-truth trajectory for the scene 2

        X_rot: rotation X angle of the reconstructed trajectory
        Z_rot: rotation Z angle of the reconstructed trajectory
        '''
        self.X = np.zeros(self.n_elem_traj + 1)
        self.Y = np.zeros(self.n_elem_traj + 1)
        self.Z = np.zeros(self.n_elem_traj + 1)
        delta_rot = np.int32(np.round(self.n_elem_traj / 10))
        self.X_rot, X_shift_idx = self.generate_fitting_phase_fun(X_rot, delta_rot)
        self.Z_rot, Z_shift_idx = self.generate_fitting_phase_fun(Z_rot, delta_rot)
        self.Y_rot = self.generate_angle_rot_phase_fun(X_shift_idx, Z_shift_idx, delta_rot)

    def ideal_rotation_traj_no_shifts(self):
        self.X = np.zeros(self.n_elem_traj + 1)
        self.Y = np.zeros(self.n_elem_traj + 1)
        self.Z = np.zeros(self.n_elem_traj + 1)
        self.X_rot = np.zeros(self.n_elem_traj + 1)
        eps = 1.1102230246251565e-16
        self.Y_rot = np.arange(0, self.tot_angle + 10 ** 3 * eps, self.tot_angle / self.n_elem_traj)
        self.Z_rot = np.zeros(self.n_elem_traj + 1)

    def evaluate_rotation_angle_traj(self):
        '''
        This function computes the rotation angle around the Y axes for scene 2 for each frame
        '''
        self.rotation_angle = np.arange(0, 1, 1 / len(self.X_rot)) * self.tot_angle

    def read_open3d_traj_to_transl_and_quaternion(self, trajectory_file):
        '''
        This function reads a trajectory from trajectory.log file and sets 3D coordinates of the Traj object
        and the quaternions
        '''
        with open(trajectory_file, "r") as f:
            data = f.readlines()
        transform_list = []
        for i in range(0, len(data), 5):
            transform_list.append(np.array([[float(string) for string in data[i + 1].split()],
                                            [float(string) for string in data[i + 2].split()],
                                            [float(string) for string in data[i + 3].split()],
                                            [float(string) for string in data[i + 4].split()]]))
        self.X = np.array([transform[0][3] for transform in transform_list])
        self.Y = np.array([transform[1][3] for transform in transform_list])
        self.Z = np.array([transform[2][3] for transform in transform_list])
        self.quat = np.array([transformations.quaternion_from_matrix(rot, 'rxyz') for rot in transform_list])

    def compute_quat_from_angles(self):
        '''
        This function computes the quaternions from Euler angles for the trajectory object
        '''
        self.quat = np.array(
            [transformations.quaternion_from_euler(X_angle / 180 * np.pi, Y_angle / 180 * np.pi, Z_angle / 180 * np.pi)
             for X_angle, Y_angle, Z_angle in zip(self.X_rot, self.Y_rot, self.Z_rot)])

    def apply_handheld_noise(self, noise_dir):
        self.X[1:len(self.X)] += np.loadtxt(noise_dir + "\\x_trasl_hist.txt")[1:len(self.X)]
        self.Y[1:len(self.X)] += np.loadtxt(noise_dir + "\\y_trasl_hist.txt")[1:len(self.X)]
        self.Z[1:len(self.X)] += np.loadtxt(noise_dir + "\\z_trasl_hist.txt")[1:len(self.X)]
        self.X_rot[1:len(self.X)] += np.loadtxt(noise_dir + "\\x_rot_hist.txt")[1:len(self.X)]
        self.Y_rot[1:len(self.X)] += np.loadtxt(noise_dir + "\\y_rot_hist.txt")[1:len(self.X)]
        self.Z_rot[1:len(self.X)] += np.loadtxt(noise_dir + "\\z_rot_hist.txt")[1:len(self.X)]

    def write_open3d_traj_log_file(self, trajectory_file):
        with open(trajectory_file, 'w') as f:
            for i, pose in enumerate(self.poses[1:]):
                f.write('{} {} {}\n'.format(i, i, i + 1))
                f.write('{0:.8f} {1:.8f} {2:.8f} {3:.8f}\n'.format(
                    pose[0, 0], pose[0, 1], pose[0, 2], pose[0, 3]))
                f.write('{0:.8f} {1:.8f} {2:.8f} {3:.8f}\n'.format(
                    pose[1, 0], pose[1, 1], pose[1, 2], pose[1, 3]))
                f.write('{0:.8f} {1:.8f} {2:.8f} {3:.8f}\n'.format(
                    pose[2, 0], pose[2, 1], pose[2, 2], pose[2, 3]))
                f.write('{0:.8f} {1:.8f} {2:.8f} {3:.8f}\n'.format(
                    pose[3, 0], pose[3, 1], pose[3, 2], pose[3, 3]))

    def create_poses_from_transl_and_rot(self):
        self.poses = []
        for index in range(len(self.X)):
            z_rot_matrix = np.array(
                [[np.cos(self.Z_rot[index] * np.pi / 180), -np.sin(self.Z_rot[index] * np.pi / 180), 0],
                 [np.sin(self.Z_rot[index] * np.pi / 180), np.cos(self.Z_rot[index] * np.pi / 180), 0],
                 [0, 0, 1]])
            y_rot_matrix = np.array(
                [[np.cos(self.Y_rot[index] * np.pi / 180), 0, np.sin(self.Y_rot[index] * np.pi / 180)],
                 [0, 1, 0],
                 [-np.sin(self.Y_rot[index] * np.pi / 180), 0, np.cos(self.Y_rot[index] * np.pi / 180)]])
            x_rot_matrix = np.array(
                [[1, 0, 0],
                 [0, np.cos(self.X_rot[index] * np.pi / 180), -np.sin(self.X_rot[index] * np.pi / 180)],
                 [0, np.sin(self.X_rot[index] * np.pi / 180), np.cos(self.X_rot[index] * np.pi / 180)]])
            rot_matrix = z_rot_matrix.dot(y_rot_matrix)
            rot_matrix = rot_matrix.dot(x_rot_matrix)
            transl = np.matmul(rot_matrix,
                               np.expand_dims(np.array([self.X[index], self.Y[index], self.Z[index]]), axis=1))
            # transl = np.expand_dims(transl, axis=1)
            pose = np.concatenate((rot_matrix, transl), axis=1)
            last_row = np.array([0, 0, 0, 1])
            last_row = np.expand_dims(last_row, axis=0)
            pose = np.concatenate((pose, last_row), axis=0)
            self.poses.append


def evaluate_general_traj_errors(reconstr_traj_file, analysis_type, n_elem_traj, tot_angle, R=0, center_point=None):
    '''
    This function computes the errors between ground-truth and reconstructed trajectories.
    The ground-truth trajectory is artificially created, since it was imposed during the dataset creation.

    reconstr_traj_file: file from which the reconstructed trajectory is read;
    analysis_type: type of analysis performed (scene 1: "transl", scene 2: "rot")
    n_elem_traj: number of elements of the reconstructed trajectories
    tot_angle: total angle of rotation of the camera around the Y axis for scene 2 and total angle of rotation around the
    circular translation trajectory for scene 1
    R: radius of the circular trajectory for scene 1
    center_point: center point of the circular trajectory for scene 2
    '''
    reconstr_traj = Traj(analysis_type, n_elem_traj, tot_angle)
    reconstr_traj.read_open3d_traj(reconstr_traj_file)
    ideal_traj = Traj(analysis_type, n_elem_traj, tot_angle)
    if analysis_type == "transl":
        ideal_traj.ideal_circular_traj(R, center_point)
        ideal_traj.evaluate_circular_traj_length(R)
    else:
        reconstr_traj.delete_360_degrees_phase_shifts()
        ideal_traj.ideal_rotation_traj(reconstr_traj.X_rot, reconstr_traj.Z_rot)
        reconstr_traj.evaluate_rotation_angle_traj()

    # error along different directions and planes
    X_error = np.abs(reconstr_traj.X - ideal_traj.X[0:reconstr_traj.X.shape[0]])
    Y_error = np.abs(reconstr_traj.Y - ideal_traj.Y[0:reconstr_traj.Y.shape[0]])
    Z_error = np.abs(reconstr_traj.Z - ideal_traj.Z[0:reconstr_traj.Z.shape[0]])
    # creating a dictionary of errors containing errors arrays (each array element is for each frame)
    errors_dict = {
        "X_error": X_error,
        "Y_error": Y_error,
        "Z_error": Z_error,
        "X_rot_error": np.abs(reconstr_traj.X_rot - ideal_traj.X_rot[0:reconstr_traj.X_rot.shape[0]]),
        "Y_rot_error": np.abs(reconstr_traj.Y_rot - ideal_traj.Y_rot[0:reconstr_traj.Y_rot.shape[0]]),
        "Z_rot_error": np.abs(reconstr_traj.Z_rot - ideal_traj.Z_rot[0:reconstr_traj.Z_rot.shape[0]]),
        "X_Z_plane_error": np.sqrt(X_error ** 2 + Z_error ** 2),
        "X_Y_plane_error": np.sqrt(X_error ** 2 + Y_error ** 2),
        "Y_Z_plane_error": np.sqrt(Y_error ** 2 + Z_error ** 2),
        "error_3D": np.sqrt(X_error ** 2 + Y_error ** 2 + Z_error ** 2)
    }

    # computing trajectory length variation (reconstructed trajectory)
    length_rec_diff = np.array(
        [np.linalg.norm(np.array([reconstr_traj.X[i], reconstr_traj.Y[i], reconstr_traj.Z[i]]) -
                        np.array([reconstr_traj.X[i - 1], reconstr_traj.Y[i - 1], reconstr_traj.Z[i - 1]]))
         for i in range(1, len(reconstr_traj.Z))])
    # computing trajectory length (reconstructed trajectory)
    length_rec = np.array([0])
    for i in range(len(length_rec_diff)):
        length_rec = np.append(length_rec, [length_rec[i] + length_rec_diff[i]], axis=0)
    # computing trajectory length variation (ground-truth trajectory)
    length_ideal_diff = np.array([np.linalg.norm(np.array([ideal_traj.X[i], ideal_traj.Y[i], ideal_traj.Z[i]])
                                                 - np.array(
        [ideal_traj.X[i - 1], ideal_traj.Y[i - 1], ideal_traj.Z[i - 1]]))
                                  for i in range(1, len(reconstr_traj.Z))])
    # computing trajectory length (ground-truth trajectory)
    length_ideal = np.array([0])
    for i in range(len(length_ideal_diff)):
        length_ideal = np.append(length_ideal, [length_ideal[i] + length_ideal_diff[i]], axis=0)
    # computing the gradient of the trajectory length difference between reconstructed and real trajectories
    if analysis_type == "rot":
        step_length_diff = np.gradient(length_rec - length_ideal,
                                       reconstr_traj.rotation_angle[1] - reconstr_traj.rotation_angle[0])
    else:
        step_length_diff = np.gradient(length_rec - length_ideal, ideal_traj.traj_length[1] - ideal_traj.traj_length[0])
    # inserting length and gradient in errors_dict
    errors_dict["length_rec"] = length_rec
    errors_dict["length_ideal"] = length_ideal
    errors_dict["step_length_diff"] = step_length_diff
    return errors_dict, reconstr_traj, ideal_traj


def plot_3d_traj_projections(reconstr_traj, ideal_traj, plot_filedir):
    '''
    This function plots the camera trajectory projected on the 3 space planes (X-Y,X-Z,Y-Z)
    of both reconstructed and ground-truth trajectories and saves them to "figures" folder
    '''
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 2.66), constrained_layout=True)
    ax.plot(reconstr_traj.X, reconstr_traj.Z)
    ax.plot(ideal_traj.X, ideal_traj.Z, linewidth=0.5)
    ax.grid()
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("X-Z plane trajectory")
    ax.set_aspect('equal', adjustable='box')
    ax.legend(["Reconstruction trajectory", "Theoretical trajectory"], loc=(1.04, 0))
    plt.savefig(plot_filedir + "\\translations1.svg", dpi=2000)
    plt.close()
    # fig.suptitle("Trajectory of 3D sensor from reconstruction")

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 2.66), constrained_layout=True)
    ax.plot(reconstr_traj.X, reconstr_traj.Y)
    ax.plot(ideal_traj.X, ideal_traj.Y, linewidth=0.5)
    # ax[1].set_xlim([0-0.05*X[len(X)-1],(1+0.05)*X[len(X)-1]])
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("X-Y plane trajectory")
    ax.legend(["Reconstruction trajectory", "Theoretical trajectory"])
    ax.grid()
    plt.savefig(plot_filedir + "\\translations2.svg", dpi=2000)
    plt.close()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 2.66), constrained_layout=True)
    ax.plot(reconstr_traj.Z, reconstr_traj.Y)
    ax.plot(ideal_traj.Z, ideal_traj.Y, linewidth=0.5)
    # ax[2].set_xlim([0 - 0.05 * Y[len(Y) - 1], (1 + 0.05) * Y[len(Y) - 1]])
    ax.set_xlabel("Z [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Z-Y plane trajectory")
    ax.legend(["Reconstruction trajectory", "Theoretical trajectory"])
    ax.grid()
    plt.savefig(plot_filedir + "\\translations3.svg", dpi=2000)
    plt.close()


def plot_camera_orientation_angles(analysis_type, reconstr_traj, ideal_traj, plot_filedir):
    '''
    This function plots the camera rotation angles (X_rot,Y_rot,Z_rot)
    of both reconstructed and ground-truth trajectories and saves them to "figures" folder
    '''
    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 2.66), constrained_layout=True)
    if analysis_type == "rot":
        ax1.plot(reconstr_traj.rotation_angle, reconstr_traj.X_rot)
        ax1.plot(reconstr_traj.rotation_angle, ideal_traj.X_rot, linewidth=0.75)
        ax1.set_xlabel("Theoretical rotation angle around Y [°]")
    else:
        ax1.plot(ideal_traj.traj_length, reconstr_traj.X_rot)
        ax1.plot(ideal_traj.traj_length, ideal_traj.X_rot, linewidth=0.75)
        ax1.set_xlabel("Theoretical trajectory length [m]")
    ax1.grid()
    ax1.set_ylabel("Rotation around X axis [°]")
    ax1.set_title("Rotation around X axis")
    ax1.legend(["Reconstruction rotation", "Theoretical rotation"])
    plt.savefig(plot_filedir + "\\rotations1.svg", dpi=2000)
    plt.close()
    # fig1.suptitle("Rotation angles of 3D sensor from reconstruction")

    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 2.66), constrained_layout=True)
    if analysis_type == "rot":
        ax1.plot(reconstr_traj.rotation_angle, reconstr_traj.Y_rot)
        ax1.plot(reconstr_traj.rotation_angle, ideal_traj.Y_rot, linewidth=0.75)
        ax1.set_xlabel("Theoretical rotation angle around Y [°]")
    else:
        ax1.plot(ideal_traj.traj_length, reconstr_traj.Y_rot)
        ax1.plot(ideal_traj.traj_length, ideal_traj.Y_rot, linewidth=0.75)
        ax1.set_xlabel("Theoretical trajectory length [m]")
    ax1.set_ylabel("Rotation around Y axis [°]")
    ax1.set_title("Rotation around Y axis")
    ax1.legend(["Reconstruction rotation", "Theoretical rotation"])
    ax1.grid()
    plt.savefig(plot_filedir + "\\rotations2.svg", dpi=2000)
    plt.close()

    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 2.66), constrained_layout=True)
    if analysis_type == "rot":
        ax1.plot(reconstr_traj.rotation_angle, reconstr_traj.Z_rot)
        ax1.plot(reconstr_traj.rotation_angle, ideal_traj.Z_rot, linewidth=0.75)
        ax1.set_xlabel("Theoretical rotation angle around Y [°]")
    else:
        ax1.plot(ideal_traj.traj_length, reconstr_traj.Z_rot)
        ax1.plot(ideal_traj.traj_length, ideal_traj.Z_rot, linewidth=0.75)
        ax1.set_xlabel("Theoretical trajectory length [m]")
    ax1.set_ylabel("Rotation around Z axis [°]")
    ax1.set_title("Rotation around Z axis")
    ax1.legend(["Reconstruction rotation", "Theoretical rotation"])
    ax1.grid()
    plt.savefig(plot_filedir + "\\rotations3.svg", dpi=2000)
    plt.close()


def plot_traj_length(analysis_type, reconstr_traj, ideal_traj, errors_dict, plot_filedir):
    '''
    This function plots the trajectory length difference between reconstructed and ground-truth trajectories
    and the gradient of that difference and saves these graphs to "figures" folder
    '''
    fig1, ax1 = plt.subplots(nrows=2, ncols=1, figsize=(5, 8), constrained_layout=True)
    if analysis_type == "rot":
        ax1[0].plot(reconstr_traj.rotation_angle, errors_dict["length_rec"] - errors_dict["length_ideal"])
        ax1[0].set_xlabel("Theoretical rotation angle around Y [°]")
    else:
        ax1[0].plot(ideal_traj.traj_length, errors_dict["length_rec"] - errors_dict["length_ideal"])
        ax1[0].set_xlabel("Theoretical trajectory length [m]")
    ax1[0].grid()
    ax1[0].set_ylabel("Length [m]")
    ax1[0].set_title("Difference reconstructed-real trajectories length")

    if analysis_type == "rot":
        ax1[1].plot(reconstr_traj.rotation_angle, errors_dict["step_length_diff"])
        ax1[1].set_xlabel("Theoretical rotation angle around Y [°]")
        ax1[1].set_ylabel("Length derivative [m/°]")
    else:
        ax1[1].plot(ideal_traj.traj_length, errors_dict["step_length_diff"])
        ax1[1].set_xlabel("Theoretical trajectory length [m]")
        ax1[1].set_ylabel("Length derivative [m/m]")
    ax1[1].set_title("Difference real-reconstructed trajectories length gradient")
    ax1[1].grid()
    plt.savefig(plot_filedir + "\\length.png")
    plt.close()


def write_all_data_to_json(errors_dict, dataset_name, reconstr_traj, ideal_traj, json_filename_path):
    '''
    This function writes all errors_dict data, regarding trajectories and errors computed to .json file
    '''
    json_file_text = {
        "acquisition_name": dataset_name,
        "X-Z Plane max error": max(errors_dict["X_Z_plane_error"]),
        "X-Z Plane mean error": np.mean(errors_dict["X_Z_plane_error"]),
        "X-Z Plane SD error": np.std(errors_dict["X_Z_plane_error"]),
        "X-Y Plane max error": max(errors_dict["X_Y_plane_error"]),
        "X-Y Plane mean error": np.mean(errors_dict["X_Y_plane_error"]),
        "X-Y Plane SD error": np.std(errors_dict["X_Y_plane_error"]),
        "Y-Z Plane max error": max(errors_dict["Y_Z_plane_error"]),
        "Y-Z Plane mean error": np.mean(errors_dict["Y_Z_plane_error"]),
        "Y-Z Plane SD error": np.std(errors_dict["Y_Z_plane_error"]),
        "3D error max": max(errors_dict["error_3D"]),
        "3D error mean": np.mean(errors_dict["error_3D"]),
        "3D error SD": np.std(errors_dict["error_3D"]),
        "X Rotation max error": max(errors_dict["X_rot_error"]),
        "X Rotation mean error": np.mean(errors_dict["X_rot_error"]),
        "X Rotation SD error": np.std(errors_dict["X_rot_error"]),
        "Y Rotation max error": max(errors_dict["Y_rot_error"]),
        "Y Rotation mean error": np.mean(errors_dict["Y_rot_error"]),
        "Y Rotation SD error": np.std(errors_dict["Y_rot_error"]),
        "Z Rotation max error": max(errors_dict["Z_rot_error"]),
        "Z Rotation mean error": np.mean(errors_dict["Z_rot_error"]),
        "Z Rotation SD error": np.mean(errors_dict["Z_rot_error"]),
        "X reconstr": list(reconstr_traj.X),
        "Y reconstr": list(reconstr_traj.Y),
        "Z reconstr": list(reconstr_traj.Z),
        "X theoretical": list(ideal_traj.X),
        "Y theoretical": list(ideal_traj.Y),
        "Z theoretical": list(ideal_traj.Z),
        "Real length": list(errors_dict["length_rec"]),
        "X rot angle": list(reconstr_traj.X_rot),
        "Y rot angle": list(reconstr_traj.Y_rot),
        "Z rot angle": list(reconstr_traj.Z_rot),
        "X theoretical rot": list(ideal_traj.X_rot),
        "Y theoretical rot": list(ideal_traj.Y_rot),
        "Z theoretical rot": list(ideal_traj.Z_rot),
        "Final length": (errors_dict["length_rec"] - errors_dict["length_ideal"])[
            len(errors_dict["length_rec"] - errors_dict["length_ideal"]) - 1],
        "Mean diff. real-reconstr": np.mean(np.abs(errors_dict["length_rec"] - errors_dict["length_ideal"])),
        "Max diff. real-reconstr": np.max(np.abs(errors_dict["length_rec"] - errors_dict["length_ideal"])),
        "SD diff. real-reconstr": np.std(np.abs(errors_dict["length_rec"] - errors_dict["length_ideal"])),
        "Diff. real-reconstr": list(errors_dict["length_rec"] - errors_dict["length_ideal"]),
        "Gradient (Step diff.) real-reconstr": list(errors_dict["step_length_diff"])
    }
    with open(json_filename_path + ".json", 'w') as outfile:
        obj = json.dump(json_file_text, outfile, indent=4)


def write_traj_file_transl_quad(traj, start_time, file_path_and_name):
    '''
    This function write to .txt files the trajectories for computing ATE and RPE indicators
    '''
    traj.compute_quat_from_angles()
    time_stamp = np.array(np.arange(len(traj.X)) + start_time)
    array_to_write = np.concatenate([np.expand_dims(time_stamp, axis=1),
                                     np.expand_dims(traj.X[0:len(time_stamp)],
                                                    axis=1), np.expand_dims(traj.Y[0:len(time_stamp)], axis=1),
                                     np.expand_dims(traj.Z[0:len(time_stamp)], axis=1),
                                     traj.quat[0:len(time_stamp), :]], axis=1)
    np.savetxt(file_path_and_name, array_to_write)


def from_indicators_to_matrix_data(file_excel, parameters_to_eval, columns_values, rows_values, error_matrix_res_file):
    '''
    This function reads values of a certain indicator in the file specified
    by "file_excel" path. These values are rearranged as 2D arrays basing on column_values and row_values variables.
    This 2D array is saved to another excel file with function create_matrix_data_to_excel
    '''
    df = pd.read_excel(file_excel, index_col=0)
    for j, param in enumerate(parameters_to_eval):
        dataset_3D_errors = np.array(df.loc[param].values.tolist())
        if j == 0:
            create_matrix_data_to_excel(dataset_3D_errors, columns_values, rows_values, error_matrix_res_file, True,
                                        param)
        else:
            create_matrix_data_to_excel(dataset_3D_errors, columns_values, rows_values, error_matrix_res_file, False,
                                        param)


def create_matrix_data_to_excel(data, columns_values, rows_values, error_matrix_res_file, new_file_flag, sheet_name):
    '''
    This function creates a 2D array basing on the values read in "data". These values are rearranged
    as 2D arrays basing on column_values and row_values variables.
    This 2D array is saved to an excel file.
    '''
    columns_values_idx = np.unique(columns_values)
    rows_values_idx = np.unique(rows_values)
    matrix_values = np.zeros((len(rows_values_idx), len(columns_values_idx)))
    for i, error in enumerate(data):
        row = np.where(rows_values_idx == rows_values[i])
        col = np.where(columns_values_idx == columns_values[i])
        matrix_values[row, col] = error
    if new_file_flag:
        writer = pd.ExcelWriter(error_matrix_res_file, engine='openpyxl')
    else:
        writer = pd.ExcelWriter(error_matrix_res_file, engine='openpyxl', mode="a")
    df_to_write = pd.DataFrame(matrix_values)
    df_to_write.columns = [columns_values_idx]
    df_to_write.index = [rows_values_idx]
    df_to_write.to_excel(writer, sheet_name=sheet_name)
    writer.save()


def RPE_data_plot_and_saving(result_delta_1, result_delta_10_perc, ATE_seq_res_dir, file_quat, df_tot_RPE, n_fragments
                             , res_fig_dir_dataset):
    trans_error_delta_1 = np.array(result_delta_1)[:, 4]
    rot_error_delta_1 = np.array(result_delta_1)[:, 5]
    trans_error_delta_10_perc = np.array(result_delta_10_perc)[:, 4]
    rot_error_delta_10_perc = np.array(result_delta_10_perc)[:, 5]
    json_text_file = {
        "RPE_transl_delta_1:": list(trans_error_delta_1),
        "RPE_transl_delta_10_perc:": list(trans_error_delta_10_perc),
        "RPE_rot_delta_1:": list(rot_error_delta_1 * 180 / np.pi),
        "RPE_rot_delta_10_perc:": list(rot_error_delta_10_perc * 180 / np.pi)
    }
    with open(ATE_seq_res_dir + "\\" + file_quat.split(".txt")[0] + "RPE.json", "w") as f:
        obj = json.dump(json_text_file, f, indent=4)
    df = pd.Series({"Name": file_quat.split(".")[0],
                    "Pose pairs delta 1": len(trans_error_delta_1),
                    "Pose pairs delta 10 perc": len(trans_error_delta_10_perc),
                    "RMSE RPE transl delta 1": np.sqrt(
                        np.dot(trans_error_delta_1, trans_error_delta_1) / len(trans_error_delta_1)),
                    "Mean RPE transl delta 1": np.mean(trans_error_delta_1),
                    "Median RPE transl delta 1": np.median(trans_error_delta_1),
                    "Std dev RPE transl delta 1": np.std(trans_error_delta_1),
                    "Min RPE transl delta 1": np.min(trans_error_delta_1),
                    "Max RPE transl delta 1": np.max(trans_error_delta_1),
                    "RMSE RPE transl delta 10 perc": np.sqrt(
                        np.dot(trans_error_delta_10_perc, trans_error_delta_10_perc) / len(trans_error_delta_10_perc)),
                    "Mean RPE transl delta 10 perc": np.mean(trans_error_delta_10_perc),
                    "Median RPE transl delta 10 perc": np.median(trans_error_delta_10_perc),
                    "Std dev RPE transl delta 10 perc": np.std(trans_error_delta_10_perc),
                    "Min RPE transl delta 10 perc": np.min(trans_error_delta_10_perc),
                    "Max RPE transl delta 10 perc": np.max(trans_error_delta_10_perc),
                    "RMSE RPE rot delta 1": np.sqrt(
                        np.dot(rot_error_delta_1, rot_error_delta_1) / len(rot_error_delta_1)) * 180 / np.pi,
                    "Mean RPE rot delta 1": np.mean(rot_error_delta_1) * 180 / np.pi,
                    "Median RPE rot delta 1": np.median(rot_error_delta_1) * 180 / np.pi,
                    "Std dev RPE rot delta 1": np.std(rot_error_delta_1) * 180 / np.pi,
                    "Min RPE rot delta 1": np.min(rot_error_delta_1) * 180 / np.pi,
                    "Max RPE rot delta 1": np.max(rot_error_delta_1) * 180 / np.pi,
                    "RMSE RPE rot delta 10 perc": np.sqrt(
                        np.dot(rot_error_delta_10_perc, rot_error_delta_10_perc) / len(
                            rot_error_delta_10_perc)) * 180 / np.pi,
                    "Mean RPE rot delta 10 perc": np.mean(rot_error_delta_10_perc) * 180 / np.pi,
                    "Median RPE rot delta 10 perc": np.median(rot_error_delta_10_perc) * 180 / np.pi,
                    "Std dev RPE rot delta 10 perc": np.std(rot_error_delta_10_perc) * 180 / np.pi,
                    "Min RPE rot delta 10 perc": np.min(rot_error_delta_10_perc) * 180 / np.pi,
                    "Max RPE rot delta 10 perc": np.max(rot_error_delta_10_perc) * 180 / np.pi
                    })
    df_tot_RPE = pd.concat([df_tot_RPE, df], axis=1)
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    peaks, properties = sign.find_peaks(trans_error_delta_1, height=np.mean(trans_error_delta_1))
    indices = np.argsort(properties['peak_heights'])
    peaks = peaks[indices]
    markers_on = list(peaks[len(peaks) - n_fragments + 1:])
    ax.plot(np.arange(len(trans_error_delta_1)) / len(trans_error_delta_1), trans_error_delta_1, '-',
            color="b", markevery=markers_on, marker='x', markeredgecolor='red')
    # ax.plot([t for t,e in err_rot],[e for t,e in err_rot],'-',color="red")
    ax.set_xlabel('reconstruction percentage [%]')
    ax.set_ylabel('translational error [m]')
    ax.set_title(r"Translational error with $\Delta$ = 1")
    plt.grid()
    plt.savefig(res_fig_dir_dataset + "\\trans_error_delta_1.png")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot((np.arange(len(trans_error_delta_1)) / len(trans_error_delta_1))[0:len(trans_error_delta_10_perc)],
            trans_error_delta_10_perc, '-', color="blue")
    # ax.plot([t for t,e in err_rot],[e for t,e in err_rot],'-',color="red")
    ax.set_xlabel('reconstruction percentage [%]')
    ax.set_ylabel('translational error [m]')
    ax.set_title(r"Translational error with $\Delta$ = 10 %")
    plt.grid()
    plt.savefig(res_fig_dir_dataset + "\\trans_error_delta_10_perc.png")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    peaks, properties = sign.find_peaks(rot_error_delta_1, height=np.mean(rot_error_delta_1))
    indices = np.argsort(properties['peak_heights'])
    peaks = peaks[indices]
    markers_on = list(peaks[len(peaks) - n_fragments + 1:])
    ax.plot(np.arange(len(rot_error_delta_1)) / len(rot_error_delta_1), rot_error_delta_1 * 180 / np.pi, '-',
            color="blue"
            , markevery=markers_on, marker='x', markeredgecolor='red')
    # ax.plot([t for t,e in err_rot],[e for t,e in err_rot],'-',color="red")
    ax.set_xlabel('reconstruction percentage [%]')
    ax.set_ylabel('rotational error [°]')
    ax.set_title(r"Rotational error with $\Delta$ = 1")
    plt.grid()
    plt.savefig(res_fig_dir_dataset + "\\rot_error_delta_1.png")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot((np.arange(len(rot_error_delta_1)) / len(rot_error_delta_1))[0:len(rot_error_delta_10_perc)],
            rot_error_delta_10_perc * 180 / np.pi, '-', color="blue")
    # ax.plot([t for t,e in err_rot],[e for t,e in err_rot],'-',color="red")
    ax.set_xlabel('reconstruction percentage [%]')
    ax.set_ylabel('rotational error [°]')
    ax.set_title(r"Rotational error with $\Delta$ = 10 %")
    plt.grid()
    plt.savefig(res_fig_dir_dataset + "\\rot_error_delta_10_perc.png")
    plt.close()
    return df_tot_RPE