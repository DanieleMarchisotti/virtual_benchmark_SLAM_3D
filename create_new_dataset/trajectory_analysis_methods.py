import numpy as np
import matplotlib.pyplot as plt
import os
import transformations
import json
import pandas as pd
import scipy.signal as sign
import time


class Traj(object):
    def __init__(self,analysis_type,n_elem,tot_angle, reconstr_sys):
        self.analysis_type=analysis_type
        self.n_elem_traj=n_elem
        self.tot_angle=tot_angle
        self.reconstr_sys = reconstr_sys
        self.X=np.array([])
        self.Y=np.array([])
        self.Z = np.array([])
        self.X_rot=np.array([])
        self.Y_rot=np.array([])
        self.Z_rot = np.array([])
        self.traj_length=np.array([])
        self.rotation_angle=np.array([])
        self.quat=np.array([])
        self.poses = np.array([])

    def read_open3d_traj(self,trajectory_file):
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

    def read_elasticfusion_traj(self,trajectory_file):
        with open(trajectory_file, "r") as f:
            file_content = f.read()
        nums = file_content.split(" ")
        data = np.array([], dtype=np.float64)
        for i, num in enumerate(nums):
            curr_nums = [float(item) for item in num.split("\n") if not item == ""]
            data = np.append(data, curr_nums)
        data = data.reshape(int(len(data) / 8), 8)
        self.X=np.array(data[:,1])
        self.Y=np.array(data[:,2])
        self.Z = np.array(data[:, 3])
        angles=np.array([transformations.euler_from_quaternion(pose[4:]) for pose in data])
        self.X_rot = angles[:, 0] * 180 / np.pi
        self.Y_rot = angles[:, 1] * 180 / np.pi
        self.Z_rot = angles[:, 2] * 180 / np.pi

    def read_traj_input(self, trajectory_file):
        with open(trajectory_file, "r") as f:
            file_content = f.read()
        nums = file_content.split(" ")
        data = np.array([], dtype=np.float64)
        for i, num in enumerate(nums):
            curr_nums = [float(item) for item in num.split("\n") if not item == ""]
            data = np.append(data, curr_nums)
        self.X = data[0:len(data):6]
        self.X_rot = data[1:len(data):6]
        self.Y = data[2:len(data):6]
        self.Y_rot = data[3:len(data):6]
        self.Z = data[4:len(data):6]
        self.Z_rot = data[5:len(data):6]
        self.n_elem_traj = len(self.X)

    def generate_observer_circular_traj(self, R, center_point):
        step_theta = self.tot_angle / self.n_elem_traj
        eps = 1.1102230246251565e-16
        angles = np.arange(0, self.tot_angle + 10 ** 3 * eps, step_theta)
        self.X = R * np.sin(angles * np.pi / 180) + center_point[0]
        self.Y = R * np.cos(angles * np.pi / 180) + center_point[1]
        self.Z = np.zeros(self.X.shape)
        self.X_rot = np.zeros(angles.shape)
        self.Y_rot = np.zeros(angles.shape)
        self.Z_rot = np.zeros(angles.shape)

    def generate_observer_linear_and_rot_traj_with_noise(self, n_elem_transl_x, n_elem_transl_y, n_elem_rot, transl_dim_x_top, transl_dim_x_bottom, transl_dim_y_left, transl_dim_y_right, noise_dir, init_pos):
        self.Z = np.zeros(int(n_elem_transl_x / 2) + int(n_elem_rot / 4) + 1)
        self.Z = np.append(self.Z, np.linspace(0, -transl_dim_y_right, int(n_elem_transl_y / 2)))
        self.Z = np.append(self.Z, -transl_dim_y_right * np.ones(int(n_elem_rot / 2) + int(n_elem_transl_x / 2)))
        self.Z = np.append(self.Z, np.linspace(-transl_dim_y_right, 0, int(n_elem_transl_y / 2)))
        self.Z = np.append(self.Z, np.zeros(int(n_elem_rot / 4)))
        self.X = np.linspace(0, transl_dim_x_bottom + 1, int((n_elem_transl_x + 2) / 2))
        self.X = np.append(self.X, np.ones(int(n_elem_rot / 2 + n_elem_transl_y / 2)) * (transl_dim_x_bottom + 1))
        self.X = np.append(self.X, transl_dim_x_bottom + 1 - np.linspace(int((transl_dim_x_top / (n_elem_transl_x / 2))), transl_dim_x_top, int(n_elem_transl_x / 2)))
        self.X = np.append(self.X, (transl_dim_x_bottom + 1 - transl_dim_x_top) * np.ones(int(n_elem_rot / 2 + n_elem_transl_y / 2)) * 1)
        self.Y = +0.3 * np.ones(int(n_elem_transl_x/2 + 1))
        self.Y = np.append(self.Y, np.linspace(0.3, 0, int(n_elem_rot / 2) + int(n_elem_transl_y / 2)))
        self.Y = np.append(self.Y, np.zeros(int(n_elem_rot / 2) + int(n_elem_transl_x / 2) + int(n_elem_transl_y / 2)))
        step_angle = 180 / (n_elem_rot / 2)
        self.Y_rot = np.zeros(int((n_elem_transl_x + 2) / 2))
        self.Y_rot = np.append(self.Y_rot, np.linspace(step_angle, 90, int(n_elem_rot / 4)))
        self.Y_rot = np.append(self.Y_rot, 90 * np.ones(int(n_elem_transl_y / 2)))
        self.Y_rot = np.append(self.Y_rot, np.linspace(step_angle + 90, 180, int(n_elem_rot / 4)))
        self.Y_rot = np.append(self.Y_rot, 180 * np.ones(int(n_elem_transl_x / 2)))
        self.Y_rot = np.append(self.Y_rot, np.linspace(180 + step_angle, 270, int(n_elem_rot / 4)))
        self.Y_rot = np.append(self.Y_rot, 270 * np.ones(int(n_elem_transl_y / 2)))
        self.Y_rot = np.append(self.Y_rot, np.linspace(270 + step_angle, 360, int(n_elem_rot / 4)))
        self.X_rot = np.zeros(n_elem_transl_x + n_elem_transl_y + n_elem_rot + 1)
        self.Z_rot = np.zeros(n_elem_transl_x + n_elem_transl_y + n_elem_rot + 1)
        # self.Z = np.zeros(n_elem_transl + n_elem_rot + 1)
        # self.X = np.linspace(0, transl_dim, int((n_elem_transl + 2) / 2))
        # self.X = np.append(self.X, np.ones(int(n_elem_rot / 2)) * transl_dim)
        # self.X = np.append(self.X, transl_dim - np.linspace(int((transl_dim / (n_elem_transl / 2))), transl_dim, int(n_elem_transl / 2)))
        # self.X = np.append(self.X, np.zeros(int(n_elem_rot / 2)))
        # self.Y = +0.3 * np.ones(int(n_elem_transl/2 + 1))
        # self.Y = np.append(self.Y, np.linspace(0.3, 0, int(n_elem_rot / 2)))
        # self.Y = np.append(self.Y, np.zeros(int(n_elem_rot / 2) + int(n_elem_transl / 2)))
        # step_angle = 180 / (n_elem_rot / 2)
        # self.Y_rot = np.zeros(int((n_elem_transl + 2) / 2))
        # self.Y_rot = np.append(self.Y_rot, np.linspace(step_angle, 180, int(n_elem_rot / 2)))
        # self.Y_rot = np.append(self.Y_rot, 180 * np.ones(int(n_elem_transl / 2)))
        # self.Y_rot = np.append(self.Y_rot, np.linspace(180 + step_angle, 360, int(n_elem_rot / 2)))
        # self.X_rot = np.zeros(n_elem_transl + n_elem_rot + 1)
        # self.Z_rot = np.zeros(n_elem_transl + n_elem_rot + 1)
        self.apply_handheld_noise(noise_dir)
        # print("Init pos: ", init_pos)
        self.X -= init_pos[0]
        self.Y -= init_pos[1]
        self.Z -= init_pos[2]
        self.X_rot -= init_pos[3]
        self.Y_rot -= init_pos[4]
        self.Z_rot -= init_pos[5]

    def ideal_circular_traj(self,R,center_point):
        step_theta = self.tot_angle / self.n_elem_traj
        eps = 1.1102230246251565e-16
        angles = np.arange(0, self.tot_angle + 10 ** 3 * eps, step_theta)
        self.X = R * np.sin(angles * np.pi / 180) + center_point[0]
        self.Z = R * np.cos(angles * np.pi / 180) + center_point[1]
        self.Y = np.zeros(self.X.shape)
        self.X_rot = np.zeros(angles.shape)
        self.Y_rot = np.zeros(angles.shape)
        self.Z_rot = np.zeros(angles.shape)

    def evaluate_circular_traj_length(self,R):
        self.traj_length=np.linspace(0, 2 * np.pi * R, self.n_elem_traj + 1)

    def delete_360_degrees_phase_shifts(self):
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

    def generate_angle_rot_phase_fun(self,X_shift_idx, Z_shift_idx, delta_rot):
        Y_shift_idx = np.int32(np.array([]))
        for X_idx in X_shift_idx:
            X_idx_low = X_idx - delta_rot / 2
            X_idx_high = X_idx + delta_rot / 2
            for Z_idx in Z_shift_idx:
                if X_idx_low < Z_idx < X_idx_high:
                    Y_shift_idx = np.append(Y_shift_idx, (X_idx + Z_idx) / 2)
        Y_shift_idx = np.int32(Y_shift_idx)
        # Y_shift_idx=np.int32((X_shift_idx+Z_shift_idx)/2)
        Y_rot_ideal = np.zeros(self.X_rot.shape)
        delta_angle = -self.tot_angle / self.n_elem_traj
        for i in range(1, len(Y_rot_ideal)):
            Y_rot_ideal[i] = Y_rot_ideal[i - 1] + delta_angle
            if i in Y_shift_idx:
                delta_angle = -delta_angle
        return Y_rot_ideal

    def ideal_rotation_traj(self,X_rot,Z_rot):
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
        self.Y_rot = np.arange(0, self.tot_angle + 10 ** 3 * eps, self.tot_angle/self.n_elem_traj)
        self.Z_rot = np.zeros(self.n_elem_traj + 1)

    def generate_observer_orbit_traj_no_shifts(self,R,pitch_angle):
        theta = np.linspace(0, self.tot_angle, self.n_elem_traj + 1)
        self.X = R* np.cos(theta* np.pi / 180)-R
        self.Y = R* np.sin(theta* np.pi / 180)
        self.Z = np.zeros(len(self.X))
        self.X_rot = -theta
        self.Y_rot = pitch_angle * np.sin(theta* np.pi / 180)
        self.Z_rot = -pitch_angle * np.cos(theta* np.pi / 180)

    def generate_local_orbit_traj(self,R,pitch_angle):
        theta = np.linspace(0, self.tot_angle, self.n_elem_traj + 1)
        self.X = -R * np.sin(theta * np.pi / 180)
        self.Y = (R * np.cos(theta * np.pi / 180)-R)*np.sin(pitch_angle*np.pi/180)
        self.Z = (-R * np.cos(theta * np.pi / 180)+R)*np.cos(pitch_angle*np.pi/180)
        self.X_rot = (2*pitch_angle)*np.cos(theta * np.pi / 180)-(2*pitch_angle)+pitch_angle*(np.sin(theta*np.pi/180))**2*np.cos(theta*np.pi/180)
        self.Y_rot = 2*pitch_angle*np.sin(theta * np.pi / 180)-18*np.sin(theta*np.pi/180)*(np.cos(theta*np.pi/180))**2
        self.Z_rot = 90-90*np.cos(theta[0:181] * np.pi / 180)
        self.Z_rot = np.append(self.Z_rot,90+180-90*np.cos(theta[1:181] * np.pi / 180))

    def evaluate_rotation_angle_traj(self):
        self.rotation_angle=np.arange(0, 1, 1/len(self.X_rot)) * self.tot_angle

    def read_open3d_traj_to_transl_and_quaternion(self,trajectory_file):
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
        self.quat=np.array(
            [transformations.quaternion_from_euler(X_angle / 180 * np.pi, Y_angle / 180 * np.pi, Z_angle / 180 * np.pi)
             for X_angle, Y_angle, Z_angle in zip(self.X_rot, self.Y_rot, self.Z_rot)])

    def ideal_linear_and_rot_traj(self, n_elem_transl_x, n_elem_transl_y, n_elem_rot, transl_dim_x_top, transl_dim_x_bottom, transl_dim_y_left, transl_dim_y_right):
        self.Z = np.zeros(int(n_elem_transl_x / 2) + int(n_elem_rot / 4) + 1)
        self.Z = np.append(self.Z, np.linspace(0, -transl_dim_y_right, int(n_elem_transl_y / 2)))
        self.Z = np.append(self.Z, -transl_dim_y_right * np.ones(int(n_elem_rot / 2) + int(n_elem_transl_x / 2)))
        self.Z = np.append(self.Z, np.linspace(-transl_dim_y_right, 0, int(n_elem_transl_y / 2)))
        self.Z = np.append(self.Z, np.zeros(int(n_elem_rot / 4)))
        self.X = np.linspace(0, transl_dim_x_bottom + 1, int((n_elem_transl_x + 2) / 2))
        self.X = np.append(self.X, np.ones(int(n_elem_rot / 2 + n_elem_transl_y / 2)) * (transl_dim_x_bottom + 1))
        self.X = np.append(self.X, transl_dim_x_bottom + 1 - np.linspace(int((transl_dim_x_top / (n_elem_transl_x / 2))), transl_dim_x_top, int(n_elem_transl_x / 2)))
        self.X = np.append(self.X, (transl_dim_x_bottom + 1 - transl_dim_x_top) * np.ones(int(n_elem_rot / 2 + n_elem_transl_y / 2)) * 1)
        self.Y = +0.3 * np.ones(int(n_elem_transl_x/2 + 1))
        self.Y = np.append(self.Y, np.linspace(0.3, 0, int(n_elem_rot / 2) + int(n_elem_transl_y / 2)))
        self.Y = np.append(self.Y, np.zeros(int(n_elem_rot / 2) + int(n_elem_transl_x / 2) + int(n_elem_transl_y / 2)))
        step_angle = 180 / (n_elem_rot / 2)
        self.Y_rot = np.zeros(int((n_elem_transl_x + 2) / 2))
        self.Y_rot = np.append(self.Y_rot, np.linspace(step_angle, 90, int(n_elem_rot / 4)))
        self.Y_rot = np.append(self.Y_rot, 90 * np.ones(int(n_elem_transl_y / 2)))
        self.Y_rot = np.append(self.Y_rot, np.linspace(step_angle + 90, 180, int(n_elem_rot / 4)))
        self.Y_rot = np.append(self.Y_rot, 180 * np.ones(int(n_elem_transl_x / 2)))
        self.Y_rot = np.append(self.Y_rot, np.linspace(180 + step_angle, 270, int(n_elem_rot / 4)))
        self.Y_rot = np.append(self.Y_rot, 270 * np.ones(int(n_elem_transl_y / 2)))
        self.Y_rot = np.append(self.Y_rot, np.linspace(270 + step_angle, 360, int(n_elem_rot / 4)))
        self.X_rot = np.zeros(n_elem_transl_x + n_elem_transl_y + n_elem_rot + 1)
        self.Z_rot = np.zeros(n_elem_transl_x + n_elem_transl_y + n_elem_rot + 1)

    def apply_handheld_noise(self, noise_dir):
        self.X += np.loadtxt(noise_dir + "\\x_trasl_hist.txt")[0:len(self.X)]
        self.Y += np.loadtxt(noise_dir + "\\y_trasl_hist.txt")[0:len(self.X)]
        self.Z += np.loadtxt(noise_dir + "\\z_trasl_hist.txt")[0:len(self.X)]
        self.X_rot += np.loadtxt(noise_dir + "\\x_rot_hist.txt")[0:len(self.X)]
        self.Y_rot += np.loadtxt(noise_dir + "\\z_rot_hist.txt")[0:len(self.X)]
        self.Z_rot += np.loadtxt(noise_dir + "\\y_rot_hist.txt")[0:len(self.X)]

    def evaluate_transl_and_rot_length(self):
        self.traj_length = np.array(np.sqrt(self.X**2 + self.Y**2 + self.Z**2))

    def write_open3d_traj_log_file(self, trajectory_file):
        with open(trajectory_file, 'w') as f:
            for i, pose in enumerate(self.poses):
                f.write('{} {} {}\n'.format(i, i, i + 1))
                f.write('{0:.8f} {1:.8f} {2:.8f} {3:.8f}\n'.format(
                    pose[0, 0], pose[0, 1], pose[0, 2], pose[0, 3]))
                f.write('{0:.8f} {1:.8f} {2:.8f} {3:.8f}\n'.format(
                    pose[1, 0], pose[1, 1], pose[1, 2], pose[1, 3]))
                f.write('{0:.8f} {1:.8f} {2:.8f} {3:.8f}\n'.format(
                    pose[2, 0], pose[2, 1], pose[2, 2], pose[2, 3]))
                f.write('{0:.8f} {1:.8f} {2:.8f} {3:.8f}\n'.format(
                    pose[3, 0], pose[3, 1], pose[3, 2], pose[3, 3]))

    def crate_poses_from_transl_and_rot(self):
        for index in range(len(self.X)):
            x_rot_matrix = np.array(
                [[np.cos(self.Y_rot[index] * np.pi / 180), -np.sin(self.Y_rot[index] * np.pi / 180), 0],
                 [np.sin(self.Y_rot[index] * np.pi / 180), np.cos(self.Y_rot[index] * np.pi / 180), 0],
                 [0, 0, 1]])
            y_rot_matrix = np.array(
                [[np.cos(self.Z_rot[index] * np.pi / 180), 0, -np.sin(self.Z_rot[index] * np.pi / 180)],
                 [0, 1, 0],
                 [np.sin(self.Z_rot[index] * np.pi / 180), 0, np.cos(self.Z_rot[index] * np.pi / 180)]])
            z_rot_matrix = np.array(
                [[1, 0, 0],
                 [0, np.cos(self.X_rot[index] * np.pi / 180), -np.sin(self.X_rot[index] * np.pi / 180)],
                 [0, np.sin(self.X_rot[index] * np.pi / 180), np.cos(self.X_rot[index] * np.pi / 180)]])
            rot_matrix = x_rot_matrix.dot(y_rot_matrix)
            rot_matrix = rot_matrix.dot(z_rot_matrix)
            transl = np.array([self.X[index], self.Y[index], self.Z[index]])
            transl = np.expand_dims(transl, axis=1)
            pose = np.concatenate((rot_matrix, transl), axis=1)
            last_row = np.array([0, 0, 0, 1])
            last_row = np.expand_dims(last_row, axis=0)
            pose = np.concatenate((pose, last_row), axis=0)
            self.poses = np.append(self.poses, pose)


def evaluate_general_traj_errors(reconstr_traj_file, analysis_type, n_elem_traj, tot_angle, reconstr_sys, R=0, center_point=None,
                                 pitch_angle=None, n_elem_transl_x=None, n_elem_transl_y=None, n_elem_rot=None,
                                 transl_dim_x_top=None, transl_dim_x_bottom=None, transl_dim_y_left=None,
                                 transl_dim_y_right=None, init_pos=None, noise_dir=None):
    reconstr_traj = Traj(analysis_type, n_elem_traj, tot_angle, reconstr_sys)
    if reconstr_traj.reconstr_sys=="ElasticFusion":
        reconstr_traj.read_elasticfusion_traj(reconstr_traj_file)
    elif reconstr_traj.reconstr_sys=="open3d":
        reconstr_traj.read_open3d_traj(reconstr_traj_file)
    else:
        raise Exception("Reconstruction system not specified or not included in analysis")
    # reconstr_traj.read_open3d_traj(reconstr_traj_file)
    ideal_traj = Traj(analysis_type, n_elem_traj, tot_angle, reconstr_sys)
    if analysis_type == "transl":
        ideal_traj.ideal_circular_traj(R,center_point)
    elif analysis_type=="rot":
        reconstr_traj.delete_360_degrees_phase_shifts()
        ideal_traj.ideal_rotation_traj(reconstr_traj.X_rot,reconstr_traj.Z_rot)
    elif analysis_type=="orbit":
        reconstr_traj.delete_360_degrees_phase_shifts()
        ideal_traj.generate_local_orbit_traj(R,pitch_angle)
    elif analysis_type == "transl_and_rot":
        ideal_traj.generate_observer_linear_and_rot_traj_with_noise(n_elem_transl_x, n_elem_transl_y, n_elem_rot,
                                                                    transl_dim_x_top, transl_dim_x_bottom,
                                                                    transl_dim_y_left, transl_dim_y_right, noise_dir, init_pos)
        ideal_traj.X = -ideal_traj.X
        ideal_traj.Z = -ideal_traj.Z
    else:
        raise Exception("The trajectory analysis type is not defined")
    if reconstr_sys == "ElasticFusion":
        ideal_traj.X = ideal_traj.X[:len(ideal_traj.X) - 1]
        ideal_traj.Y = ideal_traj.Y[:len(ideal_traj.Y) - 1]
        ideal_traj.Z = ideal_traj.Z[:len(ideal_traj.Z) - 1]
        ideal_traj.X_rot = ideal_traj.X_rot[:len(ideal_traj.X_rot) - 1]
        ideal_traj.Y_rot = ideal_traj.Y_rot[:len(ideal_traj.Y_rot) - 1]
        ideal_traj.Z_rot = ideal_traj.Z_rot[:len(ideal_traj.Z_rot) - 1]
    if analysis_type == "transl":
        ideal_traj.evaluate_circular_traj_length(R)
    elif analysis_type=="rot":
        reconstr_traj.evaluate_rotation_angle_traj()
    elif analysis_type=="orbit":
        reconstr_traj.evaluate_rotation_angle_traj()
    elif analysis_type == "transl_and_rot":
        ideal_traj.evaluate_transl_and_rot_length()
    else:
        raise Exception("The trajectory analysis type is not defined")

    # error along different directions and planes
    X_error=np.abs(reconstr_traj.X - ideal_traj.X[0:reconstr_traj.X.shape[0]])
    Y_error=np.abs(reconstr_traj.Y - ideal_traj.Y[0:reconstr_traj.Y.shape[0]])
    Z_error=np.abs(reconstr_traj.Z - ideal_traj.Z[0:reconstr_traj.Z.shape[0]])
    errors_dict={
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
    length_rec_diff = np.array(
        [np.linalg.norm(np.array([reconstr_traj.X[i], reconstr_traj.Y[i], reconstr_traj.Z[i]]) -
                        np.array([reconstr_traj.X[i - 1], reconstr_traj.Y[i - 1], reconstr_traj.Z[i - 1]]))
         for i in range(1, len(reconstr_traj.Z))])
    length_rec = np.array([0])
    for i in range(len(length_rec_diff)):
        length_rec = np.append(length_rec, [length_rec[i] + length_rec_diff[i]], axis=0)
    length_ideal_diff = np.array([np.linalg.norm(np.array([ideal_traj.X[i], ideal_traj.Y[i], ideal_traj.Z[i]])
                                                 - np.array([ideal_traj.X[i - 1], ideal_traj.Y[i - 1], ideal_traj.Z[i - 1]]))
                                  for i in range(1, len(reconstr_traj.Z))])
    length_ideal = np.array([0])
    for i in range(len(length_ideal_diff)):
        length_ideal = np.append(length_ideal, [length_ideal[i] + length_ideal_diff[i]], axis=0)
    if analysis_type == "rot" or analysis_type == "orbit":
        step_length_diff = np.gradient(length_rec - length_ideal, reconstr_traj.rotation_angle[1] - reconstr_traj.rotation_angle[0])
        step_length_diff_1 = np.array([(length_rec[i] - length_ideal[i]) - (length_rec[i - 1] - length_ideal[i - 1])
                                       for i in range(1, len(length_rec))])
    elif analysis_type == "transl":
        step_length_diff = np.gradient(length_rec - length_ideal, ideal_traj.traj_length[1] - ideal_traj.traj_length[0])
        step_length_diff_1 = np.array([(length_rec[i] - length_ideal[i]) - (length_rec[i - 1] - length_ideal[i - 1])
                                       for i in range(1, len(length_rec))])
    elif analysis_type == "transl_and_rot":
        step_length_diff = np.gradient(length_rec - length_ideal, ideal_traj.traj_length[1] - ideal_traj.traj_length[0])
        step_length_diff_1 = np.array([(length_rec[i] - length_ideal[i]) - (length_rec[i - 1] - length_ideal[i - 1])
                                       for i in range(1, len(length_rec))])
    else:
        raise Exception("The trajectory analysis type is not defined")
    errors_dict["length_rec"]=length_rec
    errors_dict["length_ideal"] = length_ideal
    errors_dict["step_length_diff"] = step_length_diff
    errors_dict["step_length_diff_1"] = step_length_diff_1
    return errors_dict,reconstr_traj,ideal_traj


def plot_3d_traj_projections(reconstr_traj,ideal_traj,plot_filedir):
    fig, ax = plt.subplots(nrows=1, ncols=3,figsize=(15,5),constrained_layout=True)
    ax[0].plot(reconstr_traj.X,reconstr_traj.Z) # ,marker='o',markevery=markers_on)
    ax[0].plot(ideal_traj.X,ideal_traj.Z,linewidth=0.5)
    ax[0].grid()
    ax[0].set_xlabel("X [m]")
    ax[0].set_ylabel("Z [m]")
    ax[0].set_title("X-Z plane trajectory")
    ax[0].legend(["Reconstruction trajectory","Theoretical trajectory"])
    fig.suptitle("Trajectory of 3D sensor from reconstruction")

    ax[1].plot(reconstr_traj.X,reconstr_traj.Y)
    ax[1].plot(ideal_traj.X, ideal_traj.Y, linewidth=0.5)
    ax[1].set_xlabel("X [m]")
    ax[1].set_ylabel("Y [m]")
    ax[1].set_title("X-Y plane trajectory")
    ax[1].legend(["Reconstruction trajectory","Theoretical trajectory"])
    ax[1].grid()

    ax[2].plot(reconstr_traj.Z,reconstr_traj.Y)
    ax[2].plot(ideal_traj.Z, ideal_traj.Y, linewidth=0.5)
    ax[2].set_xlabel("Z [m]")
    ax[2].set_ylabel("Y [m]")
    ax[2].set_title("Z-Y plane trajectory")
    ax[2].legend(["Reconstruction trajectory","Theoretical trajectory"])
    ax[2].grid()
    plt.savefig(plot_filedir+"\\displacements.png")
    plt.close()


def plot_camera_orientation_angles(analysis_type,reconstr_traj,ideal_traj,plot_filedir):
    fig1, ax1 = plt.subplots(nrows=3, ncols=1, figsize=(5, 8), constrained_layout=True)
    if analysis_type=="rot" or analysis_type=="orbit":
        ax1[0].plot(reconstr_traj.rotation_angle,reconstr_traj.X_rot)
        ax1[0].plot(reconstr_traj.rotation_angle,ideal_traj.X_rot,linewidth=0.75)
        ax1[0].set_xlabel("Theoretical rotation angle around Y [°]")
    else:
        ax1[0].plot(ideal_traj.traj_length,reconstr_traj.X_rot)
        ax1[0].plot(ideal_traj.traj_length,ideal_traj.X_rot,linewidth=0.75)
        ax1[0].set_xlabel("Theoretical trajectory length [m]")
    ax1[0].grid()
    ax1[0].set_ylabel("Rotation around X axis [°]")
    ax1[0].set_title("Rotation around X axis")
    ax1[0].legend(["Reconstruction rotation","Theoretical rotation"])
    fig1.suptitle("Rotation angles of 3D sensor from reconstruction")

    if analysis_type=="rot" or analysis_type=="orbit":
        ax1[1].plot(reconstr_traj.rotation_angle,reconstr_traj.Y_rot)
        ax1[1].plot(reconstr_traj.rotation_angle,ideal_traj.Y_rot,linewidth=0.75)
        ax1[1].set_xlabel("Theoretical rotation angle around Y [°]")
    else:
        ax1[1].plot(ideal_traj.traj_length,reconstr_traj.Y_rot)
        ax1[1].plot(ideal_traj.traj_length,ideal_traj.Y_rot,linewidth=0.75)
        ax1[1].set_xlabel("Theoretical trajectory length [m]")
    ax1[1].set_ylabel("Rotation around Y axis [°]")
    ax1[1].set_title("Rotation around Y axis")
    ax1[1].legend(["Reconstruction rotation","Theoretical rotation"])
    ax1[1].grid()

    if analysis_type == "rot" or analysis_type=="orbit":
        ax1[2].plot(reconstr_traj.rotation_angle,reconstr_traj.Z_rot)
        ax1[2].plot(reconstr_traj.rotation_angle,ideal_traj.Z_rot,linewidth=0.75)
        ax1[2].set_xlabel("Theoretical rotation angle around Y [°]")
    else:
        ax1[2].plot(ideal_traj.traj_length,reconstr_traj.Z_rot)
        ax1[2].plot(ideal_traj.traj_length,ideal_traj.Z_rot,linewidth=0.75)
        ax1[2].set_xlabel("Theoretical trajectory length [m]")
    ax1[2].set_ylabel("Rotation around Z axis [°]")
    ax1[2].set_title("Rotation around Z axis")
    ax1[2].legend(["Reconstruction rotation","Theoretical rotation"])
    ax1[2].set_xlim([0,180])
    ax1[2].set_ylim([0, 180])
    ax1[2].grid()
    plt.savefig(plot_filedir+"\\rotations.png")
    plt.close()


def plot_traj_length(analysis_type,reconstr_traj,ideal_traj,errors_dict,plot_filedir):
    fig1, ax1 = plt.subplots(nrows=2, ncols=1,figsize=(5,8),constrained_layout=True)

    if analysis_type == "rot" or analysis_type=="orbit":
        ax1[0].plot(reconstr_traj.rotation_angle,errors_dict["length_rec"] - errors_dict["length_ideal"])
        ax1[0].set_xlabel("Theoretical rotation angle around Y [°]")
    else:
        ax1[0].plot(ideal_traj.traj_length,errors_dict["length_rec"] - errors_dict["length_ideal"])
        ax1[0].set_xlabel("Theoretical trajectory length [m]")
    ax1[0].grid()
    ax1[0].set_ylabel("Length [m]")
    ax1[0].set_title("Difference reconstructed-real trajectories length")

    if analysis_type == "rot" or analysis_type=="orbit":
        ax1[1].plot(reconstr_traj.rotation_angle,errors_dict["step_length_diff"])
        ax1[1].set_xlabel("Theoretical rotation angle around Y [°]")
        ax1[1].set_ylabel("Length derivative [m/°]")
    else:
        ax1[1].plot(ideal_traj.traj_length,errors_dict["step_length_diff"])
        ax1[1].set_xlabel("Theoretical trajectory length [m]")
        ax1[1].set_ylabel("Length derivative [m/m]")
    ax1[1].set_title("Difference real-reconstructed trajectories length gradient")
    ax1[1].grid()
    plt.savefig(plot_filedir+"\\length.png")
    plt.close()


def write_all_data_to_json(errors_dict,dataset_name,reconstr_traj,ideal_traj,json_filename_path):
    json_file_text={
        "acquisition_name":dataset_name,
        "X-Z Plane max error": max(errors_dict["X_Z_plane_error"]),
        "X-Z Plane mean error": np.mean(errors_dict["X_Z_plane_error"]),
        "X-Z Plane SD error": np.std(errors_dict["X_Z_plane_error"]),
        "X-Y Plane max error":max(errors_dict["X_Y_plane_error"]),
        "X-Y Plane mean error":np.mean(errors_dict["X_Y_plane_error"]),
        "X-Y Plane SD error":np.std(errors_dict["X_Y_plane_error"]),
        "Y-Z Plane max error": max(errors_dict["Y_Z_plane_error"]),
        "Y-Z Plane mean error": np.mean(errors_dict["Y_Z_plane_error"]),
        "Y-Z Plane SD error":np.std(errors_dict["Y_Z_plane_error"]),
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
        "X reconstr":list(reconstr_traj.X),
        "Y reconstr":list(reconstr_traj.Y),
        "Z reconstr": list(reconstr_traj.Z),
        "X theoretical": list(ideal_traj.X),
        "Y theoretical": list(ideal_traj.Y),
        "Z theoretical": list(ideal_traj.Z),
        "Real length":list(errors_dict["length_rec"]),
        "X rot angle":list(reconstr_traj.X_rot),
        "Y rot angle": list(reconstr_traj.Y_rot),
        "Z rot angle": list(reconstr_traj.Z_rot),
        "X theoretical rot": list(ideal_traj.X_rot),
        "Y theoretical rot": list(ideal_traj.Y_rot),
        "Z theoretical rot": list(ideal_traj.Z_rot),
        "Final length": (errors_dict["length_rec"] - errors_dict["length_ideal"])[len(errors_dict["length_rec"] - errors_dict["length_ideal"]) - 1],
        "Mean diff. real-reconstr": np.mean(np.abs(errors_dict["length_rec"] - errors_dict["length_ideal"])),
        "Max diff. real-reconstr": np.max(np.abs(errors_dict["length_rec"] - errors_dict["length_ideal"])),
        "SD diff. real-reconstr": np.std(np.abs(errors_dict["length_rec"] - errors_dict["length_ideal"])),
        "Diff. real-reconstr":list(errors_dict["length_rec"] - errors_dict["length_ideal"]),
        "Gradient (Step diff.) real-reconstr": list(errors_dict["step_length_diff"])
    }
    with open(json_filename_path+".json", 'w') as outfile:
        obj = json.dump(json_file_text,outfile,indent=4)


def write_traj_file_transl_quad(traj,start_time,file_path_and_name):
    traj.compute_quat_from_angles()
    time_stamp = np.array(np.arange(len(traj.X)) + start_time)
    array_to_write = np.concatenate([np.expand_dims(time_stamp,axis=1),
                                           np.expand_dims(traj.X[0:len(time_stamp)],
                                            axis=1), np.expand_dims(traj.Y[0:len(time_stamp)], axis=1),
                                     np.expand_dims(traj.Z[0:len(time_stamp)], axis=1),
                                           traj.quat[0:len(time_stamp),:]], axis=1)
    np.savetxt(file_path_and_name,array_to_write)


def from_indicators_to_matrix_data(file_excel,parameters_to_eval,columns_values,rows_values,error_matrix_res_file):
    df = pd.read_excel(file_excel, index_col=0)
    for j, param in enumerate(parameters_to_eval):
        dataset_3D_errors = np.array(df.loc[param].values.tolist())
        if j==0:
            create_matrix_data_to_excel(dataset_3D_errors, columns_values, rows_values, error_matrix_res_file, True,
                                        param)
        else:
            create_matrix_data_to_excel(dataset_3D_errors, columns_values, rows_values, error_matrix_res_file, False,
                                        param)


def create_matrix_data_to_excel(data,columns_values,rows_values,error_matrix_res_file,new_file_flag,sheet_name):
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


def RPE_data_plot_and_saving(result_delta_1,result_delta_10_perc,ATE_seq_res_dir,file_quat,df_tot_RPE,n_fragments
                             ,res_fig_dir_dataset):
    trans_error_delta_1 = np.array(result_delta_1)[:, 4]
    rot_error_delta_1 = np.array(result_delta_1)[:, 5]
    trans_error_delta_10_perc = np.array(result_delta_10_perc)[:, 4]
    rot_error_delta_10_perc = np.array(result_delta_10_perc)[:, 5]

    print("compared_pose_pairs %d pairs" % (len(trans_error_delta_1)))
    print("translational_error.rmse %f m" % np.sqrt(np.dot(trans_error_delta_1, trans_error_delta_1)
                                                    / len(trans_error_delta_1)))
    print("translational_error.mean %f m" % np.mean(trans_error_delta_1))
    print("translational_error.median %f m" % np.median(trans_error_delta_1))
    print("translational_error.std %f m" % np.std(trans_error_delta_1))
    print("translational_error.min %f m" % np.min(trans_error_delta_1))
    print("translational_error.max %f m" % np.max(trans_error_delta_1))
    print("rotational_error.rmse %f deg" % (np.sqrt(np.dot(rot_error_delta_1, rot_error_delta_1)
                                                    / len(rot_error_delta_1)) * 180.0 / np.pi))
    print("rotational_error.mean %f deg" % (np.mean(rot_error_delta_1) * 180.0 / np.pi))
    print("rotational_error.median %f deg" % (np.median(rot_error_delta_1) * 180.0 / np.pi))
    print("rotational_error.std %f deg" % (np.std(rot_error_delta_1) * 180.0 / np.pi))
    print("rotational_error.min %f deg" % (np.min(rot_error_delta_1) * 180.0 / np.pi))
    print("rotational_error.max %f deg" % (np.max(rot_error_delta_1) * 180.0 / np.pi))

    print("compared_pose_pairs %d pairs" % (len(trans_error_delta_10_perc)))
    print("translational_error.rmse %f m" % np.sqrt(np.dot(trans_error_delta_10_perc, trans_error_delta_10_perc)
                                                    / len(trans_error_delta_10_perc)))
    print("translational_error.mean %f m" % np.mean(trans_error_delta_10_perc))
    print("translational_error.median %f m" % np.median(trans_error_delta_10_perc))
    print("translational_error.std %f m" % np.std(trans_error_delta_10_perc))
    print("translational_error.min %f m" % np.min(trans_error_delta_10_perc))
    print("translational_error.max %f m" % np.max(trans_error_delta_10_perc))
    print("rotational_error.rmse %f deg" % (np.sqrt(np.dot(rot_error_delta_10_perc, rot_error_delta_10_perc)
                                                    / len(rot_error_delta_10_perc)) * 180.0 / np.pi))
    print("rotational_error.mean %f deg" % (np.mean(rot_error_delta_10_perc) * 180.0 / np.pi))
    print("rotational_error.median %f deg" % (np.median(rot_error_delta_10_perc) * 180.0 / np.pi))
    print("rotational_error.std %f deg" % (np.std(rot_error_delta_10_perc) * 180.0 / np.pi))
    print("rotational_error.min %f deg" % (np.min(rot_error_delta_10_perc) * 180.0 / np.pi))
    print("rotational_error.max %f deg" % (np.max(rot_error_delta_10_perc) * 180.0 / np.pi))
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
