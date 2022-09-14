#!/usr/bin/python
# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Juergen Sturm, TUM
# All rights reserved.

"""
This script computes the relative pose error from the ground truth trajectory
and the estimated trajectory.
"""

import random
import numpy as np
import sys
import os
import pandas as pd
import json
import scipy.signal as sign

_EPS = np.finfo(float).eps * 4.0


def transform44(l):
    """
    Generate a 4x4 homogeneous transformation matrix from a 3D point and unit quaternion.

    Input:
    l -- tuple consisting of (stamp,tx,ty,tz,qx,qy,qz,qw) where
         (tx,ty,tz) is the 3D position and (qx,qy,qz,qw) is the unit quaternion.

    Output:
    matrix -- 4x4 homogeneous transformation matrix
    """
    t = l[1:4]
    q = np.array(l[4:8], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.array((
            (1.0, 0.0, 0.0, t[0])
            (0.0, 1.0, 0.0, t[1])
            (0.0, 0.0, 1.0, t[2])
            (0.0, 0.0, 0.0, 1.0)
        ), dtype=np.float64)
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3], t[0]),
        (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3], t[1]),
        (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1], t[2]),
        (0.0, 0.0, 0.0, 1.0)
    ), dtype=np.float64)


def read_trajectory(filename, matrix=True):
    """
    Read a trajectory from a text file.

    Input:
    filename -- file to be read
    matrix -- convert poses to 4x4 matrices

    Output:
    dictionary of stamped 3D poses
    """
    file = open(filename)
    data = file.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    list = [[float(v.strip()) for v in line.split(" ") if v.strip() != ""] for line in lines if
            len(line) > 0 and line[0] != "#"]
    list_ok = []
    for i, l in enumerate(list):
        if l[4:8] == [0, 0, 0, 0]:
            continue
        isnan = False
        for v in l:
            if np.isnan(v):
                isnan = True
                break
        if isnan:
            sys.stderr.write("Warning: line %d of file '%s' has NaNs, skipping line\n" % (i, filename))
            continue
        list_ok.append(l)
    if matrix:
        traj = dict([(l[0], transform44(l[0:])) for l in list_ok])
    else:
        traj = dict([(l[0], l[1:8]) for l in list_ok])
    return traj


def find_closest_index(L, t):
    """
    Find the index of the closest value in a list.

    Input:
    L -- the list
    t -- value to be found

    Output:
    index of the closest element
    """
    beginning = 0
    difference = abs(L[0] - t)
    best = 0
    end = len(L)
    while beginning < end:
        middle = int((end + beginning) / 2)
        if abs(L[middle] - t) < difference:
            difference = abs(L[middle] - t)
            best = middle
        if t == L[middle]:
            return middle
        elif L[middle] > t:
            end = middle
        else:
            beginning = middle + 1
    return best


def ominus(a, b):
    """
    Compute the relative 3D transformation between a and b.

    Input:
    a -- first pose (homogeneous 4x4 matrix)
    b -- second pose (homogeneous 4x4 matrix)

    Output:
    Relative 3D transformation from a to b.
    """
    return np.dot(np.linalg.inv(a), b)


def scale(a, scalar):
    """
    Scale the translational components of a 4x4 homogeneous matrix by a scale factor.
    """
    return np.array(
        [[a[0, 0], a[0, 1], a[0, 2], a[0, 3] * scalar],
         [a[1, 0], a[1, 1], a[1, 2], a[1, 3] * scalar],
         [a[2, 0], a[2, 1], a[2, 2], a[2, 3] * scalar],
         [a[3, 0], a[3, 1], a[3, 2], a[3, 3]]]
    )


def compute_distance(transform):
    """
    Compute the distance of the translational component of a 4x4 homogeneous matrix.
    """
    return np.linalg.norm(transform[0:3, 3])


def compute_angle(transform):
    """
    Compute the rotation angle from a 4x4 homogeneous matrix.
    """
    # an invitation to 3-d vision, p 27
    return np.arccos(min(1, max(-1, (np.trace(transform[0:3, 0:3]) - 1) / 2)))


def distances_along_trajectory(traj):
    """
    Compute the translational distances along a trajectory.
    """
    keys = traj.keys()
    keys.sort()
    motion = [ominus(traj[keys[i + 1]], traj[keys[i]]) for i in range(len(keys) - 1)]
    distances = [0]
    sum = 0
    for t in motion:
        sum += compute_distance(t)
        distances.append(sum)
    return distances


def rotations_along_trajectory(traj, scale):
    """
    Compute the angular rotations along a trajectory.
    """
    keys = traj.keys()
    keys.sort()
    motion = [ominus(traj[keys[i + 1]], traj[keys[i]]) for i in range(len(keys) - 1)]
    distances = [0]
    sum = 0
    for t in motion:
        sum += compute_angle(t) * scale
        distances.append(sum)
    return distances


def evaluate_trajectory(traj_gt, traj_est, param_max_pairs=10000, param_fixed_delta=False, param_delta=1.00,
                        param_delta_unit="s", param_offset=0.00, param_scale=1.00):
    """
    Compute the relative pose error between two trajectories.

    Input:
    traj_gt -- the first trajectory (ground truth)
    traj_est -- the second trajectory (estimated trajectory)
    param_max_pairs -- number of relative poses to be evaluated
    param_fixed_delta -- false: evaluate over all possible pairs
                         true: only evaluate over pairs with a given distance (delta)
    param_delta -- distance between the evaluated pairs
    param_delta_unit -- unit for comparison:
                        "s": seconds
                        "m": meters
                        "rad": radians
                        "deg": degrees
                        "f": frames
    param_offset -- time offset between two trajectories (to model the delay)
    param_scale -- scale to be applied to the second trajectory

    Output:
    list of compared poses and the resulting translation and rotation error
    """
    stamps_gt = list(traj_gt.keys())
    stamps_est = list(traj_est.keys())
    stamps_gt.sort()
    stamps_est.sort()

    stamps_est_return = []
    for t_est in stamps_est:
        t_gt = stamps_gt[find_closest_index(stamps_gt, t_est + param_offset)]
        t_est_return = stamps_est[find_closest_index(stamps_est, t_gt - param_offset)]
        t_gt_return = stamps_gt[find_closest_index(stamps_gt, t_est_return + param_offset)]
        if not t_est_return in stamps_est_return:
            stamps_est_return.append(t_est_return)
    if (len(stamps_est_return) < 2):
        raise Exception(
            "Number of overlap in the timestamps is too small. Did you run the evaluation on the right files?")

    if param_delta_unit == "s":
        index_est = list(traj_est.keys())
        index_est.sort()
    elif param_delta_unit == "m":
        index_est = distances_along_trajectory(traj_est)
    elif param_delta_unit == "rad":
        index_est = rotations_along_trajectory(traj_est, 1)
    elif param_delta_unit == "deg":
        index_est = rotations_along_trajectory(traj_est, 180 / np.pi)
    elif param_delta_unit == "f":
        index_est = range(len(traj_est))
    else:
        raise Exception("Unknown unit for delta: '%s'" % param_delta_unit)

    if not param_fixed_delta:
        if (param_max_pairs == 0 or len(traj_est) < np.sqrt(param_max_pairs)):
            pairs = [(i, j) for i in range(len(traj_est)) for j in range(len(traj_est))]
        else:
            pairs = [(random.randint(0, len(traj_est) - 1), random.randint(0, len(traj_est) - 1)) for i in
                     range(param_max_pairs)]
    else:
        pairs = []
        for i in range(len(traj_est)):
            j = find_closest_index(index_est, index_est[i] + param_delta)
            if j != len(traj_est) - 1:
                pairs.append((i, j))
        if (param_max_pairs != 0 and len(pairs) > param_max_pairs):
            pairs = random.sample(pairs, param_max_pairs)

    gt_interval = np.median([s - t for s, t in zip(stamps_gt[1:], stamps_gt[:-1])])
    gt_max_time_difference = 2 * gt_interval

    result = []
    for i, j in pairs:
        stamp_est_0 = stamps_est[i]
        stamp_est_1 = stamps_est[j]

        stamp_gt_0 = stamps_gt[find_closest_index(stamps_gt, stamp_est_0 + param_offset)]
        stamp_gt_1 = stamps_gt[find_closest_index(stamps_gt, stamp_est_1 + param_offset)]

        if (abs(stamp_gt_0 - (stamp_est_0 + param_offset)) > gt_max_time_difference or
                abs(stamp_gt_1 - (stamp_est_1 + param_offset)) > gt_max_time_difference):
            continue

        error44 = ominus(scale(
            ominus(traj_est[stamp_est_1], traj_est[stamp_est_0]), param_scale),
            ominus(traj_gt[stamp_gt_1], traj_gt[stamp_gt_0]))

        trans = compute_distance(error44)
        rot = compute_angle(error44)

        result.append([stamp_est_0, stamp_est_1, stamp_gt_0, stamp_gt_1, trans, rot])

    if len(result) < 2:
        raise Exception("Couldn't find matching timestamp pairs between groundtruth and estimated trajectory!")

    return result


def percentile(seq, q):
    """
    Return the q-percentile of a list
    """
    seq_sorted = list(seq)
    seq_sorted.sort()
    return seq_sorted[int((len(seq_sorted) - 1) * q)]

# RPE:
# - use transl RPE for circular trajectory
# - use rot RPE for rotational trajectory
# compute RPE delta:
# - delta=1
# - delta=0.1*n_frames
# save RPE:
# 1. transl RPE delta=1:
#   1.1 save sequence, graph and indicators (graph with stars at peaks of fragments)
# 2. transl RPE delta=0.1*n_frames
#   2.1 save sequence, graph and indicators
# 3. rot RPE delta=1:
#   3.1 save sequence, graph and indicators (graph with stars at peaks of fragments)
# 4. rot RPE delta=0.1*n_frames
#   4.1 save sequence, graph and indicators
# data_folder= "D:\\3DReconstruction\\Laser_scanner_simulation_new\\virtual_reconstruction_results"
#     # "C:\\Users\\daniele.marchisotti\\OneDrive - Politecnico di Milano\\POLIMI(Dottorato)\\" \
#       #      "Point Cloud Processing\\Laser_scanner_simulation_new\\virtual_reconstruction_results"
#
# analysis_folder="Results_22th_analysis"
# res_dir=data_folder+"\\"+analysis_folder+"\\RPE_and_ATE_sequences"
# res_fig_dir=data_folder+"\\"+analysis_folder+"\\RPE_figures"
# if not os.path.exists(res_dir):
#     os.mkdir(res_dir)
# if not os.path.exists(res_fig_dir):
#     os.mkdir(res_fig_dir)
# traj_quat_folder=data_folder+"\\"+analysis_folder+"\\transl_and_quat"
# file_names=os.listdir(traj_quat_folder)
# files_ideal=[traj_quat_folder+"\\"+file for file in file_names if file.split("_")[len(file.split("_"))-1]=="ideal.txt"]
# files_non_ideal=[traj_quat_folder+"\\"+file for file in file_names if not file.split("_")[len(file.split("_"))-1]=="ideal.txt"]
# sim_file=[file for file in file_names if not file.split("_")[len(file.split("_"))-1]=="ideal.txt"]
# sim_file_ideal=[file for file in file_names if file.split("_")[len(file.split("_"))-1]=="ideal.txt"]
# files_idx=[files_non_ideal.index("".join(sim.split("_ideal"))) for i,sim in enumerate(files_ideal)]
# sim_file=[sim_file[i] for i in files_idx]
# files_non_ideal=[files_non_ideal[i] for i in files_idx]
# df_tot=pd.DataFrame()
# file_excel=data_folder+"\\"+analysis_folder+"\\RPE_index.xlsx"
# writer = pd.ExcelWriter(file_excel, engine='openpyxl')
# random.seed(0)
# n_frames = np.int32(360*np.ones(len(sim_file))) # np.array([int(sim_file[ii].split(".txt")[0].split("_")[0].split("dataset")[1]) for ii in range(0,len(sim_file))
#                      # if not sim_file[ii].split(".txt")[0].split("_")[0].split("dataset")[1]==""]) #
# # n_frames=np.append(n_frames,np.ones(7)*360)
# # n_frames=np.insert(n_frames,0,np.ones(7)*270)
# n_fragments = np.int32(4*np.ones(len(sim_file))) # np.array([int(sim_file[ii].split(".txt")[0].split("_")[2]) for ii in range(0,len(sim_file))])
# # n_fragments=np.insert(n_fragments,0,np.array([1,2,6,8,10,15,20]))
#
# for ii,(ideal_file,file) in enumerate(zip(files_ideal,files_non_ideal)):
#     res_fig_dir_dataset=res_fig_dir+"\\"+sim_file[ii].split(".txt")[0]
#     if not os.path.exists(res_fig_dir_dataset):
#         os.mkdir(res_fig_dir_dataset)
#     # n_frames_per_fragment = int(np.ceil(n_frames / n_fragments))
#     traj_gt = read_trajectory(ideal_file)
#     traj_est = read_trajectory(file)
#
#     result_delta_1 = evaluate_trajectory(traj_gt,
#                                  traj_est,
#                                  0, # max pairs
#                                  True, # RPE for fixed delta
#                                  1, # delta = 1, for odometry
#                                  "f", # frames delta unit
#                                  0, # offset
#                                  1) # scale
#     result_delta_10_perc = evaluate_trajectory(traj_gt,
#                                  traj_est,
#                                  0,  # max pairs
#                                  True,  # RPE for fixed delta
#                                  int(n_frames[ii]*0.1), # n_frames_per_fragment,  # delta = 1, for odometry
#                                  "f",  # frames delta unit
#                                  0,  # offset
#                                  1)  # scale
#
#     stamps_delta_1= np.array(result_delta_1)[:, 0]
#     trans_error_delta_1 = np.array(result_delta_1)[:, 4]
#     rot_error_delta_1 = np.array(result_delta_1)[:, 5]
#     stamps_delta_10_perc = np.array(result_delta_10_perc)[:, 0]
#     trans_error_delta_10_perc = np.array(result_delta_10_perc)[:, 4]
#     rot_error_delta_10_perc = np.array(result_delta_10_perc)[:, 5]
#
#     print("compared_pose_pairs %d pairs" % (len(trans_error_delta_1)))
#     print("translational_error.rmse %f m" % np.sqrt(np.dot(trans_error_delta_1, trans_error_delta_1)
#                                                        / len(trans_error_delta_1)))
#     print("translational_error.mean %f m" % np.mean(trans_error_delta_1))
#     print("translational_error.median %f m" % np.median(trans_error_delta_1))
#     print("translational_error.std %f m" % np.std(trans_error_delta_1))
#     print("translational_error.min %f m" % np.min(trans_error_delta_1))
#     print("translational_error.max %f m" % np.max(trans_error_delta_1))
#     print("rotational_error.rmse %f deg" % (np.sqrt(np.dot(rot_error_delta_1, rot_error_delta_1)
#                                                        / len(rot_error_delta_1)) * 180.0 / np.pi))
#     print("rotational_error.mean %f deg" % (np.mean(rot_error_delta_1) * 180.0 / np.pi))
#     print("rotational_error.median %f deg" % (np.median(rot_error_delta_1) * 180.0 / np.pi))
#     print("rotational_error.std %f deg" % (np.std(rot_error_delta_1) * 180.0 / np.pi))
#     print("rotational_error.min %f deg" % (np.min(rot_error_delta_1) * 180.0 / np.pi))
#     print("rotational_error.max %f deg" % (np.max(rot_error_delta_1) * 180.0 / np.pi))
#
#     print("compared_pose_pairs %d pairs" % (len(trans_error_delta_10_perc)))
#     print("translational_error.rmse %f m" % np.sqrt(np.dot(trans_error_delta_10_perc, trans_error_delta_10_perc)
#                                                        / len(trans_error_delta_10_perc)))
#     print("translational_error.mean %f m" % np.mean(trans_error_delta_10_perc))
#     print("translational_error.median %f m" % np.median(trans_error_delta_10_perc))
#     print("translational_error.std %f m" % np.std(trans_error_delta_10_perc))
#     print("translational_error.min %f m" % np.min(trans_error_delta_10_perc))
#     print("translational_error.max %f m" % np.max(trans_error_delta_10_perc))
#     print("rotational_error.rmse %f deg" % (np.sqrt(np.dot(rot_error_delta_10_perc, rot_error_delta_10_perc)
#                                                        / len(rot_error_delta_10_perc)) * 180.0 / np.pi))
#     print("rotational_error.mean %f deg" % (np.mean(rot_error_delta_10_perc) * 180.0 / np.pi))
#     print("rotational_error.median %f deg" % (np.median(rot_error_delta_10_perc) * 180.0 / np.pi))
#     print("rotational_error.std %f deg" % (np.std(rot_error_delta_10_perc) * 180.0 / np.pi))
#     print("rotational_error.min %f deg" % (np.min(rot_error_delta_10_perc) * 180.0 / np.pi))
#     print("rotational_error.max %f deg" % (np.max(rot_error_delta_10_perc) * 180.0 / np.pi))
#     json_text_file={
#         "RPE_transl_delta_1:":list(trans_error_delta_1),
#         "RPE_transl_delta_10_perc:":list(trans_error_delta_10_perc),
#         "RPE_rot_delta_1:":list(rot_error_delta_1*180/np.pi),
#         "RPE_rot_delta_10_perc:":list(rot_error_delta_10_perc*180/np.pi)
#     }
#     with open(res_dir+"\\"+sim_file[ii].split(".txt")[0]+"RPE.json","w") as f:
#         obj=json.dump(json_text_file,f,indent=4)
#     df = pd.Series({"Name": sim_file[ii].split(".")[0],
#                     "Pose pairs delta 1": len(trans_error_delta_1),
#                     "Pose pairs delta 10 perc": len(trans_error_delta_10_perc),
#                     "RMSE RPE transl delta 1": np.sqrt(np.dot(trans_error_delta_1, trans_error_delta_1) / len(trans_error_delta_1)),
#                     "Mean RPE transl delta 1": np.mean(trans_error_delta_1),
#                     "Median RPE transl delta 1":np.median(trans_error_delta_1),
#                     "Std dev RPE transl delta 1":np.std(trans_error_delta_1),
#                     "Min RPE transl delta 1":np.min(trans_error_delta_1),
#                     "Max RPE transl delta 1":np.max(trans_error_delta_1),
#                     "RMSE RPE transl delta 10 perc": np.sqrt(
#                         np.dot(trans_error_delta_10_perc, trans_error_delta_10_perc) / len(trans_error_delta_10_perc)),
#                     "Mean RPE transl delta 10 perc": np.mean(trans_error_delta_10_perc),
#                     "Median RPE transl delta 10 perc": np.median(trans_error_delta_10_perc),
#                     "Std dev RPE transl delta 10 perc": np.std(trans_error_delta_10_perc),
#                     "Min RPE transl delta 10 perc": np.min(trans_error_delta_10_perc),
#                     "Max RPE transl delta 10 perc": np.max(trans_error_delta_10_perc),
#                     "RMSE RPE rot delta 1": np.sqrt(
#                         np.dot(rot_error_delta_1, rot_error_delta_1) / len(rot_error_delta_1))*180/np.pi,
#                     "Mean RPE rot delta 1": np.mean(rot_error_delta_1)*180/np.pi,
#                     "Median RPE rot delta 1": np.median(rot_error_delta_1)*180/np.pi,
#                     "Std dev RPE rot delta 1": np.std(rot_error_delta_1)*180/np.pi,
#                     "Min RPE rot delta 1": np.min(rot_error_delta_1)*180/np.pi,
#                     "Max RPE rot delta 1": np.max(rot_error_delta_1)*180/np.pi,
#                     "RMSE RPE rot delta 10 perc": np.sqrt(
#                         np.dot(rot_error_delta_10_perc, rot_error_delta_10_perc) / len(rot_error_delta_10_perc))*180/np.pi,
#                     "Mean RPE rot delta 10 perc": np.mean(rot_error_delta_10_perc)*180/np.pi,
#                     "Median RPE rot delta 10 perc": np.median(rot_error_delta_10_perc)*180/np.pi,
#                     "Std dev RPE rot delta 10 perc": np.std(rot_error_delta_10_perc)*180/np.pi,
#                     "Min RPE rot delta 10 perc": np.min(rot_error_delta_10_perc)*180/np.pi,
#                     "Max RPE rot delta 10 perc": np.max(rot_error_delta_10_perc)*180/np.pi
#                     })
#     df_tot = pd.concat([df_tot, df], axis=1)
#     import matplotlib.pyplot as plt
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     peaks, properties = sign.find_peaks(trans_error_delta_1, height=np.mean(trans_error_delta_1))
#     indices = np.argsort(properties['peak_heights'])
#     peaks=peaks[indices]
#     markers_on=list(peaks[len(peaks)-n_fragments[ii]+1:])
#     ax.plot(np.arange(len(trans_error_delta_1))/len(trans_error_delta_1), trans_error_delta_1, '-',
#             color="b",markevery=markers_on,marker='x',markeredgecolor='red')
#     # ax.plot([t for t,e in err_rot],[e for t,e in err_rot],'-',color="red")
#     ax.set_xlabel('reconstruction percentage [%]')
#     ax.set_ylabel('translational error [m]')
#     ax.set_title(r"Translational error with $\Delta$ = 1")
#     plt.grid()
#     plt.savefig(res_fig_dir_dataset+"\\trans_error_delta_1.png")
#     plt.close()
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.plot((np.arange(len(trans_error_delta_1))/len(trans_error_delta_1))[0:len(trans_error_delta_10_perc)],
#             trans_error_delta_10_perc, '-', color="blue")
#     # ax.plot([t for t,e in err_rot],[e for t,e in err_rot],'-',color="red")
#     ax.set_xlabel('reconstruction percentage [%]')
#     ax.set_ylabel('translational error [m]')
#     ax.set_title(r"Translational error with $\Delta$ = 10 %")
#     plt.grid()
#     plt.savefig(res_fig_dir_dataset+"\\trans_error_delta_10_perc.png")
#     plt.close()
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     peaks, properties = sign.find_peaks(rot_error_delta_1, height=np.mean(rot_error_delta_1))
#     indices = np.argsort(properties['peak_heights'])
#     peaks = peaks[indices]
#     markers_on = list(peaks[len(peaks) - n_fragments[ii] + 1:])
#     ax.plot(np.arange(len(rot_error_delta_1))/len(rot_error_delta_1), rot_error_delta_1*180/np.pi, '-', color="blue"
#             ,markevery=markers_on,marker='x',markeredgecolor='red')
#     # ax.plot([t for t,e in err_rot],[e for t,e in err_rot],'-',color="red")
#     ax.set_xlabel('reconstruction percentage [%]')
#     ax.set_ylabel('rotational error [°]')
#     ax.set_title(r"Rotational error with $\Delta$ = 1")
#     plt.grid()
#     plt.savefig(res_fig_dir_dataset+"\\rot_error_delta_1.png")
#     plt.close()
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.plot((np.arange(len(rot_error_delta_1))/len(rot_error_delta_1))[0:len(rot_error_delta_10_perc)],
#             rot_error_delta_10_perc*180/np.pi, '-', color="blue")
#     # ax.plot([t for t,e in err_rot],[e for t,e in err_rot],'-',color="red")
#     ax.set_xlabel('reconstruction percentage [%]')
#     ax.set_ylabel('rotational error [°]')
#     ax.set_title(r"Rotational error with $\Delta$ = 10 %")
#     plt.grid()
#     plt.savefig(res_fig_dir_dataset+"\\rot_error_delta_10_perc.png")
#     plt.close()
#
# df_tot.to_excel(writer, sheet_name="Results")
# writer.save()
