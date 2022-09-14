#!/usr/bin/python
# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Juergen Sturm, TUM
# All rights reserved.

"""
This script computes the absolute trajectory error from the ground truth
trajectory and the estimated trajectory.
"""

import sys
import numpy as np
import associate_original
import os
import json
import pandas as pd
import matplotlib.pyplot as plt


def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).

    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)

    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)

    """
    np.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U * S * Vh # np.identity(3)
    trans = data.mean(1) - rot * model.mean(1) # np.array([0,0,0])
    # trans=np.expand_dims(trans,axis=1)
    model_aligned = rot * model + trans # model
    alignment_error = model_aligned - data

    trans_error = np.sqrt(np.sum(np.multiply(alignment_error, alignment_error), 0)).A[0]

    return rot, trans, trans_error


def plot_traj(ax, stamps, traj, style, color, label):
    """
    Plot a trajectory using matplotlib.

    Input:
    ax -- the plot
    stamps -- time stamps (1xn)
    traj -- trajectory (3xn)
    style -- line style
    color -- line color
    label -- plot legend

    """
    stamps.sort()
    interval = np.median([s - t for s, t in zip(stamps[1:], stamps[:-1])])
    x = []
    y = []
    last = stamps[0]
    for i in range(len(stamps)):
        if stamps[i] - last < 2 * interval:
            x.append(traj[i][0])
            y.append(traj[i][2])
        elif len(x) > 0:
            ax.plot(x, y, style, color=color, label=label)
            label = ""
            x = []
            y = []
        last = stamps[i]
    if len(x) > 0:
        ax.plot(x, y, style, color=color, label=label)


# data_folder= "C:\\Users\\daniele.marchisotti\\OneDrive - Politecnico di Milano\\POLIMI(Dottorato)\\" \
#             "Point Cloud Processing\\Laser_scanner_simulation_new\\virtual_reconstruction_results"
# # "D:\\3DReconstruction\\Laser_scanner_simulation_new\\virtual_reconstruction_results"
# analysis_folder="Results_13th_analysis"
# res_dir=data_folder+"\\"+analysis_folder+"\\RPE_and_ATE_sequences"
# res_fig_dir=data_folder+"\\"+analysis_folder+"\\ATE_figures"
# if not os.path.exists(res_dir):
#     os.mkdir(res_dir)
# if not os.path.exists(res_fig_dir):
#     os.mkdir(res_fig_dir)
# traj_quat_folder=data_folder+"\\"+analysis_folder+"\\transl_and_quat"
# file_names=os.listdir(traj_quat_folder)
# files_ideal=[traj_quat_folder+"\\"+file for file in file_names if file.split("_")
# [len(file.split("_"))-1]=="ideal.txt"]
# files_non_ideal=[traj_quat_folder+"\\"+file for file in file_names if not file.split("_")
# [len(file.split("_"))-1]=="ideal.txt"]
# sim_file=[file for file in file_names if not file.split("_")[len(file.split("_"))-1]=="ideal.txt"]
# sim_file_ideal=[file for file in file_names if file.split("_")[len(file.split("_"))-1]=="ideal.txt"]
# files_idx=[files_non_ideal.index("".join(sim.split("_ideal"))) for i,sim in enumerate(files_ideal)]
# sim_file=[sim_file[i] for i in files_idx]
# files_non_ideal=[files_non_ideal[i] for i in files_idx]
# df_tot=pd.DataFrame()
# file_excel=data_folder+"\\"+analysis_folder+"\\ATE_index.xlsx"
# writer = pd.ExcelWriter(file_excel, engine='openpyxl')
# for ii,(ideal_file,file) in enumerate(zip(files_ideal,files_non_ideal)):
#
#     first_list = associate_original.read_file_list(file)
#     second_list = associate_original.read_file_list(ideal_file)
#
#     # matches = associate.associate(first_list, second_list, float(args.offset), float(args.max_difference))
#     # if len(matches) < 2:
#     #    sys.exit(
#     # "Couldn't find matching timestamp pairs between groundtruth and estimated trajectory! Did you choose the correct sequence?")
#
#     first_xyz = np.matrix([[float(value) for value in first_list[i][0:3]] for i in first_list.keys()]).transpose()
#     second_xyz = np.matrix(
#         [[float(value) for value in second_list[i][0:3]] for i in second_list.keys()]).transpose()
#     rot, trans, trans_error = align(second_xyz, first_xyz)
#
#     second_xyz_aligned = rot * second_xyz + trans
#
#     first_stamps = first_list.keys()
#     first_stamps=sorted(first_stamps)
#     first_xyz_full = np.matrix([[float(value) for value in first_list[b][0:3]] for b in first_stamps]).transpose()
#
#     second_stamps = second_list.keys()
#     second_stamps=sorted(second_stamps)
#     second_xyz_full = np.matrix(
#         [[float(value) for value in second_list[b][0:3]] for b in second_stamps]).transpose()
#     second_xyz_full_aligned = rot * second_xyz_full + trans
#
#     print("compared_pose_pairs %d pairs" % (len(trans_error)))
#     print("absolute_translational_error.rmse %f m" % np.sqrt(np.dot(trans_error, trans_error) / len(trans_error)))
#     print("absolute_translational_error.mean %f m" % np.mean(trans_error))
#     print("absolute_translational_error.median %f m" % np.median(trans_error))
#     print("absolute_translational_error.std %f m" % np.std(trans_error))
#     print("absolute_translational_error.min %f m" % np.min(trans_error))
#     print("absolute_translational_error.max %f m" % np.max(trans_error))
#     json_file_text={
#         "ATE sequence": list(trans_error)
#     }
#     with open(res_dir+"\\"+sim_file[ii].split(".txt")[0]+"ATE.json","w") as f:
#         obj=json.dump(json_file_text,f,indent=4)
#
#     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), constrained_layout=True)
#     # marker_points = list(np.arange(0, len(X), 1))
#     # markers_on = [item % 5 == 0 for item in marker_points]
#     ax.plot(np.arange(len(trans_error))/(len(trans_error)-1),trans_error)  # ,marker='o',markevery=markers_on)
#     ax.grid()
#     # ax[0].set_xlim([0-0.05*X[len(X)-1],(1+0.05)*X[len(X)-1]])
#     ax.set_xlabel("Percentage of reconstruction frames")
#     ax.set_ylabel("ATE [m]")
#     ax.set_title("ATE error")
#     plt.savefig(res_fig_dir + "\\" + sim_file[ii].split(".")[0] + "_ATE.png")
#     plt.close()
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     plot_traj(ax, first_stamps, first_xyz_full.transpose().A, '-', "black", "ground truth")
#     plot_traj(ax, second_stamps, second_xyz_full_aligned.transpose().A, '-', "blue", "estimated")
#     plt.show()
#     df = pd.Series({"Name": sim_file[ii].split(".")[0],
#                     "Pose pairs": len(trans_error),
#                     "RMSE ATE": np.sqrt(np.dot(trans_error, trans_error) / len(trans_error)),
#                     "Mean ATE":np.mean(trans_error),
#                     "Median ATE": np.median(trans_error),
#                     "std dev ATE": np.std(trans_error),
#                     "Min ATE": np.min(trans_error),
#                     "Max ATE": np.max(trans_error)})
#     df_tot = pd.concat([df_tot, df], axis=1)
# df_tot.to_excel(writer, sheet_name="Results")
# writer.save()
# #     # if args.save_associations:
# #     #    file = open(args.save_associations, "w")
# #     #    file.write("\n".join(
# #     #        ["%f %f %f %f %f %f %f %f" % (a, x1, y1, z1, b, x2, y2, z2) for (a, b), (x1, y1, z1), (x2, y2, z2) in
# #     #         zip(matches, first_xyz.transpose().A, second_xyz_aligned.transpose().A)]))
# #     #    file.close()
# #
# #
# #     # import matplotlib
# #
# #     # matplotlib.use('TkAgg')
# #     # import matplotlib.pyplot as plt
# #     # import matplotlib.pylab as pylab
# #     # from matplotlib.patches import Ellipse
# #
# #
