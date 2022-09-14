# importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import os
import json
import pandas as pd
import time
import import_ipynb
# importing files
import associate_original
import trajectory_analysis_methods
import evaluate_ate
import evaluate_rpe
import multiprocessing
from joblib import Parallel, delayed

# setting folders and paths for analysis results
# for scene 1
analysis_folder="..\\analysis_folder_scene_1"
# for scene 2
# analysis_folder="analysis_folder_scene_2"
json_res_dir=analysis_folder+"\\json"
figures_res_dir=analysis_folder+"\\figures"
res_dir=analysis_folder+"\\results"
ATE_seq_res_dir=analysis_folder+"\\RPE_and_ATE_sequences"
ATE_res_fig_dir=analysis_folder+"\\ATE_figures"
RPE_res_fig_dir=analysis_folder+"\\RPE_figures"
transl_and_quat_dir=analysis_folder+"\\transl_and_quat"
if not os.path.exists(ATE_seq_res_dir):
    os.mkdir(ATE_seq_res_dir)
if not os.path.exists(ATE_res_fig_dir):
    os.mkdir(ATE_res_fig_dir)
if not os.path.exists(json_res_dir):
    os.mkdir(json_res_dir)
if not os.path.exists(figures_res_dir):
    os.mkdir(figures_res_dir)
if not os.path.exists(res_dir):
    os.mkdir(res_dir)
if not os.path.exists(transl_and_quat_dir):
    os.mkdir(transl_and_quat_dir)
if not os.path.exists(RPE_res_fig_dir):
    os.mkdir(RPE_res_fig_dir)
datasets_folder=analysis_folder+"\\results"
datasets_names=os.listdir(datasets_folder)
for folder in datasets_names:
    if not os.path.exists(res_dir+"\\"+folder):
        os.mkdir(res_dir+"\\"+folder)
datasets_folder_list=[datasets_folder+"\\"+folder for folder in os.listdir(datasets_folder)]
trajectory_file=[folder+"\\scene\\trajectory.log" for folder in datasets_folder_list]

# setting excel file for results
excel_res_dir=analysis_folder
file_excel =excel_res_dir+"\\"+analysis_folder+".xlsx"
writer_error = pd.ExcelWriter(file_excel, engine='openpyxl')
df_tot_error=pd.DataFrame()
ATE_file_excel=analysis_folder+"\\ATE_index.xlsx"
ATE_writer = pd.ExcelWriter(ATE_file_excel, engine='openpyxl')
df_tot_ATE=pd.DataFrame()
df_tot_RPE=pd.DataFrame()
RPE_file_excel=analysis_folder+"\\RPE_index.xlsx"
RPE_writer = pd.ExcelWriter(RPE_file_excel, engine='openpyxl')

n_fragments = np.int32(4*np.ones(len(datasets_names)))
# for scene 1:
analysis_type="transl"
tot_angle_transl = 360
n_elem_traj_transl = np.int32(20*np.ones(len(datasets_names)))
R=0.5
center_point=[0,-0.5]
rows_values = np.int32(0*np.ones(len(datasets_names)))
columns_values = np.array([0])

# for scene 2:
# analysis_type="rot"
# tot_angle_rot=360
# n_elem_traj_rot= np.int32(360*np.ones(len(datasets_names)))
# rows_values = np.int32(0*np.ones(len(datasets_names)))
# columns_values = np.array([0,20,40,60,80])

start_time=time.time()
for ii in range(0,len(datasets_names)):
    if analysis_type=="transl":
        errors_dict,reconstr_traj,ideal_traj=trajectory_analysis_methods.evaluate_general_traj_errors(
            trajectory_file[ii], analysis_type,n_elem_traj_transl[ii],tot_angle_transl, R, center_point)
    else:
        errors_dict,reconstr_traj,ideal_traj=trajectory_analysis_methods.evaluate_general_traj_errors(
            trajectory_file[ii], analysis_type,n_elem_traj_rot[ii],tot_angle_rot)
    if not os.path.exists(figures_res_dir+"\\"+datasets_names[ii]):
        os.mkdir(figures_res_dir+"\\"+datasets_names[ii])

    trajectory_analysis_methods.plot_3d_traj_projections(reconstr_traj, ideal_traj, figures_res_dir+"\\" +
                                                         datasets_names[ii])
    trajectory_analysis_methods.plot_camera_orientation_angles(analysis_type, reconstr_traj, ideal_traj, figures_res_dir+"\\" +
                                                         datasets_names[ii])
    trajectory_analysis_methods.plot_traj_length(analysis_type, reconstr_traj, ideal_traj, errors_dict, figures_res_dir+"\\" +
                                                         datasets_names[ii])
    trajectory_analysis_methods.write_all_data_to_json(errors_dict, datasets_names[ii], reconstr_traj, ideal_traj,
                                                       json_res_dir+"\\"+datasets_names[ii])
    df = pd.Series({"name": datasets_names[ii],
                    "X-Z Plane max error": max(errors_dict["X_Z_plane_error"]),
                    "X-Z Plane mean error": np.mean(errors_dict["X_Z_plane_error"]),
                    "X-Z Plane SD error": np.std(errors_dict["X_Z_plane_error"]),
                    "X-Z Plane RMSE": np.sqrt(np.mean(errors_dict["X_Z_plane_error"]**2)),
                    "X-Y Plane max error": max(errors_dict["X_Y_plane_error"]),
                    "X-Y Plane mean error": np.mean(errors_dict["X_Y_plane_error"]),
                    "X-Y Plane SD error": np.std(errors_dict["X_Y_plane_error"]),
                    "X-Y Plane RMSE": np.sqrt(np.mean(errors_dict["X_Y_plane_error"]**2)),
                    "Y-Z Plane max error": max(errors_dict["Y_Z_plane_error"]),
                    "Y-Z Plane mean error": np.mean(errors_dict["Y_Z_plane_error"]),
                    "Y-Z Plane SD error": np.std(errors_dict["Y_Z_plane_error"]),
                    "Y-Z Plane RMSE": np.sqrt(np.mean(errors_dict["Y_Z_plane_error"]**2)),
                    "3D error max": max(errors_dict["error_3D"]),
                    "3D error mean": np.mean(errors_dict["error_3D"]),
                    "3D error SD": np.std(errors_dict["error_3D"]),
                    "3D error RMSE": np.sqrt(np.mean(errors_dict["error_3D"]**2)),
                    "X Rotation max error": max(errors_dict["X_rot_error"]),
                    "X Rotation mean error": np.mean(errors_dict["X_rot_error"]),
                    "X Rotation SD error": np.std(errors_dict["X_rot_error"]),
                    "X Rotation RMSE": np.sqrt(np.mean(errors_dict["X_rot_error"]**2)),
                    "Y Rotation max error": max(errors_dict["Y_rot_error"]),
                    "Y Rotation mean error": np.mean(errors_dict["Y_rot_error"]),
                    "Y Rotation SD error": np.std(errors_dict["Y_rot_error"]),
                    "Y Rotation RMSE": np.sqrt(np.mean(errors_dict["Y_rot_error"]**2)),
                    "Z Rotation max error": max(errors_dict["Z_rot_error"]),
                    "Z Rotation mean error": np.mean(errors_dict["Z_rot_error"]),
                    "Z Rotation SD error": np.std(errors_dict["Z_rot_error"]),
                    "Z Rotation RMSE": np.sqrt(np.mean(errors_dict["Z_rot_error"]**2)),
                    "Final length": (errors_dict["length_rec"] - errors_dict["length_ideal"])[len(errors_dict["length_rec"] - errors_dict["length_ideal"]) - 1],
                    "Mean diff. real-reconstr": np.mean(np.abs(errors_dict["length_rec"] - errors_dict["length_ideal"])),
                    "Max diff. real-reconstr": np.max(np.abs(errors_dict["length_rec"] - errors_dict["length_ideal"])),
                    "SD diff. real-reconstr": np.std(np.abs(errors_dict["length_rec"] - errors_dict["length_ideal"])),
                    "Gradient real-reconstr RMSE": np.sqrt(np.mean(errors_dict["step_length_diff"]**2))})
    df_tot_error = pd.concat([df_tot_error, df], axis=1)
    # write trajectories translation and quaternion files
    trajectory_analysis_methods.write_traj_file_transl_quad(reconstr_traj, start_time, transl_and_quat_dir+"\\"
                                                            +datasets_names[ii]+".txt")
    trajectory_analysis_methods.write_traj_file_transl_quad(ideal_traj, start_time,
                                                            transl_and_quat_dir + "\\" + datasets_names[ii] + "_ideal.txt")
    ideal_file_quat_path=transl_and_quat_dir + "\\" + datasets_names[ii] + "_ideal.txt"
    file_quat_path=transl_and_quat_dir+"\\"+datasets_names[ii]+".txt"
    ideal_file_quat=datasets_names[ii] + "_ideal.txt"
    file_quat=datasets_names[ii] + ".txt"
    first_list = associate_original.read_file_list(file_quat_path)
    second_list = associate_original.read_file_list(ideal_file_quat_path)
    first_xyz = np.matrix([[float(value) for value in first_list[i][0:3]] for i in first_list.keys()]).transpose()
    second_xyz = np.matrix(
        [[float(value) for value in second_list[i][0:3]] for i in second_list.keys()]).transpose()
    rot, trans, trans_error = evaluate_ate.align(second_xyz, first_xyz)
    json_file_text={
        "ATE sequence": list(trans_error)
    }
    with open(ATE_seq_res_dir+"\\"+file_quat.split(".txt")[0]+"ATE.json","w") as f:
        obj=json.dump(json_file_text,f,indent=4)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), constrained_layout=True)
    ax.plot(np.arange(len(trans_error))/(len(trans_error)-1),trans_error)
    ax.grid()
    ax.set_xlabel("Percentage of reconstruction frames")
    ax.set_ylabel("ATE [m]")
    ax.set_title("ATE error")
    plt.savefig(ATE_res_fig_dir + "\\" + file_quat.split(".")[0] + "_ATE.png")
    plt.close()

    df = pd.Series({"Name": file_quat.split(".")[0],
                    "Pose pairs": len(trans_error),
                    "RMSE ATE": np.sqrt(np.dot(trans_error, trans_error) / len(trans_error)),
                    "Mean ATE":np.mean(trans_error),
                    "Median ATE": np.median(trans_error),
                    "std dev ATE": np.std(trans_error),
                    "Min ATE": np.min(trans_error),
                    "Max ATE": np.max(trans_error)})
    df_tot_ATE = pd.concat([df_tot_ATE, df], axis=1)

    res_fig_dir_dataset=RPE_res_fig_dir+"\\"+file_quat.split(".txt")[0]
    if not os.path.exists(res_fig_dir_dataset):
        os.mkdir(res_fig_dir_dataset)
    traj_gt = evaluate_rpe.read_trajectory(ideal_file_quat_path)
    traj_est = evaluate_rpe.read_trajectory(file_quat_path)

    result_delta_1 = evaluate_rpe.evaluate_trajectory(traj_gt,
                                 traj_est,
                                 0, # max pairs
                                 True, # RPE for fixed delta
                                 1, # delta = 1, for odometry
                                 "f", # frames delta unit
                                 0, # offset
                                 1) # scale
    if analysis_type=="rot":
        result_delta_10_perc = evaluate_rpe.evaluate_trajectory(traj_gt,
                                 traj_est,
                                 0,  # max pairs
                                 True,  # RPE for fixed delta
                                 int(n_elem_traj_rot[ii]*0.1), # n_frames_per_fragment,  # delta = 1, for odometry
                                 "f",  # frames delta unit
                                 0,  # offset
                                 1)  # scale
    else:
        result_delta_10_perc = evaluate_rpe.evaluate_trajectory(traj_gt,
                                                                traj_est,
                                                                0,  # max pairs
                                                                True,  # RPE for fixed delta
                                                                int(n_elem_traj_transl[ii] * 0.1),
                                                                # n_frames_per_fragment,  # delta = 1, for odometry
                                                                "f",  # frames delta unit
                                                                0,  # offset
                                                                1)  # scale
    df_tot_RPE=trajectory_analysis_methods.RPE_data_plot_and_saving(result_delta_1,result_delta_10_perc,
                                                         ATE_seq_res_dir,file_quat,df_tot_RPE,n_fragments[ii]
                                                         ,res_fig_dir_dataset)

df_tot_error.to_excel(writer_error,sheet_name="Results")
writer_error.save()
df_tot_ATE.to_excel(ATE_writer, sheet_name="Results")
ATE_writer.save()
df_tot_RPE.to_excel(RPE_writer, sheet_name="Results")
RPE_writer.save()

# create error matrix
parameters_to_eval=["3D error RMSE","X Rotation RMSE","Y Rotation RMSE","Z Rotation RMSE"]
error_matrix_res_file=analysis_folder+"\\error_matrix.xlsx"
trajectory_analysis_methods.from_indicators_to_matrix_data(file_excel,parameters_to_eval,columns_values,rows_values
                                                           ,error_matrix_res_file)
parameters_to_eval=["RMSE ATE","Mean ATE","Median ATE","std dev ATE","Min ATE","Max ATE"]
ATE_matrix_excel_file=analysis_folder+"\\ATE_matrix.xlsx"
trajectory_analysis_methods.from_indicators_to_matrix_data(ATE_file_excel,parameters_to_eval,columns_values,rows_values
                                                           ,ATE_matrix_excel_file)
parameters_to_eval=["RMSE RPE transl delta 10 perc","RMSE RPE rot delta 10 perc",
                    "RMSE RPE transl delta 1","RMSE RPE rot delta 1"]
RPE_matrix_excel_file=analysis_folder+"\\RPE_matrix.xlsx"
trajectory_analysis_methods.from_indicators_to_matrix_data(RPE_file_excel,parameters_to_eval,columns_values,rows_values
                                                           ,RPE_matrix_excel_file)

def compute_recall_precision_from_two_pc(corresp_id,analysis_folder,source_file,target_file,source_file_list):
    '''
    This function computes recall and precision for a 3D reconstructed scene.
    To search the closest point of one point cloud respect to a point of the other the KDTree algorithm was used.
    '''
    print("Start analyzed dataset:",analysis_folder)
    target_pc = o3d.io.read_point_cloud(target_file)
    source_pc = o3d.io.read_point_cloud(source_file)
    # converting the reconstructed point cloud points (source_points) from m to mm
    source_points = np.asarray(source_pc.points) * 1000
    source_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_points))
    kdtree_source= o3d.geometry.KDTreeFlann(source_pc)
    kdtree_target = o3d.geometry.KDTreeFlann(target_pc)
    count=0
    print("------ Precision --------")
    start_time=time.time()
    for i,point in enumerate(np.asarray(source_pc.points)):
        [k, idx, _]=kdtree_target.search_radius_vector_3d(point,20)
        if len(idx)>0:
            count+=1
        if i%int(len(source_pc.points)/100)==0:
            print("Dataset",corresp_id,"/",len(source_file_list),"Percentage:",int(i/len(source_pc.points)*100),"%")
    precision=count/np.asarray(source_pc.points).shape[0]
    print("Precision: ",precision*100)
    print("Elapsed time:",time.time()-start_time)
    # recall
    count = 0
    print("------ Recall --------")
    for i, point in enumerate(target_pc.points):
        [k, idx, _] = kdtree_source.search_radius_vector_3d(point, 20)
        if len(idx) > 0:
            count += 1
        if i % int(len(target_pc.points) / 100) == 0:
            print("Dataset",corresp_id,"/",len(source_file_list),"Percentage:", int(i / len(target_pc.points) * 100), "%")
    recall = count / np.asarray(target_pc.points).shape[0]
    print("Recall: ", recall * 100)
    print("Elapsed time:", time.time() - start_time)
    return (precision,recall)


# for scene 1
analysis_folder="..\\analysis_folder_scene_1"
ground_truth_pc_folder="..\\ground_truth"
target_file=len(columns_values)*[ground_truth_pc_folder+"\\ground_truth_apt_subset_low.ply"]
# for scene 2
# analysis_folder="analysis_folder_scene_2"
# ground_truth_pc_folder="ground_truth"
# target_file=len(columns_values)*[ground_truth_pc_folder+"\\ground_truth_bedroom.ply"]


results_folder=analysis_folder+"\\results"
excel_file_name_prec_rec=analysis_folder+"\\precision_and_recall.xlsx"
folder_list=os.listdir(results_folder)
results_folder_list=[results_folder+"\\"+folder for folder in os.listdir(results_folder)]
source_file_list=[folder+"\\scene\\integrated.ply" for folder in results_folder_list]
precision=np.empty(len(source_file_list))
recall=np.empty(len(source_file_list))

n_items_to_parallel=len(source_file_list)

MAX_THREAD = min(multiprocessing.cpu_count(), n_items_to_parallel)
res=Parallel(n_jobs=MAX_THREAD)(delayed(compute_recall_precision_from_two_pc)(
            corresp_id,analysis_folder,source_file_list[corresp_id],target_file[corresp_id],source_file_list)
                                    for corresp_id in range(0,n_items_to_parallel))
prec=np.array([prec_item[0] for prec_item in res])
rec=np.array([prec_item[1] for prec_item in res])

trajectory_analysis_methods.create_matrix_data_to_excel(prec,columns_values,rows_values,excel_file_name_prec_rec,
                                                        True,"Precision")
trajectory_analysis_methods.create_matrix_data_to_excel(rec,columns_values,rows_values,excel_file_name_prec_rec,
                                                        False,"Recall")
