# importing required libraries
import sys

import numpy as np
import json
import os

# setting dataset folders
# for scene 1
datasets_folder="..\\datasets"
datasets_list=["dataset_0000_0_000"]
analysis_folder="..\\analysis_folder_scene_1"
if not os.path.exists(analysis_folder):
    os.mkdir(analysis_folder)
# for scene 2
# datasets_folder="datasets_scene_2"
# datasets_list=["bedroom_dataset_0000_0_000","bedroom_dataset_0020_0_000","bedroom_dataset_0040_0_000"
#                ,"bedroom_dataset_0060_0_000","bedroom_dataset_0080_0_000"]
# analysis_folder="analysis_folder_scene_2"
# datasets_folder_list=[datasets_folder+"\\"+final_folder for final_folder in datasets_list]

# setting parameters for the reconstruction system
max_depth=np.ones([len(datasets_list)])*7.0
voxel_size=np.ones([len(datasets_list)])*0.05
max_depth_diff=np.ones([len(datasets_list)])*0.07
preference_loop_closure_odometry=np.ones([len(datasets_list)])*0.1
preference_loop_closure_registration=np.ones([len(datasets_list)])*5.0
tsdf_cubic_size=np.ones([len(datasets_list)])*5.0
n_frames_per_fragment=np.ones([len(datasets_list)])*91
icp_method=["color"]*len(datasets_list)
global_registration=["ransac"]*len(datasets_list)
python_multi_threading=np.ones([len(datasets_list)])*True
results_folder=analysis_folder+"\\results"
datasets_folder_list=[datasets_folder+"\\"+dataset for dataset in datasets_list]
results_dataset_folder_list=[results_folder+"\\"+dataset for dataset in datasets_list]
depth_map_type=["redwood"]*len(datasets_list)
n_keyframes_per_n_frame=np.ones([len(datasets_list)])*5
min_depth=np.ones([len(datasets_list)])*0.3
config_folder=analysis_folder+"\\config"
if not os.path.exists(results_folder):
    os.mkdir(results_folder)
if not os.path.exists(config_folder):
    os.mkdir(config_folder)
json_file_names=[config_folder+"\\"+dataset for dataset in datasets_list]
n_fragments=np.array([4])

for i in range(len(datasets_folder_list)):
    # setting a dictionary containing all information to save to configuration file json
    json_file_text={
        "name":"Captured frames using virtual camera",
        "path_dataset":datasets_folder_list[i]+"\\",
        "path_intrinsic":datasets_folder_list[i]+"\\"+"camera_intrinsic.json",
        "max_depth":max_depth[i],
        "voxel_size":voxel_size[i],
        "max_depth_diff":max_depth_diff[i],
        "preference_loop_closure_odometry":preference_loop_closure_odometry[i],
        "preference_loop_closure_registration":preference_loop_closure_registration[i],
        "tsdf_cubic_size":tsdf_cubic_size[i],
        "icp_method":icp_method[i],
        "global_registration":global_registration[i],
        "python_multi_threading":python_multi_threading[i],
        "n_frames_per_fragment":int(n_frames_per_fragment[i]),
        "template_fragment_posegraph":"../" + results_dataset_folder_list[i]+"\\fragments/fragment_%03d.json",
        "depth_map_type":depth_map_type[i],
        "n_keyframes_per_n_frame":int(n_keyframes_per_n_frame[i]),
        "min_depth":min_depth[i],
        "template_global_posegraph":"../"+results_dataset_folder_list[i]+"\\scene/global_registration.json",
        "template_fragment_posegraph_optimized":"../"+results_dataset_folder_list[i]+"\\fragments/fragment_optimized_%03d.json",
        "template_refined_posegraph": "../"+results_dataset_folder_list[i]+"\\scene/refined_registration.json",
        "folder_fragment":"../"+results_dataset_folder_list[i]+"\\fragments/",
        "template_fragment_pointcloud": "../"+results_dataset_folder_list[i]+"\\fragments/fragment_%03d.ply",
        "folder_scene":"../"+results_dataset_folder_list[i]+"\\scene/",
        "template_global_posegraph_optimized":"../"+results_dataset_folder_list[i]+"\\scene/global_registration_optimized.json",
        "template_refined_posegraph_optimized":"../"+results_dataset_folder_list[i]+"\\scene/refined_registration_optimized.json",
        "template_global_mesh":"../"+results_dataset_folder_list[i]+"\\scene/integrated.ply",
        "template_global_traj":"../"+results_dataset_folder_list[i]+"\\scene/trajectory.log"
    }
    # saving data to json configuration file
    with open(json_file_names[i]+"_config.json", 'w') as outfile:
        obj = json.dump(json_file_text,outfile,indent=4)
