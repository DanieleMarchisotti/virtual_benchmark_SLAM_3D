import numpy as np
import cv2
import json
from skimage import measure
import multiprocessing
from joblib import Parallel,delayed
import time
import os
from numba import jit, cuda
import scipy.linalg
import process_function_test


@jit
def divide_pc_for_img_gpu(i,pcPoints,pcColors,my_matrix,step_y):
    pcPoints_divided=[]
    pcColors_divided=[]
    idx_points_5 = (pcPoints[:, 1] <= (my_matrix[0, i] - (step_y) / 2) *
                    pcPoints[:, 0]) ==(pcPoints[:, 1]>= (my_matrix[0, i] + (step_y) / 2) * pcPoints[:, 0])
    pcPoints_divided.append(pcPoints[idx_points_5,:])
    pcColors_divided.append(pcColors[idx_points_5,:])
    return pcPoints_divided,pcColors_divided


def divide_pc_for_img(i,pcPoints,pcColors,my_matrix,step_y):
    pcPoints_divided=[]
    pcColors_divided=[]
    idx_points_5 = (pcPoints[:, 1] <= (my_matrix[0, i] - (step_y) / 2) *
                    pcPoints[:, 0]) ==(pcPoints[:, 1]>= (my_matrix[0, i] + (step_y) / 2) * pcPoints[:, 0])
    pcPoints_divided.append(pcPoints[idx_points_5,:])
    pcColors_divided.append(pcColors[idx_points_5,:])
    return (pcPoints_divided,pcColors_divided)


@jit
def divide_each_row_per_pixel_gpu(i,height,pcPoints_divided,mz_matrix,step_z):
    idx_final = np.array([0])
    idx_final_len = np.array([0])

    for j in range(height):
        idx_points_6 = (pcPoints_divided[:, 2] >= (mz_matrix[j, 0] - (step_z) / 2)
                      *pcPoints_divided[:, 0]) == (pcPoints_divided[:, 2] <=
                                                    (mz_matrix[j, 0] + (step_z) / 2)* pcPoints_divided[:, 0])
        idx=np.where(idx_points_6 == 1)
        for k in range(len(idx)):
            idx_final=np.append(idx_final,idx[k])
        idx_final_len=np.append(idx_final_len,len(idx[0]))
    return (idx_final[1:],idx_final_len[1:])


def divide_each_row_per_pixel(i,height,pcPoints_divided, pcColors_divided,mz_matrix,step_z):
    pcPoints_sampled = [[] for z in range(height)]
    pcColors_sampled = [[] for z in range(height)]
    for j in range(height):
        idx_points_6 = (pcPoints_divided[i][:, 2] >= (mz_matrix[j, 0] - (step_z) / 2)
                      *pcPoints_divided[i][:, 0]) == (pcPoints_divided[i][:, 2] <=
                                                    (mz_matrix[j, 0] + (step_z) / 2)* pcPoints_divided[i][:, 0])
        pcPoints_sampled[j].append(pcPoints_divided[i][idx_points_6,:])
        pcColors_sampled[j].append(pcColors_divided[i][idx_points_6,:])
    return (pcPoints_sampled,pcColors_sampled)


def points_to_RGB_and_depth_map(width,height, res_div2, pcPoints_divided, pcColors_divided,depth,RGB_colors,yz):
    idx_len=0
    for i in range(width):
        for j in range(height):
            pcPoints_sampled = np.array(pcPoints_divided[i][res_div2[i][0][idx_len:idx_len + res_div2[i][1][j]]])
            pcColors_sampled = np.array(pcColors_divided[i][res_div2[i][0][idx_len:idx_len + res_div2[i][1][j]]])
            idx_len += res_div2[i][1][j]
            if pcPoints_sampled.shape[0] == 0 or pcPoints_sampled.shape[1] == 0:
                depth[i, j] = 0
                # Y_coord[i, j]=0
                # Z_coord[i, j]=0
                RGB_colors[i, j, :] = [0, 0, 0]
            else:
                idx_depth = np.argmin(np.sqrt((yz[i * height+j,0,:].T.dot(pcPoints_sampled.T))**2
                                              +(yz[i*height+j,1,:].T.dot(pcPoints_sampled.T))**2))
                # idx_depth = np.argmin(np.sqrt(np.sum(pcPoints_sampled ** 2, axis=1)))
                depth[i, j] = pcPoints_sampled[idx_depth, 0]
                # Y_coord[i, j] = pcPoints_sampled[i, j][idx_depth, 1];
                # Z_coord[i, j] = pcPoints_sampled[i, j][idx_depth, 2];
                RGB_colors[i, j, :] = np.uint8(pcColors_sampled[idx_depth, :] * 255)
        idx_len = 0
    return depth,RGB_colors


def get_neighbourhood_median_depth(neighbourhood_dim,Pixel,depth_img,width,height):
    left_idx=int(np.round(Pixel[0]-neighbourhood_dim/2))
    right_idx=int(np.round(Pixel[0]+neighbourhood_dim/2))
    top_idx=int(np.round(Pixel[1]-neighbourhood_dim/2))
    bottom_idx=int(np.round(Pixel[1]+neighbourhood_dim/2))
    if left_idx<0:
        left_idx=0
    if right_idx>width:
        right_idx=width
    if top_idx<0:
        top_idx=0
    if bottom_idx>height:
        bottom_idx=height
    neighbourhood=depth_img[top_idx:bottom_idx,left_idx:right_idx]
    reg_med=np.reshape(neighbourhood,[neighbourhood.shape[0]*neighbourhood.shape[1]])
    depth_median=np.median(reg_med[np.array([not R==0 for R in reg_med])])
    return depth_median


def get_neighbourhood_median_color(neighbourhood_dim,Pixel,color_img,width,height):
    left_idx=int(np.round(Pixel[0]-neighbourhood_dim/2))
    right_idx=int(np.round(Pixel[0]+neighbourhood_dim/2))
    top_idx=int(np.round(Pixel[1]-neighbourhood_dim/2))
    bottom_idx=int(np.round(Pixel[1]+neighbourhood_dim/2))
    if left_idx<0:
        left_idx=0
    if right_idx>width:
        right_idx=width
    if top_idx<0:
        top_idx=0
    if bottom_idx>height:
        bottom_idx=height
    neighbourhood=color_img[top_idx:bottom_idx,left_idx:right_idx,:]
    R_med = np.reshape(neighbourhood[:,:, 0], [neighbourhood.shape[0]*neighbourhood.shape[1]])
    G_med = np.reshape(neighbourhood[:,:, 1], [neighbourhood.shape[0]*neighbourhood.shape[1]])
    B_med = np.reshape(neighbourhood[:,:, 2], [neighbourhood.shape[0]*neighbourhood.shape[1]])
    pixel_median=np.zeros([3])
    pixel_median[0] = np.median(R_med[np.array([not R==0 for R in R_med])])
    pixel_median[1] = np.median(G_med[np.array([not G==0 for G in G_med])])
    pixel_median[2] = np.median(B_med[np.array([not B==0 for B in B_med])])
    return pixel_median


def get_null_spaces(n1):
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        res_space = pool.map(process_function_test.compute_line_for_each_camera_pixel, n1)
    return res_space


def process_points_to_RGB_and_depth_map(RGB_and_depth_inputs):
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        res_images = pool.map(process_function_test.points_to_RGB_and_depth_map, RGB_and_depth_inputs)
    return res_images


def scan_from_trajectory(FOV_h,FOV_v,width,height,dataset_folder,ideal_traj,pcColors_main,pcPoints_main, d_noise_status, camera_type):
    h_half_angle = np.tan(FOV_h / 2 / 180 * np.pi)
    v_half_angle = np.tan(FOV_v / 2 / 180 * np.pi)
    [my_matrix, mz_matrix] = np.meshgrid(
        np.linspace(-(h_half_angle + h_half_angle / (width)), (h_half_angle + h_half_angle / (width)), width + 1),
        np.linspace(-(v_half_angle + v_half_angle / (height)), (v_half_angle + v_half_angle / (height)), height + 1))
    step_z = mz_matrix[1, 0] - mz_matrix[0, 0]
    step_y = my_matrix[0, 1] - my_matrix[0, 0]
    cx_1 = width / 2
    cy_1 = height / 2
    fx_1 = (0 - cx_1) / (my_matrix[0, 0] + step_y / 2)
    fy_1 = (0 - cy_1) / (mz_matrix[0, 0] + step_z / 2)
    color_imgs_dir_list = dataset_folder + "\\color\\"
    depth_imgs_dir_list = dataset_folder + "\\depth\\"
    pc_dir_list = dataset_folder + "\\pc_real_coords\\"
    print("------  Starting 3D scan ------")
    for index in range(len(ideal_traj.X)):
        start_time=time.time()
        pcColors = pcColors_main.copy()
        pcPoints = pcPoints_main.copy()

        pcPoints[:, 0]+=ideal_traj.Z[index]
        pcPoints[:, 1] += ideal_traj.X[index]
        pcPoints[:, 2] += ideal_traj.Y[index]
        # rotations referred topoint cloud coordinate system, but angles referred to camera
        x_rot_matrix = np.array([[np.cos(ideal_traj.Y_rot[index] * np.pi / 180), -np.sin(ideal_traj.Y_rot[index] * np.pi / 180), 0],
                                 [np.sin(ideal_traj.Y_rot[index] * np.pi / 180), np.cos(ideal_traj.Y_rot[index] * np.pi / 180), 0],
                                 [0, 0, 1]])
        y_rot_matrix = np.array([[np.cos(ideal_traj.Z_rot[index] * np.pi / 180), 0, -np.sin(ideal_traj.Z_rot[index] * np.pi / 180)],
                                 [0, 1, 0],
                                 [np.sin(ideal_traj.Z_rot[index] * np.pi / 180), 0, np.cos(ideal_traj.Z_rot[index] * np.pi / 180)]])
        z_rot_matrix = np.array(
            [[1, 0, 0],
             [0, np.cos(ideal_traj.X_rot[index] * np.pi / 180), -np.sin(ideal_traj.X_rot[index] * np.pi / 180)],
             [0, np.sin(ideal_traj.X_rot[index] * np.pi / 180), np.cos(ideal_traj.X_rot[index] * np.pi / 180)]])
        rot_matrix = x_rot_matrix.dot(y_rot_matrix)
        rot_matrix = rot_matrix.dot(z_rot_matrix)
        pcPoints = rot_matrix.dot(pcPoints.transpose()).transpose()
        start_time2 = time.time()
        idx_points=pcPoints[:, 1] >= (my_matrix[0,0])* pcPoints[:, 0]
        idx_points_2 = pcPoints[:, 1] <= (my_matrix[0,width])* pcPoints[:, 0]
        idx_points_3 = pcPoints[:, 2] >= (mz_matrix[0,0])* pcPoints[:, 0]
        idx_points_4 = pcPoints[:, 2] <= (mz_matrix[height,0]) * pcPoints[:, 0]
        pcPoints = pcPoints[idx_points & idx_points_2 & idx_points_3 & idx_points_4,:]
        pcColors = pcColors[idx_points & idx_points_2 & idx_points_3 & idx_points_4, :]
        del idx_points, idx_points_2, idx_points_3, idx_points_4
        print("Intermediate elapsed time:", time.time() - start_time2)
        # start_time2 = time.time()
        # idx_points=pcPoints[:, 1] >= (my_matrix[0, 0] - (my_matrix[0, 1] - my_matrix[0, 0]) / 2)* pcPoints[:, 0]
        # pcPoints = pcPoints[idx_points,:]
        # pcColors = pcColors[idx_points,:]
        # del idx_points
        # idx_points_2 = pcPoints[:, 1] <= (my_matrix[0, width] + (my_matrix[0, 1] - my_matrix[0, 0]) / 2)* pcPoints[:, 0]
        # pcPoints = pcPoints[idx_points_2,:]
        # pcColors = pcColors[idx_points_2,:]
        # del idx_points_2
        # idx_points_3 = pcPoints[:, 2] >= (mz_matrix[0, 0] - (mz_matrix[1, 0] - mz_matrix[0, 0]) / 2)* pcPoints[:, 0]
        # pcPoints = pcPoints[idx_points_3,:]
        # pcColors = pcColors[idx_points_3,:]
        # del idx_points_3
        # idx_points_4 = pcPoints[:, 2] <= (mz_matrix[height, 0] + (mz_matrix[1, 0] - mz_matrix[0, 0]) / 2)* pcPoints[:, 0]
        # pcPoints = pcPoints[idx_points_4,:]
        # pcColors = pcColors[idx_points_4,:]
        # del idx_points_4
        print("Intermediate elapsed time:", time.time() - start_time2)

        # compute straight lines of camera pixels \

        start_time2 = time.time()
        res_divide_pc2=[]
        for i in range(width):
            res_divide_pc2.append(divide_pc_for_img_gpu(i, pcPoints, pcColors, my_matrix, step_y))
        # res_divide_pc=Parallel(n_jobs=MAX_THREAD,prefer="threads")(delayed(divide_pc_for_img)(
        #     corresp_id,pcPoints,pcColors,my_matrix,step_y)
        #                             for corresp_id in range(0,n_items_to_parallel))
        pcPoints_divided = np.array([prec_item[0] for prec_item in res_divide_pc2])
        pcColors_divided = np.array([prec_item[1] for prec_item in res_divide_pc2])
        pcPoints_divided = np.array([pcPoints_divided[z,0] for z in range(pcPoints_divided[:,0].shape[0])])
        pcColors_divided = np.array([pcColors_divided[z,0] for z in range(pcColors_divided[:, 0].shape[0])])
        print("Intermediate elapsed time:", time.time() - start_time2)
        del pcPoints,pcColors

        i=0
        res_div2=[]
        start_time2=time.time()
        for i in range(width):
            res_div2.append(divide_each_row_per_pixel_gpu(width,height,pcPoints_divided[i],mz_matrix,step_z))
        idx_len=0
        print("Intermediate elapsed time:", time.time() - start_time2)
        # start_time2 = time.time()
        # pcPoints_sampled=np.ndarray(shape=(640,480),dtype=object)
        # pcColors_sampled=np.ndarray(shape=(640,480),dtype=object)
        # for i in range(len(res_div2)):
        #   for j,item in enumerate(res_div2[i][1]):
        #       pcPoints_sampled[i,j]=np.array(pcPoints_divided[i][res_div2[i][0][idx_len:idx_len+item]])
        #       pcColors_sampled[i,j] = np.array(pcColors_divided[i][res_div2[i][0][idx_len:idx_len + item]])
        #       idx_len += item
        #   idx_len=0
        # res_divide_each_row=Parallel(n_jobs=MAX_THREAD,prefer="threads")(delayed(divide_each_row_per_pixel)(
        #     corresp_id,height, pcPoints_divided, pcColors_divided,mz_matrix,step_z)
        #                                                                   for corresp_id in range(0,n_items_to_parallel))
        # print("Intermediate elapsed time:", time.time() - start_time2)
        # pcPoints_sampled = np.array([prec_item[0] for prec_item in res_divide_each_row])
        # pcColors_sampled = np.array([prec_item[1] for prec_item in res_divide_each_row])
        # pcPoints_sampled = np.squeeze(pcPoints_sampled,axis=2)
        # pcColors_sampled = np.squeeze(pcColors_sampled,axis=2)
        # del pcPoints_divided,pcColors_divided

        depth = np.empty([width,height],dtype=np.float64)
        Y_coord = np.empty([width,height],dtype=np.float64)
        Z_coord = np.empty([width,height],dtype=np.float64)
        RGB_colors = np.empty([width,height,3],dtype=np.float64)
        # with multiprocessing.pool(processes=multiprocessing.cpu_count()) as po:
        #     res = po.map(points_to_RGB_and_depth_map, zip(width, height, res_div2, pcPoints_divided, pcColors_divided, depth, RGB_colors))
        # depth,RGB_colors=points_to_RGB_and_depth_map(width, height, res_div2, pcPoints_divided, pcColors_divided, depth, RGB_colors)
        start_time2 = time.time()
        if ideal_traj.analysis_type=="rot":
            n1 = np.array([np.ones([my_matrix.shape[0] * my_matrix.shape[1], 1]),
                           np.reshape(my_matrix.T, (my_matrix.shape[0] * my_matrix.shape[1], 1)),
                           np.reshape(mz_matrix.T, (mz_matrix.shape[0] * mz_matrix.shape[1], 1))])
            n1 = np.squeeze(n1)
            n1 = n1.T
            res_space=get_null_spaces(n1)
            yz = np.array([item[1] for item in res_space])
            yz = np.squeeze(yz)
            RGB_and_depth_inputs=[(height, res_div2[i], pcPoints_divided[i],
                                   pcColors_divided[i], depth[i], RGB_colors[i],
                                   yz[i * height:i * height+height],ideal_traj.analysis_type) for i in range(len(res_div2))]
        else:
            RGB_and_depth_inputs=[(height, res_div2[i], pcPoints_divided[i],
                                   pcColors_divided[i], depth[i], RGB_colors[i],
                                   None,ideal_traj.analysis_type) for i in range(len(res_div2))]
        res_images=process_points_to_RGB_and_depth_map(RGB_and_depth_inputs)
        # depth,RGB_colors=points_to_RGB_and_depth_map(width, height, res_div2, pcPoints_divided, pcColors_divided, depth, RGB_colors,yz)
        depth=np.array([item[0] for item in res_images])
        RGB_colors = np.array([item[1] for item in res_images])
        print("Intermediate elapsed time:", time.time() - start_time2)
        # del pcPoints_sampled,pcColors_sampled
        depth = depth.T
        depth = np.flip(depth, axis=0)
        RGB_colors=np.transpose(RGB_colors,(1,0,2))
        color_img = np.flip(RGB_colors, axis=2)
        color_img = np.flip(color_img, axis=0)
        del RGB_colors
        gray_img = cv2.cvtColor(np.uint8(color_img), cv2.COLOR_RGB2GRAY)
        gray_mask = gray_img == 0
        del gray_img
        label_image=measure.label(gray_mask,connectivity=2)
        del gray_mask
        s = measure.regionprops(label_image)

        if ideal_traj.analysis_type == "rot" and index > 55:
            threshold = 5000
        elif ideal_traj.analysis_type == "transl":
            threshold = 1000
        else:
            threshold = 500
        neighbourhood_dim = 20
        start_time2 = time.time()
        for i in range(len(s)):
            if s[i].area < threshold:
                for j in range(s[i].coords[:, 0].shape[0]):
                    color_img[s[i].coords[j, 0], s[i].coords[j, 1],:]= get_neighbourhood_median_color(
                        neighbourhood_dim,[s[i].coords[j, 1], s[i].coords[j, 0]], color_img, width, height)
                    depth[s[i].coords[j, 0], s[i].coords[j, 1]] =get_neighbourhood_median_depth(
                        neighbourhood_dim,[s[i].coords[j, 1], s[i].coords[j, 0]], depth, width, height)
        print("Intermediate elapsed time:", time.time() - start_time2)
        cv2.imwrite(color_imgs_dir_list + str(index).zfill(6) + ".png",np.uint8(color_img))
        cv2.imwrite(depth_imgs_dir_list+str(index).zfill(6)+".png",np.uint16(depth * 10000))
        del depth,color_img
        print("End 3D scan: ",index,"/",len(ideal_traj.X),"(elapsed time:",time.time()-start_time,")")
    intrinsic_dict={}
    intrinsic_dict["width"]=width
    intrinsic_dict["height"] = height
    intrinsic_dict["intrinsic_matrix"]=[fx_1, 0, 0, 0, fy_1, 0, cx_1, cy_1, 1]
    with open(dataset_folder+"\\camera_intrinsic.json","w") as f:
        obj=json.dump(intrinsic_dict,f)
    print("------  End 3D scan sequence ------")
