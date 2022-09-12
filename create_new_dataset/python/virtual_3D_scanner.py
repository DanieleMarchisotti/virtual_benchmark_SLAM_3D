import numpy as np
import cv2
import json
from skimage import measure
import multiprocessing
import time
from numba import jit, cuda
import process_function_test
import sys


@jit
def divide_pc_for_img_gpu(width, x_points, y_points, my_array):
    i=0
    while i<width:
        yield (y_points <= my_array[i+1] * x_points) & (y_points >= my_array[i] * x_points)
        i+=1


@jit
def select_pc_points_x(idx_step,pcPoints):
    yield pcPoints[idx_step, 0]


@jit
def select_pc_points_y(idx_step, pcPoints):
    yield pcPoints[idx_step, 1]


@jit
def select_pc_points_z(idx_step,pcPoints):
    yield pcPoints[idx_step, 2]


@jit
def select_pc_points(idx_step,pcPoints):
    yield pcPoints[idx_step, :]


@cuda.jit
def divide_pc_for_img_gpu_cuda(x_points, y_points, my_array, idx_points_5):
    i = cuda.grid(1)
    if i < idx_points_5.size:
        idx_points_5[i] = (y_points[i] <= (my_array[i]) *
                           x_points[i]) & (y_points[i] >= (my_array[i]) * x_points[i])


def divide_pc_for_img(i,pcPoints,pcColors,my_matrix,step_y):
    pcPoints_divided=[]
    pcColors_divided=[]
    idx_points_5 = (pcPoints[:, 1] <= (my_matrix[0, i] - (step_y) / 2) *
                    pcPoints[:, 0]) ==(pcPoints[:, 1]>= (my_matrix[0, i] + (step_y) / 2) * pcPoints[:, 0])
    pcPoints_divided.append(pcPoints[idx_points_5,:])
    pcColors_divided.append(pcColors[idx_points_5,:])
    return (pcPoints_divided,pcColors_divided)


@jit
def divide_each_row_per_pixel_gpu_gen(height,x_points,z_points,mz_matrix):
    i = 0
    while i<height:
        yield (z_points >= (mz_matrix[i])*x_points) & (z_points <= (mz_matrix[i+1])* x_points)
        i += 1


@jit
def divide_each_row_per_pixel_gpu(height,x_points,z_points,mz_matrix):
    idx_final = np.array([0])
    idx_final_len = np.array([0])

    for j in range(height):
        idx_points_6 = (z_points >= (mz_matrix[j])
                      *x_points) == (z_points <= (mz_matrix[j+1])* x_points)
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


def get_indexes_for_median_filter(Pixel,neighbourhood_dim,height,width):
    return np.array([max(0, int(np.round(Pixel[1] - neighbourhood_dim / 2))),min(height, int(np.round(Pixel[1] + neighbourhood_dim / 2))),
           max(0, int(np.round(Pixel[0] - neighbourhood_dim / 2))),
           min(width, int(np.round(Pixel[0] + neighbourhood_dim / 2)))])


@jit
def get_neighbourhood_median_area_depth(idx_pixels,depth_img):
    return depth_img[idx_pixels[0]:idx_pixels[1],idx_pixels[2]:idx_pixels[3]]


@jit
def get_neighbourhood_median_area_color(idx_pixels,color_img):
    return color_img[idx_pixels[0]:idx_pixels[1],idx_pixels[2]:idx_pixels[3],:]


def get_null_spaces(n1):
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        res_space = pool.map(process_function_test.compute_line_for_each_camera_pixel, n1)
    return res_space


def process_points_to_RGB_and_depth_map(RGB_and_depth_inputs):
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        res_images = pool.map(process_function_test.points_to_RGB_and_depth_map, RGB_and_depth_inputs)
    return res_images


def process_points_to_RGB_and_depth_map_gen(RGB_and_depth_inputs):
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        return pool.map(process_function_test.divide_each_row_per_pixel_gpu_gen, RGB_and_depth_inputs)


def points_to_RGB_and_depth_map_gen(pcPoints,idx_step_1,id):  # (pcPoints, idx_step_1,height,mz_matrix):
    # import time
    # idx_step_1 = [k for k in res_idx_width][0]
    # print("Intermediate elapsed time:", time.time() - start_time2)
    x_points = pcPoints[idx_step_1, 0]
    z_points = pcPoints[idx_step_1, 2]
    # print("Intermediate elapsed time:", time.time() - start_time2)
    depth=[]
    for j in range(height):
        res_div2 = (divide_each_row_per_pixel_gpu_gen(j, x_points, z_points, mz_matrix))

        pcPoints_sampled = x_points[[k for k in res_div2][0]]

        if pcPoints_sampled.shape[0] != 0:

            idx_depth = np.argmin(pcPoints_sampled)
            depth.append(pcPoints_sampled[idx_depth])
    return depth


def compute_one_frame(index, pcPoints_main, pcColors_main, ideal_traj, my_matrix, mz_matrix, width, height, color_imgs_dir_list,
                      depth_imgs_dir_list):
    start_time = time.time()
    pcPoints = pcPoints_main.copy()
    pcColors = pcColors_main.copy()

    pcPoints[:, 0] += ideal_traj.Z[index]
    pcPoints[:, 1] += ideal_traj.X[index]
    pcPoints[:, 2] += ideal_traj.Y[index]
    # rotations referred topoint cloud coordinate system, but angles referred to camera
    x_rot_matrix = np.array(
        [[np.cos(ideal_traj.Y_rot[index] * np.pi / 180), -np.sin(ideal_traj.Y_rot[index] * np.pi / 180), 0],
         [np.sin(ideal_traj.Y_rot[index] * np.pi / 180), np.cos(ideal_traj.Y_rot[index] * np.pi / 180), 0],
         [0, 0, 1]])
    y_rot_matrix = np.array(
        [[np.cos(ideal_traj.Z_rot[index] * np.pi / 180), 0, -np.sin(ideal_traj.Z_rot[index] * np.pi / 180)],
         [0, 1, 0],
         [np.sin(ideal_traj.Z_rot[index] * np.pi / 180), 0, np.cos(ideal_traj.Z_rot[index] * np.pi / 180)]])
    z_rot_matrix = np.array(
        [[1, 0, 0],
         [0, np.cos(ideal_traj.X_rot[index] * np.pi / 180), -np.sin(ideal_traj.X_rot[index] * np.pi / 180)],
         [0, np.sin(ideal_traj.X_rot[index] * np.pi / 180), np.cos(ideal_traj.X_rot[index] * np.pi / 180)]])
    rot_matrix = x_rot_matrix.dot(y_rot_matrix)
    rot_matrix = rot_matrix.dot(z_rot_matrix)
    pcPoints = rot_matrix.dot(pcPoints.transpose()).transpose()
    idx_points = pcPoints[:, 1] >= (my_matrix[0]) * pcPoints[:, 0]
    idx_points_2 = pcPoints[:, 1] <= (my_matrix[width]) * pcPoints[:, 0]
    idx_points_3 = pcPoints[:, 2] >= (mz_matrix[0]) * pcPoints[:, 0]
    idx_points_4 = pcPoints[:, 2] <= (mz_matrix[height]) * pcPoints[:, 0]
    pcPoints = pcPoints[idx_points & idx_points_2 & idx_points_3 & idx_points_4, :]
    pcColors = pcColors[idx_points & idx_points_2 & idx_points_3 & idx_points_4, :]
    pcColors = np.insert(pcColors, 0, 0, axis=0)
    del idx_points, idx_points_2, idx_points_3, idx_points_4
    res_idx_width_gen = divide_pc_for_img_gpu(width, pcPoints[:, 0], pcPoints[:, 1], my_matrix)
    depth = np.zeros([width, height], dtype=np.float64)
    color_idx = -np.ones([width, height], dtype=np.int32)
    for i, line_idx in enumerate(res_idx_width_gen):
        x_points = pcPoints[line_idx, 0]
        z_points = pcPoints[line_idx, 2]
        color_line_idx = np.squeeze(np.where(line_idx))
        # color_points = pcColors[line_idx, :]
        res_div2_gen = divide_each_row_per_pixel_gpu_gen(height, x_points, z_points, mz_matrix)
        for j, px_idx in enumerate(res_div2_gen):
            pcPoints_sampled = x_points[px_idx]
            try:
                pcColors_sampled = color_line_idx[px_idx]
                if pcPoints_sampled.shape[0] != 0:
                    idx_depth = np.argmin(pcPoints_sampled)
                    depth[i, j] = pcPoints_sampled[idx_depth]
                    color_idx[i, j] = pcColors_sampled[idx_depth]
            except IndexError:
                pass
                # RGB_colors[i, j, :] = pcColors_sampled[idx_depth]*255
    depth = depth.T
    # print(RGB_colors)

    color_idx=np.squeeze(np.reshape(color_idx,(width*height)))
    RGB_colors = np.uint8(pcColors[color_idx+1,:]*255)

    RGB_colors = np.array([np.reshape(RGB_colors[:, 0],(width, height)), np.reshape(RGB_colors[:, 1],(width, height)),
                           np.reshape(RGB_colors[:, 2],(width, height))])
    # print(RGB_colors.shape)
    depth = np.flip(depth, axis=0)
    RGB_colors = np.transpose(RGB_colors, (2, 1, 0))
    color_img = np.flip(RGB_colors, axis=2)
    color_img = np.flip(color_img, axis=0)
    del RGB_colors
    gray_img = cv2.cvtColor(np.uint8(color_img), cv2.COLOR_RGB2GRAY)
    gray_mask = gray_img == 0
    del gray_img
    label_image = measure.label(gray_mask, connectivity=2)
    del gray_mask
    s = measure.regionprops(label_image)

    if ideal_traj.analysis_type == "rot" and index > 55:
        threshold = 200000
    elif ideal_traj.analysis_type == "transl":
        threshold = 200000
        if 2132 < index < 2570:  # or 590<index<640:
            threshold=5000
    else:
        threshold = 200000
    neighbourhood_dim = 20
    for i in range(len(s)):
        if s[i].area < threshold:
            px_row = s[i].coords[:, 0]
            px_col = s[i].coords[:, 1]
            # color_copy=color_img.copy()
            # depth_copy=depth.copy()
            for j in range(s[i].coords[:, 0].shape[0]):
                neighbourhood_idx=get_indexes_for_median_filter([px_col[j], px_row[j]],neighbourhood_dim,height,width)
                # print("here4 ",index, "(elapsed time:", time.time() - start_time, ")")
                color_area=get_neighbourhood_median_area_color(neighbourhood_idx, color_img)
                R_col=color_area[:, :, 0]
                G_col=color_area[:, :, 1]
                B_col=color_area[:, :, 2]
                # print("here5 ", index,"(elapsed time:", time.time() - start_time, ")")
                if not (R_col == 0).all() or not (G_col == 0).all() or not (B_col == 0).all():
                    color_img[px_row[j], px_col[j], :] = [np.median(R_col[R_col!=0]),
                                                                          np.median(G_col[G_col!=0]),np.median(B_col[B_col!=0])]
                # print("here6 ", index, "(elapsed time:", time.time() - start_time, ")")
                depth_area = get_neighbourhood_median_area_depth(neighbourhood_idx, depth)
                if not (depth_area == 0).all():
                    depth[px_row[j], px_col[j]] = np.median(depth_area[depth_area!=0])
                # print("here7 ", index, "(elapsed time:", time.time() - start_time, ")")
    cv2.imwrite(color_imgs_dir_list + str(index).zfill(6) + ".png", np.uint8(color_img))
    cv2.imwrite(depth_imgs_dir_list + str(index).zfill(6) + ".png", np.uint16(depth * 1000))
    print("End 3D scan: ", index, "/", len(ideal_traj.X), "(elapsed time:", time.time() - start_time, ")")


def scan_from_trajectory(FOV_h, FOV_v, width, height, dataset_folder, ideal_traj, pcColors_main,
                         pcPoints_main, d_noise_status, camera_type):
    h_half_angle = np.tan(FOV_h / 2 / 180 * np.pi)
    v_half_angle = np.tan(FOV_v / 2 / 180 * np.pi)
    my_matrix=np.linspace(-(h_half_angle + h_half_angle / (width)), (h_half_angle + h_half_angle / (width)), width + 1)
    mz_matrix=np.linspace(-(v_half_angle + v_half_angle / (height)), (v_half_angle + v_half_angle / (height)), height + 1)
    step_z = mz_matrix[1] - mz_matrix[0]
    step_y = my_matrix[1] - my_matrix[0]
    cx_1 = width / 2
    cy_1 = height / 2
    fx_1 = (0 - cx_1) / (my_matrix[0] + step_y / 2)
    fy_1 = (0 - cy_1) / (mz_matrix[0] + step_z / 2)
    color_imgs_dir_list = dataset_folder + "\\color\\"
    depth_imgs_dir_list = dataset_folder + "\\depth\\"
    print("------  Starting 3D scan ------")
    import time
    start_time=time.time()
    from joblib import Parallel, delayed
    import multiprocessing
    MAX_THREAD = min(multiprocessing.cpu_count(), len(ideal_traj.X))
    Parallel(n_jobs=MAX_THREAD)(delayed(compute_one_frame)(
        index, pcPoints_main, pcColors_main, ideal_traj, my_matrix, mz_matrix, width, height, color_imgs_dir_list,
        depth_imgs_dir_list) for index in range(0, len(ideal_traj.X)))
    print("End 3D scan. Elapsed time:", time.time() - start_time)
    intrinsic_dict={}
    intrinsic_dict["width"] = width
    intrinsic_dict["height"] = height
    intrinsic_dict["intrinsic_matrix"]=[fx_1, 0, 0, 0, fy_1, 0, cx_1, cy_1, 1]
    with open(dataset_folder+"\\camera_intrinsic.json","w") as f:
        obj=json.dump(intrinsic_dict,f)
    print("------  End 3D scan sequence ------")
    if (d_noise_status == "Y" or d_noise_status == "y") and camera_type == "Intel RealSense D415":
        print("------  Start error application process ------")
        from add_depth_noise import add_depth_syst_and_random_noise_D415, add_depth_random_noise
        sensors_data_main_dir = "..\\sensors\\sensors_data\\"
        sensor_name = "D415\\"
        sys.path.append("..\\trajectory_reconstruction")
        import readBinFile
        sensor_data_dir = sensors_data_main_dir + sensor_name
        params_model = readBinFile.read_bin_double(sensor_data_dir + "depth_errors_parameters.bin")
        depth_array = params_model[0:4]
        B = params_model[4]
        f = params_model[5]
        rand_model = readBinFile.read_bin_double(sensor_data_dir + "random_model.bin")
        syst_model = readBinFile.read_bin_double(sensor_data_dir + "syst_model.bin")
        rand_model = np.reshape(rand_model, (height, width, 6))
        syst_model = np.reshape(syst_model, (height, width, 12))

        import time
        import os
        start_time = time.time()
        depth_img_file_paths = [depth_imgs_dir_list+item for item in os.listdir(depth_imgs_dir_list)]

        len_img_file_paths = len(depth_img_file_paths)
        Parallel(n_jobs=MAX_THREAD)(delayed(add_depth_random_noise)(path, B, f, rand_model, depth_array, i, len_img_file_paths, height, width)
                                    for i,path in enumerate(depth_img_file_paths))
        # Parallel(n_jobs=MAX_THREAD)(delayed(add_depth_syst_and_random_noise_D415)
        #                             (path, syst_model, depth_array, rand_model, len_img_file_paths, i, width, height,
        #                              B, f) for i,path in enumerate(depth_img_file_paths))
        print("Time total error application:", time.time() - start_time)

    if (d_noise_status == "Y" or d_noise_status == "y") and camera_type == "Azure Kinect DK":
        print("------  Start error application process ------")
        from add_depth_noise import add_depth_random_noise_Azure_Kinect, add_depth_random_noise_Azure_Kinect_slow
        sensors_data_main_dir = "..\\sensors\\sensors_data\\"
        sensor_name = "Azure_kinect_dk\\"
        sys.path.append("..\\trajectory_reconstruction")
        import readBinFile
        sensor_data_dir = sensors_data_main_dir + sensor_name
        params_model = readBinFile.read_bin_double(sensor_data_dir + "random_error_model_parameters.bin")
        rand_model = np.reshape(params_model, (2, width*height)).T
        import time
        import os
        start_time = time.time()
        depth_img_file_paths = [depth_imgs_dir_list+item for item in os.listdir(depth_imgs_dir_list)]
        len_img_file_paths = len(depth_img_file_paths)
        Parallel(n_jobs=MAX_THREAD)(delayed(add_depth_random_noise_Azure_Kinect)
                                    (path, rand_model, len_img_file_paths, i, width, height
                                     ) for i,path in enumerate(depth_img_file_paths))
        print("Time total error application:", time.time() - start_time)
