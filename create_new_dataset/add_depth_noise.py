import numpy as np
import readIntrinsicJson
import open3d as o3d
import pcFunc
import sys
sys.path.append("..\\trajectory_reconstruction")
import readBinFile
import matplotlib.pyplot as plt
from numba import jit
import time


@jit(nopython=True)
def create_model_generator(model, model_len, model_line_len):
    i = 0
    j = 0
    while i < model_len:
        while j < model_line_len:
            yield model[i, j, :]
            j += 1
        i += 1
        j = 0


@jit(nopython=True)
def create_rand_generator(model, model_len, model_line_len):
    i = 0
    j = 0
    while i < model_len:
        while j < model_line_len:
            yield model[i, j]
            j += 1
        i += 1
        j = 0


@jit(nopython=True)
def create_model_generator_1D(model, width, height):
    row = 0
    col = 0
    while row < height:
        while col < width:
            yield model[col+row*width, :]
            col += 1
        row += 1
        col = 0
# def add_depth_random_noise(depth_img_file, B, f, params_matrix, depth_array, img_index, len_img_file_paths, height, width):
#     start_time = time.time()
#     reconstr_map = np.zeros((height, width))
#     depth_img = np.asarray(o3d.io.read_image(depth_img_file))
#     depth_gen = create_depth_generator(depth_img, height, width)
#     rand_model_gen = create_model_generator(params_matrix, height, width)
#     reconstr_map = []
#     for i in range(height):
#         for j in range(width):
#             if (params_matrix[i, j, :] == 0).all() or not depth_img[i, j]:
#                 reconstr_map[i, j] = 0
#             elif (params_matrix[i, j, 0:4] == 0).any():
#                 prev_point = np.max(np.where((params_matrix[i, j, :] > 0) == False))
#                 if depth_img[i, j] <= depth_array[prev_point]:
#                     reconstr_map[i, j] = 0
#                 elif depth_array[prev_point] < depth_img[i, j] <= depth_array[prev_point + 1]:
#                     reconstr_map[i, j] = depth_img[i, j] + np.random.normal(0, (depth_img[i, j] - depth_array[prev_point]) * params_matrix[i, j, 5])
#                 else:
#                     reconstr_map[i, j] = depth_img[i, j] + np.random.normal(0, params_matrix[i, j, 4] * depth_img[i, j] ** 2 / (B * f))
#             else:
#                 reconstr_map[i, j] = depth_img[i, j] + np.random.normal(0, params_matrix[i, j, 4] * depth_img[i, j] ** 2 / (B * f))
#     reconstr_map = np.array(reconstr_map)
#     reconstr_map = np.reshape(reconstr_map, (height, width))
#     o3d.io.write_image(depth_img_file, o3d.geometry.Image(np.uint16(reconstr_map)))
#     print("End 3D scan: ", img_index, "/", len_img_file_paths, "(elapsed time:", time.time() - start_time, ")")


def add_depth_random_noise(depth_img_file, B, f, params_matrix, depth_array, img_index, len_img_file_paths, height, width):
    start_time = time.time()
    depth_img = np.asarray(o3d.io.read_image(depth_img_file))
    depth_gen = create_depth_generator(depth_img, height, width)
    rand_model_gen = create_model_generator(params_matrix, height, width)
    reconstr_map = []
    for i,(params_matrix_rand_curr, depth_value) in enumerate(zip(rand_model_gen,depth_gen)):
        if not depth_value or (params_matrix_rand_curr == 0).all():
            reconstr_map.append(0)
        elif (params_matrix_rand_curr[0:4] == 0).any():
            prev_point = np.max(np.where((params_matrix_rand_curr > 0) == False))
            if depth_value <= depth_array[prev_point]:
                reconstr_map.append(0)
            elif depth_array[prev_point] < depth_value <= depth_array[prev_point + 1]:
                reconstr_map.append(depth_value + np.random.normal(0, (depth_value - depth_array[prev_point]) * params_matrix_rand_curr[5]))
            else:
                reconstr_map.append(depth_value + np.random.normal(0, params_matrix_rand_curr[4] * depth_value ** 2 / (B * f)))
        else:
            reconstr_map.append(depth_value + np.random.normal(0, params_matrix_rand_curr[4] * depth_value ** 2 / (B * f)))
    reconstr_map = np.array(reconstr_map)
    reconstr_map = np.reshape(reconstr_map, (height, width))
    o3d.io.write_image(depth_img_file, o3d.geometry.Image(np.uint16(reconstr_map)))
    print("End 3D scan: ", img_index, "/", len_img_file_paths, "(elapsed time:", time.time() - start_time, ")")


def add_depth_syst_noise(depth_img, params_matrix, depth_array):
    reconstr_map = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            if (params_matrix[i,j, :] == 0).all() or params_matrix[i,j, 7] == 0 or not depth_img[i,j]:
                reconstr_map[i,j] = 0
            elif (params_matrix[i,j, 4:8] == 0).any():
                prev_point = np.max(np.where((params_matrix[i,j, 4:8] > 0) == False))
                if depth_img[i, j] <= depth_array[prev_point]:
                    reconstr_map[i,j] = 0
                elif depth_array[prev_point] < depth_img[i, j] <= depth_array[prev_point + 1]:
                    reconstr_map[i,j] = (depth_img[i,j] - depth_array[prev_point]) * params_matrix[i,j,11]
                else:
                    reconstr_map[i,j] = np.polyval(params_matrix[i,j, 8:11], depth_img[i,j])
            else:
                reconstr_map[i,j] = np.polyval(params_matrix[i,j, 8:11], depth_img[i, j])


@jit(nopython=True)
def create_depth_generator(model, model_len, model_line_len):
    i = 0
    j = 0
    while i < model_len:
        while j < model_line_len:
            yield model[i, j]
            j += 1
        i += 1
        j = 0


def add_depth_syst_and_random_noise_D415(depth_img_file, params_matrix_syst, depth_array,
                                    params_matrix_rand, len_img_file_paths, img_index, width, height, B, f):
    start_time = time.time()
    rand_model_gen = create_model_generator(params_matrix_rand, height, width)
    syst_model_gen = create_model_generator(params_matrix_syst, height, width)
    # rand_mat_gen = create_rand_generator(np.random.normal(0, 1, (height, width)), height, width)
    depth_img = np.asarray(o3d.io.read_image(depth_img_file))
    depth_gen = create_depth_generator(depth_img, height, width)
    reconstr_map = []
    for i,(params_matrix_rand_curr,params_matrix_syst_curr,depth_value) in enumerate(zip(rand_model_gen,syst_model_gen,depth_gen)):
        if not depth_value or (params_matrix_syst_curr== 0).all() or params_matrix_syst_curr[7] == 0:
            reconstr_map.append(0)  # [i//width, i - i//width*width] = 0
        elif (params_matrix_syst_curr[4:8] == 0).any():
            prev_point = np.max(np.where((params_matrix_syst_curr[4:8] > 0) == False))
            if depth_value <= depth_array[prev_point]:
                reconstr_map.append(0)  # reconstr_map[i//width, i - i//width*width] = 0
            # elif depth_array[prev_point] < depth_value <= depth_array[prev_point + 1]:
            #     reconstr_map.append((depth_value - depth_array[prev_point]) * params_matrix_syst_curr[11] + \
            #                         depth_value + rand_num * \
            #                             (depth_value - depth_array[prev_point]) * params_matrix_rand_curr[5])  # [i//width, i - i//width*width] =
            else:
                reconstr_map.append(depth_value**2 * params_matrix_syst_curr[8] + \
                                    depth_value * params_matrix_syst_curr[9] + params_matrix_syst_curr[10] + \
                                    + depth_value + np.random.normal(0, params_matrix_rand_curr[4] *
                                                                     depth_value ** 2 / (B * f)))  # [i//width, i - i//width*width] =
        else:
            reconstr_map.append(depth_value**2 * params_matrix_syst_curr[8] + \
                                    depth_value * params_matrix_syst_curr[9] + params_matrix_syst_curr[10] + \
                                    depth_value + np.random.normal(0, params_matrix_rand_curr[4] *
                                                                     depth_value ** 2 / (B * f)))  # [i//width, i - i//width*width] =
    reconstr_map = np.array(reconstr_map)
    reconstr_map = np.reshape(reconstr_map, (height, width))
    o3d.io.write_image(depth_img_file,o3d.geometry.Image(np.uint16(reconstr_map)))
    print("End 3D scan: ", img_index, "/", len_img_file_paths, "(elapsed time:", time.time() - start_time, ")")
    # return reconstr_map


def add_depth_random_noise_Azure_Kinect_slow(depth_img_file, rand_model, len_img_file_paths, img_index, width, height):
    start_time = time.time()
    depth_img = np.asarray(o3d.io.read_image(depth_img_file))
    for row in range(height):
        for col in range(width):
            depth_value = depth_img[row, col]/1000
            if depth_value > 0.5:
                depth_img[row, col]=depth_value*1000 + np.random.normal(0, rand_model[col+row*width, 0]*depth_value+rand_model[col+row*width, 1])
    o3d.io.write_image(depth_img_file, o3d.geometry.Image(np.uint16(depth_img)))
    print("End 3D scan: ", img_index, "/", len_img_file_paths, "(elapsed time:", time.time() - start_time, ")")


def add_depth_random_noise_Azure_Kinect(depth_img_file, rand_model, len_img_file_paths, img_index, width, height):
    start_time = time.time()
    depth_img = np.asarray(o3d.io.read_image(depth_img_file))
    depth_gen = create_depth_generator(depth_img, height, width)
    rand_model_gen = create_model_generator_1D(rand_model, width, height)
    reconstr_map = []
    for i, (params_matrix_rand_curr, depth_value) in enumerate(zip(rand_model_gen, depth_gen)):
        if depth_value > 500:
            rand_std = params_matrix_rand_curr[0]*depth_value/1000+params_matrix_rand_curr[1]
            if rand_std > 0:
                reconstr_map.append(depth_value + np.random.normal(0, rand_std))
            else:
                reconstr_map.append(depth_value)
        else:
            reconstr_map.append(depth_value)
    reconstr_map = np.array(reconstr_map)
    print(reconstr_map.shape)
    reconstr_map = np.reshape(reconstr_map, (height, width))
    o3d.io.write_image(depth_img_file, o3d.geometry.Image(np.uint16(reconstr_map)))
    print("End 3D scan: ", img_index, "/", len_img_file_paths, "(elapsed time:", time.time() - start_time, ")")


if __name__ == '__main__':
    intrinsic_path = "C:\\Users\\daniele.marchisotti\\OneDrive - Politecnico di Milano\\POLIMI(Dottorato)\\" \
                     "Point Cloud Processing\\Laser_scanner_simulation_new\\camera_intrinsic.json"
    dataset_path="C:\\Users\\daniele.marchisotti\\OneDrive - Politecnico di Milano\\POLIMI(Dottorato)\\" \
                    "Point Cloud Processing\\Laser_scanner_simulation_new\\datasets" \
                    "\\dataset_0000_0_000_test_cuda"
    sensors_data_main_dir="..\\sensors\\sensors_data\\"
    sensor_name="D415\\"
    sensor_data_dir=sensors_data_main_dir+sensor_name
    depth_img_path=dataset_path+"\\depth\\000000.png"
    color_img_path=dataset_path+"\\color\\000000.png"
    [cx, cy, fx, fy, width, height] = readIntrinsicJson.from_realsense([intrinsic_path])
    cx=320
    cy=240
    params_model = readBinFile.read_bin_double(sensor_data_dir + "depth_errors_parameters.bin")
    B = params_model[4]
    f = params_model[5]
    depth_array = params_model[0:4]
    import multiprocessing
    MAX_THREADS = multiprocessing.cpu_count()

    rand_model = readBinFile.read_bin_double(sensor_data_dir+"random_model.bin")
    syst_model = readBinFile.read_bin_double(sensor_data_dir + "syst_model.bin")

    rand_model = np.reshape(rand_model, (height, width, 6))
    syst_model = np.reshape(syst_model, (height, width, 12))
    test_img_depth = np.asarray(o3d.io.read_image(depth_img_path))
    start_time = time.time()
    reconstr_depth = np.array([[add_depth_random_noise(test_img_depth[i,j], B, f, rand_model[i,j,:], depth_array)
                                     for j in range(width)] for i in range(height)])
    print("One image elapsed time:", time.time() - start_time)
    reconstr_depth += add_depth_syst_noise(test_img_depth, syst_model, depth_array)
    print("One image elapsed time:",time.time()-start_time)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 3), constrained_layout=True)
    pos_0=ax[0].imshow(test_img_depth, vmax=6000)
    ax[0].set_title("Experimental random error")
    pos_1=ax[1].imshow(reconstr_depth, vmax=6000)
    ax[1].set_title("Reconstructed random error")
    fig.colorbar(pos_1, ax=ax)

    test_img_color=np.asarray(o3d.io.read_image(color_img_path))
    pc_test=pcFunc.computeRGBXYZ_PointCloud_from_RGBD_images(reconstr_depth,test_img_color,
                                                             cx, cy, fx, fy, width, height)
    o3d.visualization.draw_geometries([pc_test])
