import numpy as np
from numba import jit

def compute_line_for_each_camera_pixel(line):
    n1 = np.array([line / np.linalg.norm(line)])
    import scipy.linalg
    yz = np.array([scipy.linalg.null_space(np.array([line])).T])
    return (n1,yz)


def points_to_RGB_and_depth_map(input):
    (height, res_div2, pcPoints_divided, pcColors_divided,depth,RGB_colors,yz,analysis_type)=input
    idx_len=0
    for j in range(height):
        pcPoints_sampled = np.array(pcPoints_divided[res_div2[0][idx_len:idx_len + res_div2[1][j]]])
        pcColors_sampled = np.array(pcColors_divided[res_div2[0][idx_len:idx_len + res_div2[1][j]]])
        idx_len += res_div2[1][j]
        if pcPoints_sampled.shape[0] == 0 or pcPoints_sampled.shape[1] == 0:
            depth[j] = 0
            # Y_coord[i, j]=0
            # Z_coord[i, j]=0
            RGB_colors[j, :] = [0, 0, 0]
        else:
            if analysis_type=="rot":
                idx_depth = np.argmin(np.sqrt((yz[j,0,:].T.dot(pcPoints_sampled.T))**2+(yz[j,1,:].T.dot(pcPoints_sampled.T))**2))
            else:
                idx_depth = np.argmin(np.sqrt(np.sum(pcPoints_sampled ** 2, axis=1)))
            depth[j] = pcPoints_sampled[idx_depth, 0]
            # Y_coord[i, j] = pcPoints_sampled[i, j][idx_depth, 1];
            # Z_coord[i, j] = pcPoints_sampled[i, j][idx_depth, 2];
            RGB_colors[j, :] = np.uint8(pcColors_sampled[idx_depth, :] * 255)
    return depth,RGB_colors


def divide_pc_for_img_gpu(input):
    (i, pcPoints, my_matrix)=input
    idx_points_5 = (pcPoints[:, 1] <= (my_matrix[0, i+1]) *
                    pcPoints[:, 0]) &(pcPoints[:, 1]>= (my_matrix[0, i]) * pcPoints[:, 0])
    return idx_points_5


@jit
def divide_each_row_per_pixel_gpu_gen(inputs):
    (px_idx, x_points)=inputs
    pcPoints_sampled = x_points[px_idx]
    if pcPoints_sampled.shape[0] != 0:
        idx_depth = np.argmin(pcPoints_sampled)
        depth = pcPoints_sampled[idx_depth]
        return depth
    else:
        return 0


