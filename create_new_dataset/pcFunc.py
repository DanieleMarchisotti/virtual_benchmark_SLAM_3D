import struct
import numpy as np
import readIntrinsicJson
import open3d as o3d


def computePointCloud_bin(filePathSaved,cx,cy,fx,fy,height,width):
    with open(filePathSaved, mode='rb') as file: # b is important -> binary
        fileContent = file.read()
    data=struct.unpack("H" * ((len(fileContent)) // 2), fileContent)
    Image=np.reshape(data,(height,width))
    # cv2.imshow('title',Image/4200)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    X=np.empty((height,width))
    Y=np.empty((height,width))
    Z=np.empty((height,width))
    for m in range(len(Image)):
        for n in range(len(Image[0])):
            Z[m][n]=Image[m][n]
            X[m][n]=(n-cx)*Z[m][n]/fx
            Y[m][n]=(m-cy)*Z[m][n]/fy
    Z=np.reshape(Z,(width*height,1))
    Y=np.reshape(Y,(width*height,1))
    X=np.reshape(X,(width*height,1))
    data3D=[X,Y,Z]
    return data3D


def computePointCloud_from_coords(depth_image,cx,cy,fx,fy,height,width):
    X=np.empty((len(depth_image),len(depth_image[0])))
    Y=np.empty((len(depth_image),len(depth_image[0])))
    Z=np.empty((len(depth_image),len(depth_image[0])))
    for m in range(len(depth_image)):
        for n in range(len(depth_image[0])):
            Z[m][n]=depth_image[m][n]
            Y[m][n]=(m-cy)*Z[m][n]/fy
            X[m][n]=(n-cx)*Z[m][n]/fx
    return X,Y,Z


def computePointCloud_from_depth(depth_file,cx,cy,fx,fy,width,height,pcd=None):
    depth_image=o3d.io.read_image(depth_file)
    depth=np.asarray(depth_image)
    X,Y,Z=computePointCloud_from_coords(depth,cx,cy,fx,fy,width,height)
    X=np.reshape(X,(width*height,1))
    Y=np.reshape(Y,(width*height,1))
    Z=np.reshape(Z,(width*height,1))
    coord3D=np.concatenate([X,Y,Z],axis=1)
    if pcd is None:
        pcd = o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(coord3D)
    return pcd


def computeRGBXYZ_PointCloud_from_depth(depth_file,color_file, cx, cy, fx, fy, width, height):
    color = np.asarray(o3d.io.read_image(color_file))
    data3D=computePointCloud_from_depth(depth_file,cx,cy,fx,fy,width,height)
    R_color=np.reshape(color[:,:,0].transpose(),[np.prod(color[:,:,0].shape),1])
    G_color=np.reshape(color[:,:,1].transpose(),[np.prod(color[:,:,1].shape),1])
    B_color=np.reshape(color[:,:,2].transpose(),[np.prod(color[:,:,2].shape),1])
    color_pc=np.squeeze(np.array([R_color,G_color,B_color])).transpose()
    data3D.colors=o3d.utility.Vector3dVector(np.float64(color_pc)/255)
    return data3D


def computeRGBXYZ_PointCloud_from_RGBD_images(depth,color, cx, cy, fx, fy, width, height):
    X,Y,Z=computePointCloud_from_coords(depth,cx,cy,fx,fy,width,height)
    X=np.reshape(X,(width*height,1))
    Y=np.reshape(Y,(width*height,1))
    Z=np.reshape(Z,(width*height,1))
    coord3D=np.concatenate([X,Y,Z],axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(coord3D)
    R_color=np.reshape(color[:,:,0],[np.prod(color[:,:,0].shape),1])
    G_color=np.reshape(color[:,:,1],[np.prod(color[:,:,1].shape),1])
    B_color=np.reshape(color[:,:,2],[np.prod(color[:,:,2].shape),1])
    color_pc=np.squeeze(np.array([R_color,G_color,B_color])).transpose()
    pcd.colors=o3d.utility.Vector3dVector(np.float64(color_pc)/255)
    return pcd


def rotate_point_cloud_by_angles_XYZRGB(X_rot_ideal,Y_rot_ideal,Z_rot_ideal,data3D_2):
    # rotations referred to camera coordinate system
    data3D_2_points = np.asarray(data3D_2.points).copy()
    data3D_2_colors = np.asarray(data3D_2.colors).copy()
    x_rot_matrix = np.array([[1, 0, 0],
                             [0, np.cos(X_rot_ideal * np.pi / 180), - np.sin(X_rot_ideal * np.pi / 180)],
                             [0, np.sin(X_rot_ideal * np.pi / 180), np.cos(X_rot_ideal * np.pi / 180)]])
    y_rot_matrix = np.array([[np.cos(Y_rot_ideal * np.pi / 180), 0, np.sin(Y_rot_ideal * np.pi / 180)],
                             [0,1,0],
                             [-np.sin(Y_rot_ideal * np.pi / 180), 0, np.cos(Y_rot_ideal * np.pi / 180)]])
    z_rot_matrix = np.array([[np.cos(Z_rot_ideal * np.pi / 180), -np.sin(Z_rot_ideal * np.pi / 180), 0],
                             [np.sin(Z_rot_ideal * np.pi / 180), np.cos(Z_rot_ideal * np.pi / 180), 0],
                             [0, 0, 1]])

    rot_matrix = x_rot_matrix.dot(y_rot_matrix)
    rot_matrix=rot_matrix.dot(z_rot_matrix)
    data3D_2_points = data3D_2_points.dot(rot_matrix)
    data3D_2.colors = o3d.utility.Vector3dVector(data3D_2_colors)
    data3D_2.points=o3d.utility.Vector3dVector(data3D_2_points)
    return data3D_2,data3D_2_colors


def rotate_point_cloud_by_angles_XYZ(X_rot_ideal,Y_rot_ideal,Z_rot_ideal,data3D_2):
    # rotations referred to camera coordinate system
    data3D_2_points = np.asarray(data3D_2.points).copy()
    x_rot_matrix = np.array([[1, 0, 0],
                             [0, np.cos(X_rot_ideal * np.pi / 180), - np.sin(X_rot_ideal * np.pi / 180)],
                             [0, np.sin(X_rot_ideal * np.pi / 180), np.cos(X_rot_ideal * np.pi / 180)]])
    y_rot_matrix = np.array([[np.cos(Y_rot_ideal * np.pi / 180), 0, np.sin(Y_rot_ideal * np.pi / 180)],
                             [0,1,0],
                             [-np.sin(Y_rot_ideal * np.pi / 180), 0, np.cos(Y_rot_ideal * np.pi / 180)]])
    z_rot_matrix = np.array([[np.cos(Z_rot_ideal * np.pi / 180), -np.sin(Z_rot_ideal * np.pi / 180), 0],
                             [np.sin(Z_rot_ideal * np.pi / 180), np.cos(Z_rot_ideal * np.pi / 180), 0],
                             [0, 0, 1]])
    rot_matrix = x_rot_matrix.dot(y_rot_matrix)
    rot_matrix=rot_matrix.dot(z_rot_matrix)
    data3D_2_points = data3D_2_points.dot(rot_matrix)
    data3D_3=o3d.geometry.PointCloud()
    data3D_3.points=o3d.utility.Vector3dVector(data3D_2_points)
    return data3D_3


def shift_point_cloud(X_shift,Y_shift,Z_shift,data3D=None):
    if data3D is None:
        data3D_points = np.asarray(data3D.points).copy()
    else:
        data3D_points = np.asarray(data3D.points)
    data3D_points[:, 0] += X_shift
    # target_points[:,0]=-target_points[:,0]
    data3D_points[:, 1] += Y_shift
    data3D_points[:, 2] += Z_shift
    if data3D is None:
        data3D=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data3D_points))
    else:
        data3D.points=o3d.utility.Vector3dVector(data3D_points)
    return data3D


def convert_ply_pc_double_to_float(file_path,file_path_new):
    with open(file_path,mode='r') as file:  # b is important -> binary
        fileContent = file.read()
    fileContent=fileContent.replace("double","float")
    with open(file_path_new,mode='r') as file:  # b is important -> binary
        file.write(fileContent)
