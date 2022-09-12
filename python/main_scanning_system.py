import readIntrinsicJson
import numpy as np
import open3d as o3d
import trajectory_analysis_methods
import os

sensors_dict={
    0: "Intel RealSense D415",
    1: "Azure Kinect DK"
}

if __name__ == '__main__':
    input_folder = "..\\input"
    # read camera_intrinsic.json file
    [cx,cy,fx,fy,width,height] = readIntrinsicJson.from_realsense([input_folder + "\\camera_intrinsic.json"])

    # compute FOV
    FOV_h=2*np.arctan2(width,(2*fx))*180/np.pi
    FOV_v=2*np.arctan2(height,(2*fy))*180/np.pi

    # for scene 1
    # read point cloud to sample
    pcToSample=o3d.io.read_point_cloud(input_folder+"\\apt_subset_low.ply")
    # o3d.visualization.draw_geometries([pcToSample])
    originX=max(np.asarray(pcToSample.points)[:,0])
    originY=np.mean(np.asarray(pcToSample.points)[:,1])
    originZ=np.mean(np.asarray(pcToSample.points)[:,2])
    # downsample point cloud
    cl, ind = pcToSample.remove_statistical_outlier(nb_neighbors=4,std_ratio=2.0)
    data_points=np.asarray(pcToSample.points)[ind,:]
    data_colors=np.asarray(pcToSample.colors)[ind,:]
    # tot_angle=360
    # n_elem=20
    analysis_type="transl"
    # R=0.5
    # ideal_traj = trajectory_analysis_methods.Traj(analysis_type, n_elem, tot_angle, "open3d")
    # ideal_traj.ideal_circular_traj(R,[0,0])
    # ideal_traj.X=-ideal_traj.X
    # ideal_traj.Z=-ideal_traj.Z
    ideal_traj = trajectory_analysis_methods.Traj(analysis_type, 0, 360, "open3d")
    ideal_traj.read_traj_input("..\\input\\ground_truth_trajectory.txt")
    ideal_traj.apply_handheld_noise(input_folder + "\\handheld_noise")
    data_points[:,0]-=originX
    data_points[:,0]-=0.5
    data_points[:,1]+=0.15
    data_points[:,2]-=originZ
    data_points[:,0]=-data_points[:,0]
    pcToSample=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data_points))
    pcToSample.colors=o3d.utility.Vector3dVector(data_colors)
    # save images directory
    datasets_dir = "..\\datasets\\"
    datasets_list = ["dataset_0000_0_000"]

    # for scene 2
    # read point cloud to sample
    # pcToSample=o3d.io.read_point_cloud(main_folder+"\\bedroom.ply")
    # o3d.visualization.draw_geometries([pcToSample])
    # originX=np.mean(np.asarray(pcToSample.points)[:,0])
    # originY=np.mean(np.asarray(pcToSample.points)[:,1])
    # originZ=np.mean(np.asarray(pcToSample.points)[:,2])
    # tot_angle=360
    # n_elem=360
    # analysis_type="rot"
    # ideal_traj = trajectory_analysis_methods.Traj(analysis_type, n_elem, tot_angle)
    # ideal_traj.ideal_rotation_traj_no_shifts()
    # data_points=np.asarray(pcToSample.points)
    # data_colors=np.asarray(pcToSample.colors)
    # data_points[:,0]-=originX
    # data_points[:,1]-=originY
    # data_points[:,2]-=originZ
    # data_points[:,0]=-data_points[:,0]
    # pcToSample=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data_points))
    # pcToSample.colors=o3d.utility.Vector3dVector(data_colors)
    # # save images directory
    # datasets_dir="C:\\Users\\daniele.marchisotti\\OneDrive - Politecnico di Milano\\POLIMI(Dottorato)\\" \
    #              "Point Cloud Processing\\Laser_scanner_simulation_new\\bedroom_datasets\\"
    # datasets_list=["dataset_0000_0_000_test1"]
    # x_traj{i,j}=-x_traj{i,j};
    # y_traj{i,j}=-y_traj{i,j};

    # for scene 2
    # originX=np.mean(np.asarray(pcToSample.points)[:,0])
    # originY=np.mean(np.asarray(pcToSample.points)[:,1])
    # originZ=np.mean(np.asarray(pcToSample.points)[:,2])

    # angle_traj=-angle_traj;
    # center the point cloud into the origin

    datasets_folder_list=[]
    color_imgs_dir_list=[]
    depth_imgs_dir_list=[]
    pc_dir_list=[]
    for i,dataset in enumerate(datasets_list):
        datasets_folder_list.append(datasets_dir+dataset)
        color_imgs_dir_list.append(datasets_folder_list[i]+"\\color\\")
        depth_imgs_dir_list.append(datasets_folder_list[i]+"\\depth\\")
        pc_dir_list.append(datasets_folder_list[i]+"\\pc_real_coords\\")
        if not os.path.exists(datasets_folder_list[i]):
            os.mkdir(datasets_folder_list[i])
        if not os.path.exists(color_imgs_dir_list[i]):
            os.mkdir(color_imgs_dir_list[i])
        if not os.path.exists(depth_imgs_dir_list[i]):
            os.mkdir(depth_imgs_dir_list[i])
    pcColors_main=np.asarray(pcToSample.colors).copy()
    pcPoints_main=np.asarray(pcToSample.points).copy()
    del pcToSample
    d_noise_status = input("Apply depth noise on the dataset? (Y/N)")
    if d_noise_status == "Y" or d_noise_status == "y":
        print("------- Available sensors to simulate --------")
        for i, camera in enumerate(sensors_dict):
            print(i, ":", sensors_dict[i])
        camera_idx = int(input("Insert the sensor index to simulate: "))
        camera_type = sensors_dict[camera_idx]
    else:
        camera_type = ""
    import virtual_3D_scanner
    virtual_3D_scanner.scan_from_trajectory(FOV_h,FOV_v,width,height,
                                                  datasets_folder_list[0],ideal_traj,pcColors_main,pcPoints_main,
                                                  d_noise_status,camera_type)
