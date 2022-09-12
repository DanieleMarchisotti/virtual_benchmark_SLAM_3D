import open3d as o3d
import readIntrinsicJson
import pcFunc
import numpy as np

main_folder='C:\\Users\\daniele.marchisotti\\OneDrive - Politecnico di Milano\\POLIMI(Dottorato)\\' \
            'Point Cloud Processing\\Laser_scanner_simulation_new\\bedroom_datasets'
dataset="bedroom_dataset_0000_0_000"
rgbd_folder=main_folder+"\\"+dataset
color_folder=rgbd_folder+"\\color"
depth_folder=rgbd_folder+"\\depth"
img_example_folder=main_folder+"\\images_example_datasets"
color = np.asarray(o3d.io.read_image(color_folder+"\\000020.png"))
depth = np.asarray(o3d.io.read_image(depth_folder+"\\000020.png"))

intrinsicsFilePath=rgbd_folder+"\\camera_intrinsic.json"
cx,cy,fx,fy,width,height=readIntrinsicJson.from_realsense([intrinsicsFilePath])
cx=320
cy=240

data3D=pcFunc.computePointCloud_from_depth(depth_folder+"\\000020.png",cx,cy,fx,fy,width,height)
R_color=np.reshape(color[:,:,0],[np.prod(color[:,:,0].shape),1])
G_color=np.reshape(color[:,:,1],[np.prod(color[:,:,1].shape),1])
B_color=np.reshape(color[:,:,2],[np.prod(color[:,:,2].shape),1])
color_pc=np.array([R_color,G_color,B_color])
color_pc=np.squeeze(color_pc)
color_pc=color_pc.transpose()
data3D.colors=o3d.utility.Vector3dVector(np.float64(color_pc)/255)
flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
data3D,_=pcFunc.rotate_point_cloud_by_angles_XYZRGB(-22.5,-45,0,data3D)
data3D.transform(flip_transform)
view=o3d.visualization.Visualizer()
view.create_window()
view.add_geometry(data3D)
view.get_render_option().load_from_json("renderoption.json")
view.update_geometry(data3D)
# view.poll_events()
# view.update_renderer()
view.run()
view.capture_screen_image(img_example_folder+"\\"+dataset+".png")
view.destroy_window()
# data3D=pcFunc.read_float32_bin_point_cloud_from_depth_eye(filePathSaved)
# o3d.visualization
# o3d.visualization.draw_geometries([data3D])

