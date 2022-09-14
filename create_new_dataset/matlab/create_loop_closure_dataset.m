clear
clc
close all

%% folders of new datasets
datasets_dir="C:\Users\daniele.marchisotti\"+...
"OneDrive - Politecnico di Milano\POLIMI(Dottorato)\"+...
"Point Cloud Processing\Laser_scanner_simulation_new\bedroom_datasets\";
datasets_list=["bedroom_dataset_0000_0_000","loop_closure_bedroom_dataset_0002"...
    ,"loop_closure_bedroom_dataset_0004","loop_closure_bedroom_dataset_0006"...
    ,"loop_closure_bedroom_dataset_0008","loop_closure_bedroom_dataset_0010"...
    ,"loop_closure_bedroom_dataset_0012","loop_closure_bedroom_dataset_0014"...
    ,"loop_closure_bedroom_dataset_0016","loop_closure_bedroom_dataset_0018"...
    ,"loop_closure_bedroom_dataset_0020"];
for i=1:length(datasets_list)
    datasets_folder_list(i)=datasets_dir+datasets_list(i);
    color_imgs_dir_list(i)=datasets_folder_list(i)+"\color\";
    depth_imgs_dir_list(i)=datasets_folder_list(i)+"\depth\";
    if ~exist(datasets_folder_list(i), 'dir')
       mkdir(datasets_folder_list(i))
    end
    if ~exist(color_imgs_dir_list(i), 'dir')
       mkdir(color_imgs_dir_list(i))
    end
    if ~exist(depth_imgs_dir_list(i), 'dir')
       mkdir(depth_imgs_dir_list(i))
    end
end
n_file_to_delete=[2,4,6,8,10,12,14,16,18,20];

%% generate datasets withoud loop closure
[fx,fy,cx,cy,width,height] = readIntrinsicFile(datasets_folder_list(1)+"\camera_intrinsic.json");
parfor i=2:length(depth_imgs_dir_list)
    copyfile(depth_imgs_dir_list(1), depth_imgs_dir_list(i));
    copyfile(color_imgs_dir_list(1), color_imgs_dir_list(i));
    current_depth_files=dir(depth_imgs_dir_list(i));
    current_color_files=dir(color_imgs_dir_list(i));
    current_depth_files=current_depth_files(end-n_file_to_delete(i-1)+1:end);
    current_color_files=current_color_files(end-n_file_to_delete(i-1)+1:end);
    for j=1:length(current_depth_files)
        delete(strcat(current_depth_files(j).folder,'\',current_depth_files(j).name));
        delete(strcat(current_color_files(j).folder,'\',current_color_files(j).name));
    end
    writeIntrinsicFile(fx,fy,cx,cy,width,height,datasets_folder_list(i)+"\camera_intrinsic.json")
end

