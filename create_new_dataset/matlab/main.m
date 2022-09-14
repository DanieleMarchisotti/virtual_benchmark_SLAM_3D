clear
clc
close all

%% read camera_intrinsic.json file
[fx,fy,cx,cy,width,height] = readIntrinsicFile("camera_intrinsic.json");

%% compute FOV
FOV_h=2*atan2(width,(2*fx))*180/pi;
FOV_v=2*atan2(height,(2*fy))*180/pi;

%% read point cloud to sample
pcToSample=pcread("..\input\apt_subset_low.ply");
figure;
pcshow(pcToSample);
xlabel("X [m]");
ylabel("Y [m]");
zlabel("Z [m]");

%%

originX=pcToSample.Location(find(pcToSample.Location(:,1)==max(pcToSample.Location(:,1))),1);%mean(pcToSample.Location(:,1));
originY=mean(pcToSample.Location(:,2));
originZ=mean(pcToSample.Location(:,3));
% downsample point cloud
gridStep=0.0001;
% pcToSample = pcdownsample(pcToSample,'gridAverage',gridStep);
% pcToSample = pcdenoise(pcToSample,'NumNeighbors',4,'Threshold',0.08);
%%
tot_angle=360;
n_elem_360=[30];
n_elem=round(tot_angle./(360./n_elem_360));
R_increment=[0];
R=0.5;
n_elem_transl=30;
n_elem_rot=30;
transl_dim=2;
for i=1:length(n_elem)
    for j=1:length(R_increment)
        [x_traj{i,j},y_traj{i,j},z_traj{i,j},x_angle_traj{i,j},y_angle_traj{i,j},...
            z_angle_traj{i,j}] = create_circular_trajectory(...
            R,n_elem_transl,tot_angle);
    end
end

%%
% center the point cloud into the origin
D = zeros(size(pcToSample.Location));
D(:,1)=-originX(1);
D(:,1)=D(:,1)-0.5;
D(:,2)=D(:,2)+0.15;
D(:,3)=-originZ(1);
pcColors_main=pcToSample.Color;
pcPoints_main=pcToSample.Location+D;
pcPoints_main(:,1)=-pcPoints_main(:,1);
pcMain=pointCloud(pcPoints_main);
pcMain.Color=pcToSample.Color;
clear pcToSample;
figure;
pcshow(pcMain);
xlabel("X [m]");
ylabel("Y [m]");
zlabel("Z [m]");
%%
% save images directory
datasets_dir="..\datasets\";
for i=1:length(n_elem)
    datasets_list(i)="dataset_D415";
end
for i=1:length(datasets_list)
    datasets_folder_list(i)=datasets_dir+datasets_list(i);
    color_imgs_dir_list(i)=datasets_folder_list(i)+"\color\";
    depth_imgs_dir_list(i)=datasets_folder_list(i)+"\depth\";
    pc_dir_list(i)=datasets_folder_list(i)+"\pc_real_coords\";
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
clear pcMain;
%%
for i=1:length(n_elem)
    for j=1:length(R_increment)
        virtual_scanner_3D(pcPoints_main,pcColors_main,x_traj{i,j},y_traj{i,j},z_traj{i,j},x_angle_traj{i,j},...
            y_angle_traj{i,j},z_angle_traj{i,j},datasets_folder_list(i)...
            ,FOV_h,FOV_v,width,height);
    end
end

