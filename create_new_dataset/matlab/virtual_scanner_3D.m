function virtual_scanner_3D(pcPoints_main,pcColors_main,...
    x_traj,y_traj,z_traj,x_angle_traj,y_angle_traj,z_angle_traj,...
    datasets_folder_list,FOV_h,FOV_v,width,height,x_angle_traj2,...
    y_angle_traj2,z_angle_traj2)
    h_half_angle=tan(FOV_h/2/180*pi);
    v_half_angle=tan(FOV_v/2/180*pi);
    [my_matrix,mz_matrix]=meshgrid(linspace(-(h_half_angle+h_half_angle/(width)),(h_half_angle+h_half_angle/(width)),width+1),...
        linspace(-(v_half_angle+v_half_angle/(height)),(v_half_angle+v_half_angle/(height)),height+1));
    step_z=mz_matrix(2,1)-mz_matrix(1,1);
    step_y=my_matrix(1,2)-my_matrix(1,1);
    cx_1=width/2;
    cy_1=height/2;
    fx_1=(0-cx_1)/(my_matrix(1,1)+step_y/2);
    fy_1=(0-cy_1)/(mz_matrix(1,1)+step_z/2);
    color_imgs_dir_list=datasets_folder_list+"\color\";
    depth_imgs_dir_list=datasets_folder_list+"\depth\";
    pc_dir_list=datasets_folder_list+"\pc_real_coords\";
    tic
    for index=1:length(x_traj)
        pcPoints=[];
        pcColors=pcColors_main;
        pcPoints(:,1)=pcPoints_main(:,1)+x_traj(index);
        pcPoints(:,2)=pcPoints_main(:,2)+y_traj(index);
        pcPoints(:,3)=pcPoints_main(:,3)+z_traj(index);
        % rotations referred to point cloud coordinate system, but
        % angles referred to camera
        pcPoints = apply_rotation(x_angle_traj(index),y_angle_traj(index),z_angle_traj(index),pcPoints);
        if nargin > 13
            pcPoints = apply_rotation(x_angle_traj2(index),y_angle_traj2(index),z_angle_traj2(index),pcPoints);
        end
        idx_points=pcPoints(:,2)>=(my_matrix(1,1)-(my_matrix(1,2)-my_matrix(1,1))/2).*pcPoints(:,1);
        pcPoints=pcPoints(idx_points,:);
        pcColors=pcColors(idx_points,:);
        clear idx_points;
        idx_points_2=pcPoints(:,2)<=(my_matrix(1,width)+(my_matrix(1,2)-my_matrix(1,1))/2).*pcPoints(:,1);
        pcPoints=pcPoints(idx_points_2,:);
        pcColors=pcColors(idx_points_2,:);
        clear idx_points_2;
        idx_points_3=pcPoints(:,3)>=(mz_matrix(1,1)-(mz_matrix(2,1)-mz_matrix(1,1))/2).*pcPoints(:,1);
        pcPoints=pcPoints(idx_points_3,:);
        pcColors=pcColors(idx_points_3,:);
        clear idx_points_3;
        idx_points_4=pcPoints(:,3)<=(mz_matrix(height,1)+(mz_matrix(2,1)-mz_matrix(1,1))/2).*pcPoints(:,1);
%         idx_points_4=pcPoints(:,3)<(mz_matrix(height,1)).*pcPoints(:,1);
        pcPoints=pcPoints(idx_points_4,:);
        pcColors=pcColors(idx_points_4,:);
        clear idx_points_4;
        pcPoints_divided={};
        pcColors_divided={};
        parfor i=1:width
            idx_points_5=(pcPoints(:,2)<=(my_matrix(1,i)-(step_y)/2)*pcPoints(:,1))==...
                (pcPoints(:,2)>=(my_matrix(1,i)+(step_y)/2)*pcPoints(:,1));
            pcPoints_divided{i}=pcPoints(idx_points_5,:);
            pcColors_divided{i}=pcColors(idx_points_5,:);
        end
        toc
        clear pcPoints pcColors;
        % sample point clouds along height
        pcPoints_sampled={};
        pcColors_sampled={};
        parfor i=1:length(pcPoints_divided)
            for j=1:height
                idx_points_6=(pcPoints_divided{i}(:,3)>=(mz_matrix(j,1)-(step_z)/2)*...
                pcPoints_divided{i}(:,1))==(pcPoints_divided{i}(:,3)<=(mz_matrix(j,1)+(step_z)/2)...
                *pcPoints_divided{i}(:,1));
                pcPoints_sampled{j,i}=pcPoints_divided{i}(idx_points_6,:);
                pcColors_sampled{j,i}=pcColors_divided{i}(idx_points_6,:);
            end
        end
        clear pcPoints_divided pcColors_divided;
        [rows,cols]=size(pcPoints_sampled);
        depth=[];
        Y_coord=[];
        Z_coord=[];
        RGB_colors={};
        for i=0:rows-1
            for j=0:cols-1
                if size(pcPoints_sampled{i+1,j+1},1)==0
                    depth(i+1,j+1)=0;
                    RGB_colors{i*cols+j+1}=[0,0,0];
                else
                    [~,idx_depth]=min(sqrt(sum(pcPoints_sampled{i+1,j+1}.^2,2)));
                    depth(i+1,j+1)=pcPoints_sampled{i+1,j+1}(idx_depth,1);
                    RGB_colors{i*cols+j+1}=uint8(pcColors_sampled{i+1,j+1}(idx_depth,:));
                end
            end
        end
        clear pcPoints_sampled pcColors_sampled;
        parfor i=1:length(RGB_colors)
            RGB_colors_array(i,:)=RGB_colors{i};
        end
        clear RGB_colors;
        depth=flip(depth,1);

        color_img=cat(3,reshape(RGB_colors_array(:,1),[width,height]),reshape(RGB_colors_array(:,2),...
        [width,height]),reshape(RGB_colors_array(:,3),[width,height]));
        color_img=flip(color_img,2);
        color_img=cat(3,color_img(:,:,1)',color_img(:,:,2)',color_img(:,:,3)');
        clear RGB_colors_array;

        gray_img=rgb2gray(color_img);
        gray_mask=gray_img==0;
        clear gray_img;
        s=regionprops(gray_mask,'PixelList');
        clear gray_mask;
        threshold=100;
        neighbourhood_dim=20;
        for i=1:length(s)
            if (length(s(i).PixelList)<threshold && not(isempty(s(i).PixelList)))
                for j=1:length(s(i).PixelList(:,1))
                    color_img(s(i).PixelList(j,2),s(i).PixelList(j,1),:)...
                    = get_neighbourhood_median_color(neighbourhood_dim,...
                    [s(i).PixelList(j,2),s(i).PixelList(j,1)],color_img,width,height);
                    depth(s(i).PixelList(j,2),s(i).PixelList(j,1))=...
                    get_neighbourhood_median_depth(neighbourhood_dim,...
                    [s(i).PixelList(j,2),s(i).PixelList(j,1)],depth,width,height);
                end
            end
        end
        clear s;
        imwrite(uint8(color_img),color_imgs_dir_list+num2str(index-1,'%06d')+".png");
        imwrite(uint16(depth*1000),depth_imgs_dir_list+num2str(index-1,'%06d')+".png");
        toc
    end 
    intrinsic_struct_new.width=width;
    intrinsic_struct_new.height=height;
    intrinsic_struct_new.intrinsic_matrix=[fx_1,0,0,0,fy_1,0,cx_1,cy_1,1];
    json_txt=jsonencode(intrinsic_struct_new);
    fid=fopen(datasets_folder_list+"\camera_intrinsic.json",'w');
    fwrite(fid,json_txt);
    fclose(fid);
end

