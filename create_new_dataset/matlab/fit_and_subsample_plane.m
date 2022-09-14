clear
close all
clc

%%

pcTest=pcread("apt_low_new_traj_4_table.ply");
figure;
pcshow(pcTest);
xlabel("X [m]");
ylabel("Y [m]");
zlabel("Z [m]");
hold on;

plane_model=pcfitplane(pcTest,0.01);

[X,Y] = meshgrid(min(pcTest.Location(:,1)):0.001:max(pcTest.Location(:,1)),min(pcTest.Location(:,2)):0.001:max(pcTest.Location(:,2)));
Z=(-plane_model.Parameters(4)-plane_model.Parameters(2)*Y-plane_model.Parameters(1)*X)/plane_model.Parameters(3);
surf(X,Y,Z);
%%
[row,col]=size(X);
pcPoints=pcTest.Location;
parfor i=1:row
    for j=1:col
        [~,idx_min]=min(vecnorm(([X(i,j),Y(i,j),Z(i,j)]-pcPoints)'));
        % [~,idx_min]=min(sqrt(sum(([X(i,j),Y(i,j),Z(i,j)]-pcPoints)'.^2)));
        idx_points(i,j)=idx_min;
    end
    disp(i);
end
% [row,col]=size(X);
% pcPoints=pcTest.Location;
% parfor i=1:row
%     for j=1:col
%         idx_min=knnsearch(Mdl,[X(i,j),Y(i,j),Z(i,j)],'K',1)
%         idx_points(i,j)=idx_min;
%         disp(j);
%     end
%     disp(i);
% end

%%
[idx_row,idx_col]=size(idx_points);
idx_points1=reshape(idx_points,[idx_row*idx_col,1]);
X=reshape(X,[idx_row*idx_col,1]);
Y=reshape(Y,[idx_row*idx_col,1]);
Z=reshape(Z,[idx_row*idx_col,1]);
colors=pcTest.Color(idx_points1,:);
points=[X,Y,Z];
pcRes=pointCloud(points);
pcRes.Color=colors;
figure;
pcshow(pcTest);
xlabel("X [m]");
ylabel("Y [m]");
zlabel("Z [m]");
figure;
pcshow(pcRes);
xlabel("X [m]");
ylabel("Y [m]");
zlabel("Z [m]");

