clear
clc
close all

%%

I=single(imread('C:\Users\daniele.marchisotti\OneDrive - Politecnico di Milano\POLIMI(Dottorato)\Point Cloud Processing\Laser_scanner_simulation_new\datasets\prova_dataset_0000\depth\000000.png'));
height=480;
width=640;
x_coord=[];
y_coord=[];
cx=320;
cy=240;
for row=1:height
    for col=1:width
        y_coord(row,col)=(row-cy)*I(row,col)/fy;
        x_coord(row,col)=(col-cx)*I(row,col)/fx;
    end
end
x_coord=reshape(x_coord,[numel(x_coord),1]);
y_coord=reshape(y_coord,[numel(y_coord),1]);
I_array=reshape(I,[numel(I),1]);
pc_test2=pointCloud([x_coord,y_coord,single(I_array)]);
figure;
pcshow(pc_test2);