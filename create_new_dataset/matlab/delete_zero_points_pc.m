clc
clear
close all

%%
pc_original=pcread("ground_truth_bedroom.ply");
minDistance=1000;
[labels,numClusters] = pcsegdist(pc_original,minDistance);
indices=labels==1;
pc_new = select(pc_original,indices);
figure;
pcshow(pc_original);
figure;
pcshow(pc_new);
pcwrite(pc_new,"ground_truth_bedroom.ply",'PLYFormat','binary');