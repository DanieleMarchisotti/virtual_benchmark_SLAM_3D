clear
clc
close all

my=0;
mz=0;
n1=[1,my,mz];
for i=1:length(n1(:,1))
   n1(i,:)=n1(i,:)./norm(n1(i,:));
end
X_space=linspace(0,3,10);
Y_space=X_space*my;
Z_space=X_space*mz;
ex_point=[1.68768,3,4];
figure;
plot3(X_space,Y_space,Z_space);
hold on;
plot3(ex_point(1),ex_point(2),ex_point(3),'o');
quiver3(0,0,0,n1(1),n1(2),n1(3));
yz=null(n1)';
prj_point=ex_point*yz';
sqrt(sum(prj_point.^2,2))