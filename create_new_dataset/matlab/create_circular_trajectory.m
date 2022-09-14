function [x,y,z,x_angle,y_angle,z_angle] = create_circular_trajectory(R,n_elem,tot_angle)
    % create 2D trajectory
    theta=linspace(0,tot_angle,n_elem+1);
    x=R.*cos(theta.*pi/180);
    y=R.*sin(theta.*pi/180);
    z=zeros(length(x));
    x_angle=zeros(length(x));
    y_angle=zeros(length(x));
    z_angle=zeros(length(x));
    %x(2:end)=x(2:end)-x(1:end-1);
    %y(2:end)=y(2:end)-y(1:end-1);
    %step=360/n_elem;
    %angle=linspace(0,45,ceil(n_elem/8)+1);
    %angle=[angle,linspace(45-step,-45,floor(n_elem/4))];
    %angle=[angle,linspace(-45+step,45,floor(n_elem/4))];
    %angle=[angle,linspace(45-step,-45,floor(n_elem/4))];
    %angle=[angle,linspace(-45+step,0,floor(n_elem/8))];
    %angle(2:end)=angle(2:end)-angle(1:end-1);
end

