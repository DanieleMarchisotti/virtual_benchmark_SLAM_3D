function [x,y,z,x_angle,y_angle,z_angle] = create_linear_rot_trajectory(n_elem_transl,n_elem_rot,transl_dim)
    x=zeros(n_elem_transl+n_elem_rot+1,1);
    y=linspace(0,transl_dim,(n_elem_transl+2)/2);
    y=[y ones(1,n_elem_rot/2)*transl_dim];
    y=[y transl_dim-linspace(transl_dim/(n_elem_transl/2),transl_dim,(n_elem_transl)/2)];
    y=[y zeros(1,n_elem_rot/2)];
    z=zeros(n_elem_transl+n_elem_rot+1,1);
    % step=tot_angle/n_elem;
    step_angle=180/(n_elem_rot/2);
    x_angle=zeros(1,(n_elem_transl+2)/2);
    x_angle=[x_angle linspace(step_angle,180,n_elem_rot/2)];
    x_angle=[x_angle 180*ones(1,n_elem_transl/2)];
    x_angle=[x_angle linspace(180+step_angle,360,n_elem_rot/2)];
    y_angle=zeros(n_elem_transl+n_elem_rot+1,1);
    z_angle=zeros(n_elem_transl+n_elem_rot+1,1);
end

