function [pcPoints] = apply_rotation(x_angle,y_angle,z_angle,pcPoints)
    x_rot_matrix=[cos(x_angle*pi/180) -sin(x_angle*pi/180) 0;
                sin(x_angle*pi/180) cos(x_angle*pi/180) 0;
                0 0 1];
    y_rot_matrix=[cos(z_angle*pi/180) 0 sin(z_angle*pi/180);
        0 1 0;
        -sin(z_angle*pi/180) 0 cos(z_angle*pi/180)];
    z_rot_matrix=[1 0 0;
        0 cos(y_angle*pi/180) -sin(y_angle*pi/180);
        0 sin(y_angle*pi/180) cos(y_angle*pi/180)];
    rot_matrix=x_rot_matrix*y_rot_matrix*z_rot_matrix;
    pcPoints=(rot_matrix*pcPoints')';
end

