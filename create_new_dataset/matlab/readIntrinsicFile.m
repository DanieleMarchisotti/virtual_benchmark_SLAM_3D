function [fx,fy,cx,cy,width,height] = readIntrinsicFile(filename)
    [intrinsic_struct] = readIntrinsicJson(filename);
    fx=intrinsic_struct.intrinsic_matrix(1);
    fy=intrinsic_struct.intrinsic_matrix(5);
    cx=intrinsic_struct.intrinsic_matrix(7);
    cy=intrinsic_struct.intrinsic_matrix(8);
    width=intrinsic_struct.width;
    height=intrinsic_struct.height;
end

