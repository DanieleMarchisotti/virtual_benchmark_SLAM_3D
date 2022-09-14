function [intrinsic_struct] = readIntrinsicJson(filename)
%READINTRINSICJSON Summary of this function goes here
%   Detailed explanation goes here
    fid=fopen(filename,'r');
    text=fread(fid,inf);
    fclose(fid);
    text = char(text');
    intrinsic_struct = jsondecode(text);
end

