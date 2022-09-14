function writeIntrinsicFile(fx,fy,cx,cy,width,height,filename)
    intrinsic_struct_new.width=width;
    intrinsic_struct_new.height=height;
    intrinsic_struct_new.intrinsic_matrix=[fx,0,0,0,fy,0,cx,cy,1];
    json_txt=jsonencode(intrinsic_struct_new);
    fid=fopen(filename,'w');
    fwrite(fid,json_txt);
    fclose(fid);
end