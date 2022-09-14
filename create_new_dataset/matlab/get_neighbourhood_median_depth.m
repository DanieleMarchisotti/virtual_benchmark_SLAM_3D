function [depth_median] = get_neighbourhood_median_depth(neighbourhood_dim,...
Pixel,depth_img,width,height)
    left_idx=round(Pixel(1)-neighbourhood_dim/2);
    right_idx=round(Pixel(1)+neighbourhood_dim/2);
    top_idx=round(Pixel(2)-neighbourhood_dim/2);
    bottom_idx=round(Pixel(2)+neighbourhood_dim/2);
    if left_idx<1
        left_idx=1;
    end
    if right_idx>height
        right_idx=height;
    end
    if top_idx<1
        top_idx=1;
    end
    if bottom_idx>width
        bottom_idx=width;
    end
    neighbourhood=depth_img(left_idx:right_idx,top_idx:bottom_idx);
    reg_med=reshape(neighbourhood,[numel(neighbourhood),1]);
    depth_median=median(reg_med(reg_med~=0));
end

