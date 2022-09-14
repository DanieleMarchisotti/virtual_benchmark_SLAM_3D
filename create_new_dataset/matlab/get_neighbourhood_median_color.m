function [pixel_median] = get_neighbourhood_median_color(neighbourhood_dim,...
Pixel,color_img,width,height)
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
    neighbourhood=color_img(left_idx:right_idx,top_idx:bottom_idx,:);
    R_med=reshape(neighbourhood(:,:,1),[numel(neighbourhood(:,:,1)),1]);
    G_med=reshape(neighbourhood(:,:,2),[numel(neighbourhood(:,:,2)),1]);
    B_med=reshape(neighbourhood(:,:,3),[numel(neighbourhood(:,:,3)),1]);
    pixel_median(1)=median(R_med(R_med~=0));
    pixel_median(2)=median(G_med(G_med~=0));
    pixel_median(3)=median(B_med(B_med~=0));
end

