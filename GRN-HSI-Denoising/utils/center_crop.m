function I2 = center_crop(I,h,w)

[H,W,~] = size(I);
x_cent = round(H/2);
y_cent = round(W/2);
xmin = x_cent - floor(h/2);
ymin = y_cent - floor(w/2);
I2 = I(xmin:xmin+h-1,ymin:ymin+h-1,:);




