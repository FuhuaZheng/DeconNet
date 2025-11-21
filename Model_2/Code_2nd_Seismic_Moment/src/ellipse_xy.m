function [x,y]=ellipse_xy(Lc,Wc,dtheta)
    theta = [0:dtheta:360]'./180.*pi;
    x = Lc.*cos(theta);
    y = Wc.*sin(theta);
end