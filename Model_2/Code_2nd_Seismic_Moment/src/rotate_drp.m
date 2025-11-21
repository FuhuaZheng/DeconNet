function [xnew,ynew] = rotate_drp(x,y,drp)
    xnew = x.*cos(drp) + y.*sin(drp);
    ynew = -x.*sin(drp) + y.*cos(drp);
    return;
end
