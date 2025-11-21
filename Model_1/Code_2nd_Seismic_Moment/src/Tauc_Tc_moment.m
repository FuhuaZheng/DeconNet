function [tauc,tc]=Tauc_Tc_moment(ASTF,dt)
% this function is used to calculate tauc=2sqrt(var(ASTF))
% ASTF: have been normalized
% dt: sample interval
    ASTF=reshape(ASTF,[],1);
    t=[0:1:length(ASTF)-1]'*dt;
    s = integral_trapezoid(ASTF,dt);
    ASTF = ASTF/s;
    tc = integral_trapezoid(ASTF.*t,dt);
    tauc=2*sqrt(integral_trapezoid(ASTF.*t.^2,dt)-tc^2);%var(x)=E(x^2)-E^2(x)
end

function s=integral_trapezoid(y,dx)
    s=0;
    for i=1:length(y)-1
        s=s+(y(i+1)+y(i))/2*dx;
    end
end