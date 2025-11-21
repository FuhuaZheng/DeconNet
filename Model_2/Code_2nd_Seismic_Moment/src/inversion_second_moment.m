function [L_c,W_c,dir,drp,tau,v0,mv0,pre,G]=inversion_second_moment(toa,az,taus,vs,strike,dip)
%%
d=(reshape(taus,[],1)/2).^2;
G=[];
zz = strike;
zz = zz*pi/180;
rotm1=[cos(zz), sin(zz),  0;
    -sin(zz), cos(zz), 0;
    0,0,1];
zz = dip;
zz = zz*pi/180;
rotm2=[1, 0, 0;
    0, cos(zz), sin(zz);
    0, -sin(zz), cos(zz)];

for i = 1:length(toa)
    rh=sin(toa(i)*pi/180);
    rv = cos(toa(i)*pi/180);%z positive down
    rsvz=rv;
    rsvx=rh*cos(az(i)*pi/180);%x positive north
    rsvy=rh*sin(az(i)*pi/180);%y positive east

    s=(1/vs(i))*[rsvx,rsvy,rsvz]';
    %%
    s=rotm1*s;
    %%
    s=rotm2*s;
    %        s(2) = s(2); s(3) = s(3);
    % u(0,2) = u(0,2) - 2s*u(1,1) +s*u(2,0)*s;
    G=[G;1,-2*s(1),-2*s(2),s(1)^2,2*s(1)*s(2),s(2)^2];
    %G[tt,xt,yt,xx,xy,yy]'=d
end
%     [xopt,L_c,W_c,vx,vy,tau]=seconds_2d_v2(G,d);
[xopt,L_c,W_c,vx,vy,tau]=seconds_2d_cvx(G,d);

pre = 2*sqrt(G*reshape(xopt(1:6),[],1));
drp = atan2(-vy,vx);
drp = drp/pi*180;
if(drp<0)
    drp = drp+360;
end
dir = sqrt(vx^2+vy^2)/L_c*tau;
%     v0 = L_c/tau;
%vsr = L_c/tau*(1+dir)/2;
v0=xopt(2:3)/xopt(1); mv0=sqrt(sum(v0.^2));

end

