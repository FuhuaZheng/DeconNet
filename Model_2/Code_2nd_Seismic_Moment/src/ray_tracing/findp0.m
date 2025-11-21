function p0 = findp0(x,pm) 

zero = 1e-7;
p1 = 0;
p2 = pm;
while (abs(p1-p2)>zero)
    p0 = 0.5*(p1+p2);
    dtdp0 = dtdp(x,p0);
    if(abs(dtdp0)<=zero)
        break;
    end
    if(dtdp0>0)
        p1 = p0;
    else
        p2 = p0;
    end
end

%--------------------------------------------------%
function dtdp0 = dtdp(x,p)

global ns ray_len vps topp bttm

dtdp0 = 0;
pp = p*p;
for i = topp:bttm
    dtdp0 = dtdp0-ray_len(i,1)/sqrt(vps(i,1)-pp)-ray_len(i,2)/sqrt(vps(i,2)-pp);
end
dtdp0 = x+p*dtdp0;

