function [tt dtt] = taup(p,x)

global ns vps ray_len topp bttm num_lay

pp = p*p;
tt = 0;
for i=topp:bttm
    dtt(i) = vps(i,1)*ray_len(i,1)/sqrt(vps(i,1)-pp)+vps(i,2)*ray_len(i,2)/sqrt(vps(i,2)-pp);
    tt = dtt(i)+tt;
end

