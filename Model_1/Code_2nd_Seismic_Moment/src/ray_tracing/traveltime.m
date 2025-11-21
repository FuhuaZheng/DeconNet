function [tt0,toa0]=traveltime(depth,x,model)

global ray_len ns vv vmin vps topp bttm

if nargin<3
  vel0 = [
   5.40  0.0
   6.38  4.0
   6.59  9.0
   6.73 16.0
   6.86 20.0
   6.95 25.0
   7.80 51.0;
   8.00 81.0];
  model = [vel0(:,1),  vel0(:,2)];
end

h  = model(:,2);    % column vector of depths corresponding to the tops of layers (km)
vl = model(:,1);    % velocity within each layer (km/s)
num_lay0 = length(h)-1;
h0 = depth;   % source depth

% add a new layer boundary at the source depth without changing the model
if(length(find(h0==h))==0)
  i = max(find(h0>h));
  h = [h(1:i); h0; h(i+1:end)];
  vl = vl([1:i i:end]);
else
  i = find(h0==h)-1;
end

ns = i;
thk = diff(h);
num_lay = length(thk);
vmin = 0.001;

for i = 1:num_lay
    if(vl(i,1)<vmin)
        vl(i,1) = vmin;
    end
    vps(i,1) = 1/vl(i)^2;
    vps(i,2) = vps(i,1);
end
 
%------------direct arrival------------------------%
topp = 1;
bttm = ns;
for i = topp:bttm
    ray_len(i,1) = thk(i);
    ray_len(i,2) = 0;
end
pmax = min([99999,1./(vl(1:ns))']);
p = findp0(x,pmax);
p0 = p;
[tt dtt] = taup(p0,x);
tt0 = tt;
dtt0 = dtt;
toa0 = 180 - asin(p0*vl(ns))/pi*180;

%-------------reflected arrival from below---------%
for bttm = (ns+1):(num_lay-1)
    ray_len(bttm,1) = 2.*thk(bttm);
    ray_len(bttm,2) = 0;
    pmax = min(pmax,1/vl(bttm));
    p = findp0(x,pmax);
    [tt dtt] = taup(p,x);
    if(tt<tt0)
        tt0 = tt;
        dtt0 = dtt;
        p0 = p;
        toa0 = asin(p0*vl(ns))/pi*180;
    end
end


%------------reconstruct dtt to include all layers-%
dtt_tmp = [dtt0 zeros(1,num_lay+1-length(dtt0))];
if(num_lay==num_lay0)
    dtt0 = dtt_tmp;
else
    if(ns==1)
        dtt0 = [dtt_tmp(1)+dtt_tmp(2), dtt_tmp(3:end)];
    elseif(ns==num_lay)
        dtt0 = [dtt_tmp(1:ns-1) dtt_tmp(num_lay)+dtt_tmp(num_lay+1)];
    else
        dtt0 = [dtt_tmp(1:ns-1) dtt_tmp(ns)+dtt_tmp(ns+1) dtt_tmp(ns+2:end)];
    end
end
    

clearvars -global ray_len ns vv vmin vps topp bttm
