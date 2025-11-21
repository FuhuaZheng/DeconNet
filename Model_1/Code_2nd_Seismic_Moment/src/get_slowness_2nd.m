function [toa,az,vs] = get_slowness_2nd(evdp,evla,evlo,stla,stlo,vp_mod,vs_mod,top_mod,phas)
    % phas: 0-P,1-S
    % vs: The velocity of the corresponding phase

    R = 6371;
    nsta = length(phas);
    toa = zeros(nsta,1);
    az = zeros(nsta,1);
    vs = zeros(nsta,1);
    
    [layer,~] = which_layer(evdp,top_mod);
    vp_s = vp_mod(layer);
    vs_s = vs_mod(layer);
    for i = 1:nsta
        [delta,az(i)] = distance(evla,evlo,stla(i),stlo(i),R);
        if(phas(i) == 0)
            [~,toa(i)]=traveltime(evdp,delta,[vp_mod,top_mod]);
            vs(i) = vp_s;
        else
            [~,toa(i)]=traveltime(evdp,delta,[vs_mod,top_mod]);
            vs(i) = vs_s;
        end
    end
end
function [layer,h]= which_layer(evdp,top)
    % this function is to calculate which layer the hypocenter locate
    if(length(top)==1)
        layer = 1;
        h = evdp;
        return;
    end
    top = reshape(top,[],1);
    top = [top;6371];
    for i = 2:length(top)
        if(top(i)>=evdp)
            layer = i - 1;
            break;
        end
    end
    h = evdp - top(layer);
end