function [tt,astf] = astf_forward(nt,Fs,dx,type_slip,...
                                Lc,Wc,drp,vr,hyp,...
                                stk,dip,toa,az,vs)
                                                        
    dt = 1.0/Fs;
    tt = [0:nt-1]'*dt-0.2*nt*dt;
    dy = dx;
    r2d = 180.0/pi;
    rtoa = toa/r2d;
    raz = az/r2d;
    rstk = stk/r2d;
    rdip = dip/r2d;
    rdrp = drp/r2d;
    nstk = round(2.0*Lc/dx);
    ndip = round(2.0*Wc/dy);
    nsta = length(az);

%% initial
    astf = zeros(nt,nsta);
    xstk = zeros(nstk,1);
    ydip = zeros(ndip,1);
    slip = zeros(ndip,nstk);
    srx = zeros(ndip,nstk);
    sry = zeros(ndip,nstk);
    t_shift = -1.0*ones(ndip,nstk);

%% 
    for i = 1:nstk
        xstk(i) = -Lc + dx/2.0 + (i-1)*dx;
    end
    for i = 1:ndip
        ydip(i) = -Wc + dy/2.0 + (i-1)*dy;
    end
    xhyp = Lc*hyp;
    yhyp = 0.0;
    
    for j = 1:nstk
       for i = 1:ndip
          ro = xstk(j)^2/Lc^2 +ydip(i)^2/Wc^2;
          if( ro >1.0)
              continue
          end
          ra2 = (xstk(j)-xhyp)^2+(ydip(i)-yhyp)^2;
          ra = sqrt(ra2);
          if(type_slip==1)
             slip(i,j) = 1.0; 
          else
             slip(i,j) = sqrt(1.0-ro);
          end
          t_shift(i,j) = ra/vr;
          srx(i,j) = (xstk(j)-xhyp)/ra/vr;
          sry(i,j) = (ydip(i)-yhyp)/ra/vr;
       end
    end

%%
    for k = 1:nsta
       sn = sin(rtoa(k))*cos(raz(k))/vs(k);
       se = sin(rtoa(k))*sin(raz(k))/vs(k);
       sd = cos(rtoa(k))/vs(k);
       [sxtmp,sytmp,~] = ned2xyz(sn,se,sd,rstk,rdip);
       [sx,sy] = rotate_drp(sxtmp,sytmp,rdrp);
       
       for j = 1:nstk
          for i=1:ndip
              if(slip(i,j)<0.0)
                 continue; 
              end
              sxtmp = sx - srx(i,j);
              sytmp = sy - sry(i,j);
              T1 = abs(sxtmp*dx);
              T2 = abs(sytmp*dy);
              tmp_shift = sx*(xstk(j)-xhyp)+sy*(ydip(i)-yhyp);
              delay = t_shift(i,j) - tmp_shift - (T1+T2)/2.0;
              stf = trap(dt,nt,T1,T2,delay);
              astf(:,k) = astf(:,k) + slip(i,j)*stf*dx*dy;
          end
       end
       astf(:,k) = astf(:,k)/(sum(astf(2:nt-1,k))+0.5*(astf(1,k)+astf(nt,k)))/dt;
    end


end

function y = trap(dt,nt,T1,T2,shift)
    y = zeros(nt,1);
    t_shift = dt*nt*0.2 + shift;
    if(T2<T1)
       T = T1;
       T1 = T2;
       T2 = T;
    end
    T = T1 + T2;
    
    if(T2 < 0.0001*dt)
       return; 
    elseif(T1<0.0001*T2)
        for i = 1:nt
           tmp = (i-1)*dt - t_shift;
           if(tmp < 0)
               continue;
           elseif(tmp<=T2)
               y(i) = 1.0;
           else
               break;
           end
        end
        y = y/T2;
        return;
    else
        for i = 1:nt
           tmp = (i-1)*dt - t_shift;
           if(tmp<0)
               continue;
           elseif(tmp<=T1)
               y(i) = tmp;
           elseif(tmp<=T2)
               y(i) = T1;
           elseif(tmp<=T)
               y(i) = (T-tmp);
           else
               break;
           end
        end
        y = y/(T1*T2);
    end
    return;
end

function [x,y,z] = ned2xyz(n,e,d,stk,dip)
    x = cos(stk).*n + sin(stk).*e;
    y = -cos(dip).*sin(stk).*n + cos(dip).*cos(stk).*e + sin(dip).*d;
    z = sin(dip).*sin(stk).*n - sin(dip).*cos(stk).*e + cos(dip).*d;
    return;
end

function [xnew,ynew] = rotate_drp(x,y,drp)
    xnew = x.*cos(drp) + y.*sin(drp);
    ynew = -x.*sin(drp) + y.*cos(drp);
    return;
end