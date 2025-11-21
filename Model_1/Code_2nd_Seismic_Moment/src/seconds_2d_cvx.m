function [xopt,L_c,W_c,vx,vy,tauc]=seconds_2d_cvx(Ain,bin);
% implement the basic second moments inversion in cvx toolbox
A=Ain*10; b=bin*10;  %sometimes need this for precision;  cvx_precision doesn't do it.

cvx_begin
 cvx_precision best
 n=6;
 variables tt xt yt xx xy yy
 x=[tt, xt, yt, xx,xy,yy]'
 Xposvolume=[tt,xt,yt; xt, xx, xy; yt,xy,yy];
 minimize( norm(A*x-b,2) )
 subject to
   Xposvolume==semidefinite(3);
cvx_end

%make output
xopt=x;
X=[xopt(4), xopt(5);
   xopt(5),xopt(6);];
[U,S,V]=svd(X);
L_c=2*sqrt(S(1,1));
W_c=2*sqrt(S(2,2));
vx=xopt(2)/xopt(1);
vy=xopt(3)/xopt(1);
tauc=2*sqrt(xopt(1)); 
