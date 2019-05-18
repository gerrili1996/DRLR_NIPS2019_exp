function zp = z_subproblem(vec,rho)
% z_subproblem(y,d2,rho) returns the optimal solution of the z-subproblem:
%  min_z g(z) + 0.5*rho*||z - vec||^2
% where
%   vec = y^{k+1} - b - v^k/rho
%
% This is equivalent with zp = prox_g(vec,rho) that is a subroutine in
% y_subproblem

% Created by Zirui Zhou.

zp = vec;
zp((zp>=0)&(zp<=1/rho)) = 0;
zp(zp>1/rho) = zp(zp>1/rho) - 1/rho;

end

