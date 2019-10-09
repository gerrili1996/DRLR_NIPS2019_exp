function yp = y_subproblem(rho,d1,d2,y0)
% y_subproblem(rho,d1,d2,y0) solves the following problem
%  min_y phi(y) := \left\{ f(y) + 0.5*rho*||y - d1||^2 + min_z { g(z) +  0.5*rho*||z - (y-d2)||^2 } \right\}

% where
%   f(y) = sum(log(1+exp(-y)));
%   g(z) = sum(max(z,0));
%   d1 = Ax^{k+1} + u^k/rho;
%   d2 = b + v^k/rho

% Method: Semismooth Newton Method, starting from y0
% Notice that phi(y) is a function of y. For any y, the inner problem on z has closed-form solution.

maxiter = 5000;
tol = 1e-8;
y = y0;
f = func_val_phi(y,rho,d1,d2);

for k=1:maxiter
    g = grad_val_phi(y,rho,d1,d2);
    Hd = Hess_val_phi(y,rho,d2);
    dir = (Hd.^(-1)).*(-g);      % Semismooth Newton direction
    
    % line search
    ss = 1;
    y_new = y + ss*dir;
    f_new = func_val_phi(y_new,rho,d1,d2);
    while f_new > f + 0.2*ss*dot(g,dir)
        ss = ss*0.5;
        y_new = y + ss*dir;
        f_new = func_val_phi(y_new,rho,d1,d2);
%         if ss<1e-3
%             error('line search step size too small')
%         end
    end
%     fprintf('Semismooth Newton Iteration %d, \t step size = %1.5f, \t g = %1.5f\n', k, ss,norm(g));
    
    % stopping condition
   if norm(y_new-y)<tol
       break
   end
    % update
    y = y_new;
    f = f_new;

end

% fprintf('y-subproblem finished with # of Semismooth Newton iterations: %d \n', k);
yp = y;


end


function fval = func_val_phi(y,rho,d1,d2)
% func_val_phi(y,rho,d1,d2) evaluates the function value of phi

zb = prox_g(y-d2,rho);
fval = sum(log(1+exp(-y))) + 0.5*rho*norm(y-d1)^2 + sum(max(zb,0)) + 0.5*rho*norm(zb-y+d2)^2;

end

function gval = grad_val_phi(y,rho,d1,d2)
% grad_val_phi(y,rho,d1,d2) evaluates the gradient of phi at y

zb = prox_g(y-d2,rho);
gval = -exp(-y)./(1+exp(-y)) + rho*(y-d1) + rho*(y-d2-zb);
end

function Hval = Hess_val_phi(y,rho,d2)
% Hess_val_phi(y,rho,d1,d2) evaluates the generalized Hessian of phi at y

h = zeros(size(y));
h((y>d2)&(y<d2+1/rho)) = 1;
Hval = exp(-y)./((1+exp(-y)).^2) + rho + rho*h;
end


function zb = prox_g(vec,rho)
% prox_g(vec,rho) finds the optimal solution of the following problem
%  min_z g(z) + 0.5*rho*||z - vec||^2    (proximal operator of g)

zb = vec;
zb((zb>=0)&(zb<=1/rho)) = 0;
zb(zb>1/rho) = zb(zb>1/rho) - 1/rho;

end

