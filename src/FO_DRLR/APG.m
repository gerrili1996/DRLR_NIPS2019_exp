%% Accelerated projected gradient with restart
%  Model : \min 0.5||Ax - b||^2   s.t. norm(x,inf)<= lambda
% __author__ = 'Jiajin Li'
% __email__ = 'gerrili1996@gmail.com'

function [xp,k] = APG(ATb,B,L,lambda,x0) 

maxiter = 500;
tol = 1e-6;
ss = 1/L;

xo = x0;
x = x0; 
for k=1:maxiter
    beta = k/(k+3);
    y = x + beta*(x - xo);
    grad_y = (B*y-ATb);
    %% Case: Dim large 
    % grad_y = (AT*(A*y)-ATb);
    w = y - ss*grad_y;
%     w(w>=lambda) = lambda(w>=lambda);
%     w(w<=-lambda) = -lambda(w<=-lambda);
    w(w>=lambda) =lambda;
    w(w<=-lambda) =-lambda;
    xo = x;
    x = w;
    if norm(x-y)<tol
        break
    end
end
xp = x;
end