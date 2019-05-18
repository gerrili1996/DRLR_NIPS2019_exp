%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%  logistic regression with infty norm regularization  
%  subprobelm solver by FISTA
%   min_x {0.5* (x-b)'*H*(x-b) + g'* x + lambda * ||x|_infty}
% __author__ = 'Jiajin Li'
% __email__ = 'gerrili1996@gmail.com'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [delta,iter] = QP_FISTA(H,g,b,lambda,x0)
x = x0;
x_old = x0; 
iter = 0; 
while 1
    obj_old = 0.5*(x-b)'*H*(x-b)+g'*(x-b) +lambda* norm(x,'inf');
    ss = 1;
    iter = iter +1;
    y = x + iter/(iter+3)*(x-x_old);
    x_old = x;   
    grad_f = g + H*(y-b);
    y_mid = y-ss*grad_f;
    x = prox_inf(y_mid,lambda*ss);
    ly = 0.5*(y-b)'*H*(y-b)+g'*(y-b);
    lx =0.5*(x-b)'*H*(x-b)+g'*(x-b);
    % backtracking line search goes here ! 
    while lx > ly+ grad_f'*(x-y)+(0.5/ss)*(x-y)'*(x-y) 
        ss = 0.8*ss;
        y_mid = y-ss*grad_f;
        x = prox_inf(y_mid,lambda*ss);
        lx =0.5*(x-b)'*H*(x-b)+g'*(x-b);
    end 
    obj = 0.5*(x-b)'*H*(x-b)+g'*(x-b) +lambda* norm(x,'inf');
    % fprintf("iter:%d, function value:%1.6e\n", iter,obj); 
    if abs(obj-obj_old)< 1e-4
        break;
    end 
end
delta = x-b;
end  