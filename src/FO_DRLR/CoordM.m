%% Coordinate Minimization with restart 
%  Model : \min 0.5||Ax - b||^2   s.t. norm(x,inf)<= lambda
% __author__ = 'Jiajin Li'
% __email__ = 'gerrili1996@gmail.com'

function [xp,iter] = CoordM(D,c,lambda,x0,d,idx)

% Input data: 
%            # D: the hessian matrix D = A'*A; 
%            # c: b'*A
x = x0;  
iter = 0; 
a = D * x;
while 1 
    iter = iter +1;
    xt =x;
    for k = 1:d
        i = idx(k);
        x_i = x(i); 
        G = a(i)-c(i);
        %G = A(:,i)'*w  - c(i);
        PG = (x_i==-lambda)*min(G,0)+(x_i==lambda)*max(G,0) + G ;
        if PG ~= 0 
            x_old = x_i;
            x_i = min(max(x_i-G/D(i,i),-lambda),lambda);
            % w = w + (x_i-x_old)*A(:,i);
            a = a + (x_i-x_old)*D(:,i);
            x(i) = x_i;
        end 
    end 
    %obj = 0.5*(A*x-b)'*(A*x-b);
    %fprintf("the iteration:%d, the objective value: %1.6e\n",iter,obj);
    if  norm(xt-x)<5e-4
        break;
    end 
end 
xp =x; 
end