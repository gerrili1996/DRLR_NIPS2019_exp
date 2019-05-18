% logistic regression -Classic Newton Method with 
% Amijo type line search strategy 
% __author__ = 'Jiajin Li'
% __email__ = 'gerrili1996@gmail.com'

function optimal = LR_newton(A,mu,tau,test)

[N,d] = size(A);
x = zeros(d,1);
iter = 0;
while 1
    iter = iter + 1;
    x_old = x;
    y = A*x;
    obj_old = sum(log(1+exp(-y)));
 	z = sigmoid(y);
	grad_f = sum(repmat(z-1,[1,d]).*A,1);
	B = z.*(1-z);
    s = 1:1:N;
	D = sparse(s,s,B);
	hessian =  A'*D*A + mu * eye(d);
    ss = 1; 
    delta = pinv(hessian)*grad_f';
%% Amijo type line search goes here !!!
    while true 
        x = x_old - ss * delta; 
        obj = sum(log(1+exp(-A*x)));
        if (obj < obj_old - 0.0001* ss * norm(grad_f))
            break;
        end 
        ss = 0.5*ss;
    end 
    if test == 1
    	fprintf("iter:%d, function value:%1.6e\n", iter,obj);
    end 
    max_tol = abs(obj-obj_old);
    if max_tol < tau
        break;
    end 

end 
optimal.obj = obj;
optimal.x = x;
end