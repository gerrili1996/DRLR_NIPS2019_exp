% logistic regression with infty norm regularization  
% __author__ = 'Jiajin Li'
% __email__ = 'gerrili1996@gmail.com'

function optimal = LR_ProximalNewton(A,mu,tau,test,e)

[N,d] = size(A);
x = zeros(d,1);
iter = 0;
ss =1;
alpha =0.02;

while 1
    iter = iter + 1;
    y = A*x;
    obj_old = sum(log(1+exp(-y))) + e*N*norm(x,'inf');
 	z = sigmoid(y);
	grad_f = sum(repmat(z-1,[1,d]).*A,1);
	B = z.*(1-z);
    s = 1:1:N;
	D = sparse(s,s,B);
	hessian =  A'*D*A + mu * eye(d);
    [delta,k] = QP_FISTA(hessian,grad_f',x,N*e,x);
    x = x + ss*delta;
    obj= sum(log(1+exp(-A*x))) + e*N*norm(x,'inf');
    while obj > obj_old + alpha * ss * (grad_f*delta+e*N*(norm(x+delta,'inf')-norm(x,'inf')))
        ss = 0.8 *ss;
        x = x + ss*delta;
        obj= sum(log(1+exp(-A*x))) + e*N*norm(x,'inf');
    end    
    
    if test == 1
       fprintf("Iter:%d, function value:%1.6e, iteration of subproblem:%d\n", iter,obj,k);
        %fprintf("Iter:%d, function value:%1.6e\n", iter,obj);
    end 
    max_tol = abs(obj-obj_old);
    if max_tol < tau
        break;
    end 

end 
optimal.obj = obj;
optimal.x =x;
end