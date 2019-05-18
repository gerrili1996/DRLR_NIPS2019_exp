%%  Primal-Dual hybrid gradient (PDHG) method
%  Model :  min_{||x||_\infty <= lambda} \max_{||y||_\infty <= 1}
%           \sum\limits_{i=1}^N {log(1+exp(-a_i^Tx) + 0.5(a_i^Tx-b)} + 0.5*y'(Ax-b) 
% __author__ = 'Jiajin Li'
% __email__ = 'gerrili1996@gmail.com'


function output  = PDHG(A,param)
    
	% hyper parameter setup  
    d = param.d; 
    N = param.N; 
    Tau = param.Tau; % the stepsize for primal update
    Sigma = param.Sigma; % the stepsize for dual update 
    maxiter = param.maxiter;
	delta = param.delta;  % stopping criterion 
	lambda = param.lambda; 
	b = param.lambda * param.kappa; 
	x = ones(d,1);
    y = ones(N,1); 
    AT = A'; 
	iter =1;  
    obj(1) = DRO_obj (A,x,b,N);
    time(1) = 0; 
    
    while  1
        tic;
    	iter = iter + 1;
        

        % the primal update step 
        x_old =x;
        z = A*x;
        grad_f = bsxfun(@times,sigmoid(z)-0.5,A);
        grad_x = (sum(grad_f,1)' + 0.5*AT*y)/N;
        x = x - Tau * grad_x; 
        x(x>=lambda) =lambda;
        x(x<=-lambda) =-lambda;
        theta = 0.5; 
        x_hat = x + theta*(x-x_old); 
       % the dual update step 
        grad_y = A*x_hat-b;
        y  = y + Sigma * grad_y; 
        y(y>=1) =1;
        y(y<=-1) =-1;
  
        
        itertime  = toc; 
        obj(iter) = DRO_obj (A,x,b,N);
        time(iter) = time(iter-1) + itertime;
    	tol = (abs(obj(iter)-obj(iter-1)))./obj(iter-1);
        fprintf("iter:%d, the objective function:%1.6e\n",iter,obj(iter));
    
    	if tol < delta 
    		break;
    	end 
    	if iter > maxiter 
    		break;
    	end  
    end
    output.beta = x; 
    output.time = time; 
    output.obj = obj; 
    output.iter = iter; 
    output.objective = DRO_obj (A,x,b,N); 
end 