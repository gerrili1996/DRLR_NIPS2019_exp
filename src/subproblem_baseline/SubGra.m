%% Subgradient Method 
%  Model :  \sum\limits_{i=1}^N {log(1+exp(-a_i^Tx) + 0.5(a_i^Tx-b)} 
%           + 0.5*||Ax-b||_1 + I_{||x||_infty \leq \lambda}
% __author__ = 'Jiajin Li'
% __email__ = 'gerrili1996@gmail.com'


function output  = SubGra(A,param)
    
	% hyper parameter setup  
    d = param.d; 
    N = param.N; 
    stepsize = param.stepsize; 
	maxiter = param.maxiter;
	delta = param.delta;  
    rho = param.rho; 
	% lambda = param.lambda; 
	b = param.lambda * param.kappa; 
    AT= A'; 
	x = ones(d,1);
	iter =1;  
    obj(1) = DRO_obj (A,x,b,N);
    time(1) = 0; 
    while  1
        tic;
    	iter = iter + 1;
    	% obj_old = DRO_obj (A,x,b,N); 
        z = A*x; 
        grad_f = bsxfun(@times,sigmoid(z)-0.5,A);
        grad_smooth = sum(grad_f,1)'/N;
        % grad = repmat(-exp( -A*x)./ (1+exp( -A*x)) + 0.5,[1,d]);
    	subgrad =  (0.5/N) * AT*sign(z-b); 
    	x =  x - stepsize * (grad_smooth + subgrad);
        itertime  = toc; %- time(iter-1); 
        obj(iter) = DRO_obj (A,x,b,N);
        time(iter) = time(iter-1) + itertime;
    	% obj = DRO_obj (A,x,b,N);
    	tol = max(abs(obj(iter)-obj(iter-1)));
        fprintf("iter:%d, the objective function:%1.6e\n",iter,obj(iter));
        if mod(iter, 1) == 0
            stepsize = stepsize * rho; 
        end 
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