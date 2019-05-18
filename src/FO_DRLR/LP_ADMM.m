%% ADMM with Adaptive Penalty 
%  Model : \min_{x,y} f(y) + ||y-b||_1 + I_{||x||_infty \leq \lambda}
%           s.t. Ax-y = 0 
%  where f(y) =  \sum\limits_{i=1}^N (log(1+exp(-y_i)) + 0.5*(y_i-b))  /N
% __author__ = 'Jiajin Li'
% __email__ = 'gerrili1996@gmail.com'

function output = LP_ADMM(A,param,test) 
    d = param.d; 
    N = param.N; 
	maxiter = param.maxiter;
	delta = param.delta; % stopping tolerance 
	rho = param.rho; % initial penalty parameter 
	lambda = param.lambda; 
    gamma = param.gamma; 
	b = param.lambda * param.kappa; 
    Hessian = param.Hessian;
    AT = param.AT; 
    L = param.L; 

	% Initialization 
	x = ones(d,1);
	y = ones(N,1); 
	w = ones(N,1);
	iter = 1; 
    if test == 0
    	while 1
    		iter = iter +1;  
            % Adaptive Linearized Proximal ADMM main iterative steps
            rho = gamma*rho;
            rho_r = 1/rho;
            x_mid = y + rho_r*w; 
            ATb = AT*x_mid;
            % The quadratic programming with box constraints subproblem solver
            %[x,APG_iter]  =  APG(ATb,Hessian,L,lambda,x); 
            x =ACG(ATb,Hessian,lambda,x);
            G = A*x;
            grad_f = -exp( -y)./ (1+exp( -y)) + 0.5;
            y_mid = G-rho_r*(w + grad_f/N);
            y = prox_l1(y_mid-b, 0.5*rho_r/N)+b;
            u = G-y; % residual 
            w = w - rho * u;
            max_tol = max(abs(u));
    		if max_tol < delta
    			break;
            end 
            if iter > maxiter
                 break;
            end 
        end  
        output.beta = x; 
        output.iter = iter; 
    	output.objective = DRO_obj (A,x,b,N); 
    else 
        obj(1) = DRO_obj (A,x,b,N);
        time(1) = 0; 
        while 1
            tic; 
            iter = iter +1;  
            % Adaptive Linearized Proximal ADMM main iterative steps
            rho = gamma*rho;
            rho_r = 1/rho;
            x_mid = y + rho_r*w; 
            ATb = AT*x_mid;
            % The quadratic programming with box constraints subproblem solver
            %[x,APG_iter]  =  APG(ATb,Hessian,L,lambda,x); 
            x = ACG(ATb,Hessian,lambda,x);
            G = A*x;
            grad_f = -exp( -y)./ (1+exp( -y)) + 0.5;
            y_mid = G-rho_r*(w + grad_f/N);
            y = prox_l1(y_mid-b, 0.5*rho_r/N)+b;
            u = G-y;
            w = w - rho * u;
            max_tol = max(abs(u));
            itertime  = toc;
            obj(iter) = DRO_obj (A,x,b,N);
            fprintf('iter: %d, objective:%1.6e\n',iter,obj(iter));
            time(iter) = time(iter-1) + itertime; 
            if max_tol < delta
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
end 
function objective = DRO_obj (A,x,b,N)
	objective = sum(log(1+exp(-A*x)) + max(A*x-b,0))/N; 
end 