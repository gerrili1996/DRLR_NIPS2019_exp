%% Standard two block ADMM Method
%  min f(y) + g(z) + delta(x)
%  s.t.  Ax = y
%         z = y - b
% where 
%    f(y) = sum( log(1+exp(-y)) )
%    g(z) = sum( max(z,0) )
%    delta(x) = indicator function for norm(x,inf)<=Del
% Remark: Consider the variables y and z as one block, and x as the other block
%         x-subproblem solver: FISTA with restart 
%         (y,z)-subproblem solver: semi-smooth newton method 
% __author__ = 'Jiajin Li'
% __email__ = 'gerrili1996@gmail.com'

function  optimal = SADMM(A,param)
    d = param.d;
    N = param.N;
    Hessian = param.Hessian;
    AT = param.AT; 
    L = param.L; 
    lambda = param.lambda;
    kappa = param.kappa; 
    b = param.lambda * param.kappa;
    maxiter = param.maxiter; 
    tol = param.tol; 
    rho = param.rho; 

    % maxiter = 10000; 
    %tol = 1e-2; 
    %obj = zeros(1,maxiter);

    %% initilization 
    sigma = 1;
    x = ones(d,1);
    y = A*x;
    u = zeros(N,1);  % Dual vector for Ax = y
    v = zeros(N,1);  % Dual vector for y = z
    obj(1) = DRO_obj (A,x,b,N);
    iter = 1; 
    time(1) = 0; 

    while 1
       tic;
       iter = iter+1; 
       % update rule: Accelerated Project Gradient Descent with restart 
       ATb = AT*(y-u/rho);
       % xp = x_subproblem(A, y-u/rho, lambda, x);
       [xp,~] =  APG(ATb,Hessian,L,lambda,x);
       yp = y_subproblem(rho, A*xp+u/rho, b+v/rho, y);
       zp = z_subproblem(yp-b-v/rho, rho);
       % Dual Update
       up = u + sigma*rho*(A*xp - yp);
       vp = v + sigma*rho*(zp - yp + b);
       time(iter) = toc +time(iter-1);


       a = [norm(x-xp),norm(y-yp),norm(u-up),norm(v-vp)]; 
       max_tol = max(a); 
       if max_tol<tol
           break
       end
       if iter >maxiter
           break
       end 
       x = xp;
       y = yp;
       u = up;
       v = vp;
       obj(iter+1) = DRO_obj (A,x,b,N); 
       fprintf('Iteration %d of ADMM, obj = %1.6e\n', iter, obj(iter+1));
    end
    fval_opt = DRO_obj (A,x,b,N);
    optimal.objective = fval_opt; 
    optimal. obj = obj; 
    optimal.time =time;
end






