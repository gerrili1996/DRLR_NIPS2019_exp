%% Compare our first-order algorithm framework with Yamlip Solver 
% Real dataset from UCI: adult a1a- a9a dataset 
% __author__ = 'Jiajin Li'
% __email__ = 'gerrili1996@gmail.com'

clc                                        
clear all 
% DRLR setting 
kappa = 1; 
epsilon = 0.1; 
fid=fopen('experiment_result/CPUtime_UCI_Adult.txt','a+');
for i = 1:9  
   %% Our First-order Algorithmic Framework 
    dataset_name = ['a',num2str(i),'a'];
    path = ['dataset/',dataset_name];
    fprintf("Processing...dataset:%s\n",dataset_name);
    [y_train,x_train] = libsvmread(path);
    A = bsxfun(@times,x_train,y_train);
    [N,d] = size(A);
    Hessian = A'*A;
    AT = A';
    L = max(eig(Hessian));
    LPADMM_param.d = d;
    LPADMM_param.N = N;
    LPADMM_param.maxiter = 100000; 
    LPADMM_param.epsilon = epsilon;
    LPADMM_param.delta = 1e-5; 
    LPADMM_param.rho = 0.001; 
    LPADMM_param.gamma = 1.04; 
    LPADMM_param.kappa = kappa;
    LPADMM_param.Hessian = Hessian;
    LPADMM_param.AT = AT; 
    LPADMM_param.L = L; 
    tic;
    [lambda_opt,~] = Golden_search(A, LPADMM_param); 
    LPADMM_param.lambda = lambda_opt;
    LPADMM_param.b =lambda_opt*kappa; 
    LPADMM_output = LP_ADMM(A, LPADMM_param,0);
    objective = LPADMM_output.objective + lambda_opt * LPADMM_param.epsilon;
    Our_cputime = toc;
    
    % Yamlip Solver
    data.x = x_train';
    data.y = y_train';
    solver_param.kappa = kappa;
    solver_param.pnorm = inf;
    solver_param.kappa = kappa;
    solver_param.ell = 1;
    solver_param.epsilon = epsilon;
    solver_param.solver = 'ipopt';
    tic;
    solver_output = DRLR(data,solver_param);
    Solver_cputime = toc;
    % clear all mexfunction to make the 'ipopt' solver to reload in the matlab setting;
    Solver_obj = solver_output.objective;
    Solver_lambda = solver_output.lambda;
    clear functions;
    mean_our = mean(Our_cputime);
    var_our = sqrt(var(Our_cputime));
    mean_solver = mean(Solver_cputime);
    var_solver = sqrt(var(Solver_cputime));
    Ratio = ceil(mean_solver / mean_our);
    fprintf(fid,"(N,d,gamma,Rho) = (%d,%d,%1.2e,%1.0e), our_cputime:%1.6e(+-)%1.6e,solver_cputime:%1.6e(+-)%1.6e, Ratio: %d;\n",...
        N,d,LPADMM_param.gamma,LPADMM_param.rho,mean_our, var_our, mean_solver,var_solver, Ratio);
    
end 
fprintf("Completed\rThe results are saved as CPUtime_UCI_Adult.txt\n");
fclose(fid);
