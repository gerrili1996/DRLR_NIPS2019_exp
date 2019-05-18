%% Compare our first-order algorithm framework with Yamlip Solver 
% The Sythetic Dataset Random Generator comes from the following paper 
%  "Distributionally robust Logisitic Regression -- NIPS 2015"   
% __author__ = 'Jiajin Li'
% __email__ = 'gerrili1996@gmail.com'

clc                                        
clear all 
load param_sythetic.mat; % load the related parameters (Nt,D,Gamma,Rho)
% Nt: the sample size 
% D: the dimension of feature 
% Gamma: the initial penalty paramter 
% Rho: the Shrinking parameter to increase the penalty 
r_fre = 30; % the frequency to rerun the experiments; 
Our_cputime = zeros(r_fre,1);
fid=fopen('experiment_result/CPUtime_Sythetic.txt','a+');
for i = 1:12%1:length(D)
    clc
    d = D(i);
    N = Nt(i);
    rho = Rho(i);
    gamma = Gamma(i); %For non-adaptive ADMM, set gammato be 0;
    fprintf("Processing...dataset(N,d)=(%d,%d)\n",Nt(i),D(i));
    for k = 1:r_fre
        rng(k); % random seed for different sythetic dataset; 
        kappa = 1; 
        epsilon = 0.1; 
        beta = randn(d,1); 
        beta = beta / norm(beta) ;  % normalization 
        x = randn(d,N);
        y = double(rand(1,N)<exp(beta'*x)./(1+exp(beta'*x)));
        y = 2*y-1;
        data.x = x;
        data.y = y;
        A = bsxfun(@times,x,y)';
        
        
       %% Our First-order Algorithmic Framework for Sythetic Data 
        Hessian = A'*A;
        AT = A';
        L = max(eig(Hessian));
        LPADMM_param.d = d;
        LPADMM_param.N = N;
        LPADMM_param.maxiter = 100000; 
        LPADMM_param.epsilon = epsilon;
        LPADMM_param.delta = 1e-5; 
        LPADMM_param.rho = rho; 
        LPADMM_param.gamma = gamma; 
        LPADMM_param.kappa = kappa;
        LPADMM_param.Hessian = Hessian;
        LPADMM_param.AT = AT;        
        LPADMM_param.L = L; 
        tic;
        [lambda_opt,~] = Golden_search(A, LPADMM_param); 
        Our_cputime(k)= toc;
        LPADMM_param.lambda = lambda_opt;
        LPADMM_param.b =lambda_opt*kappa;
        LPADMM_output = LP_ADMM(A, LPADMM_param,0);
        objective = LPADMM_output.objective + lambda_opt * LPADMM_param.epsilon;
        % Yamlip Solver 
        solver_param.kappa = kappa;
        solver_param.pnorm = inf;
        solver_param.kappa = kappa;
        solver_param.ell = 1;
        solver_param.epsilon = epsilon;
        solver_param.solver = 'ipopt';
        tic;
        solver_output = DRLR(data,solver_param);
        Solver_cputime(k) = toc;
%         clear all mexfunction to make the 'ipopt' solver to reload in the
%         matlab setting;
        Solver_obj(k) = solver_output.objective;
        Solver_lambda(k) = solver_output.lambda;
        clear functions;
    end 
    mean_our = mean(Our_cputime);
    var_our = sqrt(var(Our_cputime));
    mean_solver = mean(Solver_cputime);
    var_solver = sqrt(var(Solver_cputime));
    Ratio = ceil(mean_solver / mean_our); 
% our method + yamlip 
    fprintf(fid,"(N,d,gamma,Rho) = (%d,%d,%1.2e,%1.0e), our_cputime:%1.6e(+-)%1.6e,solver_cputime:%1.6e(+-)%1.6e, Ratio: %d;\n",...
            N,d,gamma,rho,mean_our, var_our, mean_solver,var_solver, Ratio);
% only for our mehtod
%     fprintf(fid,"(N,d,gamma,Rho) = (%d,%d,%1.2e,%1.2e), our_cputime:%1.6e(+-)%1.6e;\n",...
%              N,d,gamma,rho,mean_our, var_our);
end 
fprintf("Completed\rThe results are saved as CPUtime_Sythetic.txt\n");
fclose(fid);
