%% Created by Huang Sen
%Experiment for RLR and LR
% license: Huangsen1993@gmail.com 
function [optimal] = DRLR_LP_ADMM(A,trainx,trainy,testx,testy,param)
   
    [N d] = size(trainx);
    beta = randn(d,1);
    beta = beta / norm(beta) ;  % normalization
    %% LP_ADMM for Sythetic Data
    LPADMM_param.maxiter = 10000;
    LPADMM_param.epsilon = param.epsilon;
    LPADMM_param.delta = 1e-5;
    LPADMM_param.rho = 0.001;
    LPADMM_param.gamma = 1.04;
    LPADMM_param.kappa = param.kappa;
    LPADMM_param.d = d;
    LPADMM_param.N = N;
    LPADMM_param.Hessian = param.Hessian;
    LPADMM_param.AT =A';
    LPADMM_param.L =  max(eig(LPADMM_param.Hessian));
    tic;
    [lambda_opt,~] = Golden_search(A, LPADMM_param);
    LPADMM_param.lambda = lambda_opt;
    LPADMM_param.b =lambda_opt*param.kappa;
    LPADMM_output = LP_ADMM(A, LPADMM_param,0);
    obj = LPADMM_output.objective + lambda_opt * LPADMM_param.epsilon;
    %LPADMM_output.objective + epsilon * norm(LPADMM_output.beta,'inf')
    cpu_time.LPADMM = toc;
    optimal.x = LPADMM_output.beta;
    optimal.obj = obj;

end
