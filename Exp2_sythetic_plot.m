%% Plot figure : test beta-subproblem performance 
% __author__ = 'Jiajin Li'
% __email__ = 'gerrili1996@gmail.com

clc                                        
clear all 

%% Random Generator 
rng(10);
d = 100;
N = 10000; 
kappa = 1; 
epsilon =0.1; 
beta = randn(d,1);
beta = beta / norm(beta) ;  % normalization 
x = randn(d,N);
y = double(rand(1,N)<exp(beta'*x)./(1+exp(beta'*x)));
y = 2*y-1;
data.x = x;
data.y = y;
A = (repmat(y, [d,1]).* x)';

%% Real dataset from libsvm 
% you need add your libsvm package path here. 
% kappa = 1; 
% lambda = 1;
% epsilon =0.1; 
% dataset_name = 'a1a';
% path = ['dataset/',dataset_name];
% fprintf("dataset:%s\n",dataset_name);
% [y_train,x_train] = libsvmread(path);
% A = bsxfun(@times,x_train,y_train);
% [N,d] = size(A);

%% Adaptive LP-ADMM
fprintf('\n --------------  Adaptive LP-ADMM  for DRLR ------------- \n')
ALPADMM_param.d = d;
ALPADMM_param.N = N;
ALPADMM_param.Hessian = A'*A; 
ALPADMM_param.AT = A';
ALPADMM_param.L = max(eig(ALPADMM_param.Hessian));
ALPADMM_param.maxiter = 100000; 
ALPADMM_param.delta = 1e-6; 
ALPADMM_param.rho = 0.001; 
ALPADMM_param.gamma = 1.01; 
ALPADMM_param.kappa = kappa;
ALPADMM_param.lambda = 0.1; 
ALPADMM_param.epsilon = epsilon; 
ALPADMM_output = LP_ADMM(A, ALPADMM_param,1);

%% LP-ADMM
fprintf('\n -------------- LP-ADMM for DRLR -------------- \n')
LPADMM_param.d = d;
LPADMM_param.N = N;
LPADMM_param.Hessian = A'*A; 
LPADMM_param.AT = A';
LPADMM_param.L = max(eig(ALPADMM_param.Hessian));
LPADMM_param.maxiter = 100000; 
LPADMM_param.delta = 2e-6; 
LPADMM_param.rho =0.01; 
LPADMM_param.gamma = 1; 
LPADMM_param.kappa = kappa;
LPADMM_param.lambda = 0.1; 
LPADMM_output = LP_ADMM(A, LPADMM_param,1);

%% L-ADMM
fprintf('\n --------------  Linearized ADMM  for DRLR ------------- \n')
LADMM_param.d = d;
LADMM_param.N = N;
LADMM_param.Hessian = A'*A; 
LADMM_param.AT = A';
LADMM_param.L = max(eig(ALPADMM_param.Hessian));
LADMM_param.maxiter = 100000; 
LADMM_param.delta = 1e-6; 
LADMM_param.rho = 0.1; 
LADMM_param.kappa = kappa;
LADMM_param.lambda = 0.1; 
LADMM_param.epsilon = epsilon; 
LADMM_output = LADMM(A,LADMM_param);

%% Standard ADMM
fprintf('\n -------------- Standard ADMM for DRO ------------- \n')
SADMM_param.d = d;
SADMM_param.N = N;
SADMM_param.Hessian = A'*A; 
SADMM_param.AT = A';
SADMM_param.L = max(eig(ALPADMM_param.Hessian));
SADMM_param.maxiter = 100000; 
SADMM_param.kappa = kappa;
SADMM_param.lambda = 0.1; 
SADMM_param.tol = 1e-4; 
SADMM_param.rho = 10; 
SADMM_output = SADMM(A,SADMM_param);

%% Subgradient 
fprintf('\n --------------  Subgradient method for DRO ------------- \n')
SG_param.d = d; 
SG_param.N = N; 
SG_param.rho = 0.999;
SG_param.stepsize = 0.1;
SG_param.kappa = kappa;
SG_param.lambda = 0.1; 
SG_param.maxiter = 100000;
SG_param.delta = 1e-11; 
SG_output =SubGra(A, SG_param);

%% Primal-Dual Hybrid Gradient Method
fprintf('\n --------------  Primal-Dual Hybrid Gradient Method for DRO ------------- \n')
PDHG_param.d = d; 
PDHG_param.N = N; 
PDHG_param.rho = 1;
PDHG_param.Tau= 0.3;
PDHG_param.Sigma = 0.3; 
PDHG_param.kappa = kappa;
PDHG_param.lambda = 0.1; 
PDHG_param.maxiter = 3500;
PDHG_param.delta = 1e-12; 
PDHG_output = PDHG(A, PDHG_param);

%% plot figure
% co = [0.25 0.25 0.25;
%       0 0.5 0;
%       1 0 0;
%       0 0.75 0.75;
%       0.75 0 0.75;
%       0 0 1;
%       0.75 0.75 0];
%set(groot,'defaultAxesColorOrder',co)
[Lowest index] = min([SADMM_output.objective,ALPADMM_output.objective,PDHG_output.objective,...\
    SG_output.objective,LPADMM_output.objective,LADMM_output.objective]);
algorithm = ["SADMM","ALPADMM","PDHG","SG","LPADMM","LADMM"];
algori_name = algorithm(index);
plot(ALPADMM_output.time,log10(abs(ALPADMM_output.obj-Lowest)),'Color',[228,26,28]/255,'LineWidth',2) 
hold on; 
plot(LPADMM_output.time,log10(abs(LPADMM_output.obj-Lowest)),'Color',[55,126,184]/255,'LineWidth',2)
hold on;
plot(SG_output.time,log10(abs(SG_output.obj-Lowest)),'Color',[77,175,74]/255,'LineWidth',1)
hold on; 
plot(PDHG_output.time,log10(abs(PDHG_output.obj-Lowest)),'Color',[152,78,163]/255,'LineWidth',2)
hold on; 
plot(SADMM_output.time,log10(abs(SADMM_output.obj-Lowest)),'Color',[255,127,0]/255,'LineWidth',2)
hold on; 
plot(LADMM_output.time,log10(abs(LADMM_output.obj-Lowest)),'Color',[166,86,40]/255,'LineWidth',2)
hold on; 
legend('Adaptive LP-ADMM','LP-ADMM','Subgradient','PDHG','SADMM','LADMM')
st=sprintf('Number of Samples = %d, Dimensions = %d',N,d);
title(st,'FontName','Times','FontSize',12);
xlabel(sprintf('CPU time (secs.); Total iterations %d',length(ALPADMM_output.time)),...
        'FontName','Times','FontSize',12)
ylabel('Objective function log_{10}(F-F^*)','FontSize',12,'FontName','Times')
grid on
fprintf("Completed\r The best function value is %2.7f returned by %s\n",Lowest,algori_name);