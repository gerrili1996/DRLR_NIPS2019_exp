%% Plot figure : Test Errors for DRLR, RLR, LR models 
% __author__ = 'Huang Sen'
% __email__ = 'Huangsen1993@gmail.com'
clear
clear all 

%-------------------------------------------------------------------------------------------------------
r_fre = 30;
samples = 1000;
%Samples for training (a1a-a9a)
rate = 0.6; 
%The percentage of sample for training (MNIST)
fid=fopen('experiment_result\exp3_log.txt','a+');
data = {'a1a','a2a'};
%'a1a','a2a','a3a','a4a','a5a','a6a','a7a','a8a','a9a'
kappa = 7;
epsilon = 0.3;
for l = 1:length(data) %dataset
    for i = 1:length(epsilon)
        for j =1:length(kappa)
            for k=1:r_fre    %repeat                   
                if l<=2
                    str = ['dataset\',data{l}];
                    [raw_y_train,raw_x_train] = libsvmread(str);
                    index = randperm(length(raw_y_train),samples);
                    x_train = raw_x_train(index,:);
                    y_train = raw_y_train(index);
                    str = ['dataset\',data{l},'.t'];
                    [y_test,x_test] = libsvmread(str);
                else
                    str = ['dataset\',data{l}];
                    [raw_y_train,raw_x_train] = libsvmread(str);
                    index = randperm(length(raw_y_train),round(rate*length(raw_y_train)));
                    x_train = raw_x_train(index,:);
                    y_train = raw_y_train(index);
                    index_test = setdiff([1:length(raw_y_train)],index);
                    y_test = raw_y_train(index_test);
                    x_test = raw_x_train(index_test,:);
                end
                %parameters for RLR
                param.epsilon =0.1;
                param.test = 0;
                param.mu = 1e-5; %Samples for training(a1a-a9a)
                param.tau = 1e-5;
                %parameters for DRLR
                A = sparse(repmat(y_train,[1,size(full(x_train),2)]).*full(x_train));
                N = length(y_train);
                d = size(full(x_train),2);
                x0 = zeros(d,1);
                Hessian = A'*A;
                param.epsilon = epsilon(i);
                param.kappa = kappa(j);
                param.Hessian = Hessian;
                fprintf("The %d-th time for %s\n",k,data{l});
                acc = test_acc(A,x_train,y_train,x_test,y_test,param);
                LR_Test_acc(k) = acc.LR*100;
                RLR_Test_acc(k) = acc.RLR*100;
                DRLR_Test_acc(k) = acc.DRLR*100;
            end      
            mean_test_LR = mean(LR_Test_acc);
            var_test_LR = sqrt(var(LR_Test_acc));
            mean_test_RLR = mean(RLR_Test_acc);
            var_test_RLR = sqrt(var(RLR_Test_acc));
            mean_test_DRLR = mean(DRLR_Test_acc);
            var_test_DRLR = sqrt(var(DRLR_Test_acc));
            fprintf(fid,"(dataset,epsilon,kappa) = (%1s,%1.1e,%1.0e),LR_Test:%2.2f(+-)%2.2f,RLR_Test:%2.2f(+-)%2.2f, DRLR_Test:%2.2f(+-)%2.2f;\n",...
                data{l},epsilon(i),kappa(j),mean_test_LR, var_test_LR,mean_test_RLR,var_test_RLR,mean_test_DRLR,var_test_DRLR);        end
    end
end
fprintf("Completed\rThe results are saved as exp3_log.txt\n");
fclose(fid);
%-------------------------------------------------------------------------------------------------------


function [acc] = predict(x_test,y_test,beta)
    m = length(beta);
    k = size(x_test,2);
    if m >k
        x_test(:,k+1:m) = 0;
    else
        x_test = x_test(:,1:m);
    end
    z = x_test*beta;
    prob = sigmoid(z);
    prob(prob>0.5)=1;
    prob(prob<0.5)=-1;
    acc = sum(y_test == prob)/length(y_test);
end 


function [acc] = test_acc(A,x_train,y_train,x_test,y_test,param)
% Input: 
%        A : data matrix 
%        [x_test,y_test]: test data
%        param: 
    
    A = sparse(repmat(y_train,[1,size(full(x_train),2)]).*full(x_train));
    N = length(y_train);
    d = size(full(x_train),2);
    x0 = zeros(d,1);
    
    e = param.epsilon;
    test = param.test;
    mu = param.mu;
    tau = param.tau;
    
    fprintf("Processing...Logistic Regression\n");
    optimal_LR = LR_newton(A,mu,tau,test);
    fprintf("Processing...Regularized Logistic Regression\n");
    optimal_RLR = LR_ProximalNewton(A,mu,tau,test,e);
    fprintf("Processing...Distributionally Robust Logistic Regression\n");
    optimal_DRLR = DRLR_LP_ADMM(A,x_train,y_train,x_test,y_test,param);

    acc.LR = predict(x_test,y_test,optimal_LR.x); 
    acc.RLR = predict(x_test,y_test,optimal_RLR.x); 
    acc.DRLR = predict(x_test,y_test,optimal_DRLR.x); 
    %acc.DRLR = predict(x_test,y_test,solver_output.beta); 
end 