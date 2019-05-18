%% The Yamlip solver for Distributionally Robust Logisitic Regression 
%  copy from  Soroosh Shafiee 
%  Distributionally robust Logisitic Regression 
% __author__ = 'Jiajin Li'
% __email__ = 'gerrili1996@gmail.com'

function Optimal = DRLR_subproblem(data, parameters)
    % Define Variables
    x = data.x;
    y = data.y;
    [n, N] = size(x);
    epsilon = parameters.epsilon;
    kappa = parameters.kappa;
    pnorm = parameters.pnorm;
    solver = parameters.solver;
    lambda = parameters.lambda; 
    Optimal= [];
    
    % Define Decision Variables
    % sdpvar to define YALMIP's symbolic decision variable 
    beta = sdpvar(n,1);
    %lambda = sdpvar(1,1);
    s = sdpvar(1,N);  
    
    % Declare constraints 
    if isinf(pnorm)
        constraints = [beta <= lambda, -lambda <= beta];
    elseif pnorm == 1
        s1 = sdpvar(n,1);      
        constraints = [beta <= s1, -s1 <= beta, sum(s1) <= lambda];
    else
        constraints = [norm(beta,pnorm) <= lambda];
    end
   
    constraints = [constraints, log( 1 + exp(-y .*(beta'*x)) ) <= s ];
    constraints = [constraints, log( 1 + exp(-y .*(beta'*x)) ) + y .*(beta'*x) - lambda *kappa <= s];
    
    
    %% Construct the optimization problem 
    ops = sdpsettings('solver',solver,'verbose',0,'saveduals',0,'ipopt.max_iter',2000);
    objective = 1/N*sum(s); % lambda*epsilon +
    diagnosis = solvesdp(constraints, objective, ops);
    if ~ strcmp(diagnosis.info(1:19),'Successfully solved')
        disp('error');
    end
    Optimal.beta = double(beta);
    Optimal.lambda = double(lambda);
    Optimal.objective = double(objective);
    Optimal.diagnosis = diagnosis;     
    clearvars -except Optimal 
    % clear all the variable except for the output result 
end