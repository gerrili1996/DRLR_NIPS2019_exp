%% Golden Section Search for Unimodel function 
%  Model : argmin h(\lambda)
%           s.t. \lambda \in [0, 0.2785/epsilon]
% __author__ = 'Jiajin Li'
% __email__ = 'gerrili1996@gmail.com'

function [lambda_opt,k] = Golden_search(A, LPADMM_param)
	a = 0;                            % start of interval
	b = 0.2785/LPADMM_param.epsilon;   % end of interval
	delta = 1e-6;              % accuracy value
	maxiter = 500;                       % maximum number of iterations
	tau = double((sqrt(5)-1)/2);      % golden proportion coefficient, around 0.618
	k=0;                            % number of iterations

	x1 = a + (1-tau)*(b-a);            
	x2 = a + tau*(b-a);                    

	LPADMM_param.lambda = x1; 
    LPADMM_param.b = x1 * LPADMM_param.kappa;
	output = LP_ADMM(A,LPADMM_param,0);
	f_x1 = output.objective + x1 * LPADMM_param.epsilon; 

       
	LPADMM_param.lambda = x2; 
    LPADMM_param.b =x2*LPADMM_param.kappa;
	output = LP_ADMM(A,LPADMM_param,0);
	f_x2 = output.objective + x2* LPADMM_param.epsilon; 


	while ((abs(b-a)>delta) && (k<maxiter))
		k = k + 1; 
        % fprintf('iter: %d \n ', k);   
        % see the progress; 
    	if(f_x1<f_x2)
        	b = x2;
        	x2 = x1;
        	x1 = a + (1-tau)*(b-a);  
        	f_x2 = f_x1;
            LPADMM_param.lambda = x1; 
            LPADMM_param.b =x1*LPADMM_param.kappa;
            output = LP_ADMM(A,LPADMM_param,0);
            f_x1 = output.objective + x1 * LPADMM_param.epsilon; 
    	else
        	a = x1;
        	x1 = x2;
        	x2 = a + tau*(b-a);
        	f_x1=f_x2;
            LPADMM_param.lambda = x2; 
            LPADMM_param.b =x2*LPADMM_param.kappa;
            output = LP_ADMM(A,LPADMM_param,0);
            f_x2 = output.objective + x2* LPADMM_param.epsilon; 
    	end
    end 
    lambda_opt = 0.5*(x1+x2); 
end 