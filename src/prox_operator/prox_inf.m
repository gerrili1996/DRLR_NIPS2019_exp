%% Proximal of L_1 norm 
%  Model : Argmin{||x||_infty+\frac{1}{2*\lambda}||x-v||_2^2}
% __author__ = 'Jiajin Li'
% __email__ = 'gerrili1996@gmail.com
function x = prox_inf(v,lambda)
    x = v- lambda*ProjectOntoL1Ball(v/lambda, 1);
end 