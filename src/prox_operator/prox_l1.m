%% Proximal of L_1 norm 
%  Model : Argmin{||x||_1+\frac{1}{2*\lambda}||x-v||_2^2}
% __author__ = 'Jiajin Li'
% __email__ = 'gerrili1996@gmail.com'

function x = prox_l1(v, lambda)
    x = max(0, v - lambda) - max(0, -v - lambda);
end