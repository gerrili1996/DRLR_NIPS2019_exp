function x =ACG(ATb,ATA,lambda,x)
% x_subproblem(A,b,Del,x0) solves the following problem
%    min 0.5||Ax - b||^2   s.t. norm(x,inf)<=lambda
%
% Method: Conjugate gradient method with active set
% Created by Huang Sen
tol = 1e-5;
g = ATA*x - ATb;
index_lower = (x>=lambda);
index_upper = (x<=-lambda);
index_all = index_lower|index_upper;
boundset  =  (index_all)&(-x.*g>=0);%index_lower|index_upper
r = - g;
r(boundset)=0;
gamma  = r'*r;
for k = 1:5000
    boundset_old = boundset;
    boundset  =  (index_all)&(-x.*g>=0);
    r = - g;
    r(boundset) = 0;
    if all(r==0) 
        break
    end
    gamma1 = gamma;
    gamma  = norm(r)^2;
    if k==1||(length(boundset_old)~=length(boundset))||~all(boundset==boundset_old) %~(sum(index_all)==0)%
        p = r;
    else
        beta = gamma / gamma1;
        p = r + beta*p;
    end
    q = ATA*p;
    alpha = gamma /(p'*q);
    x_k_1 = x;
    x     = x + alpha*p;
    index_lower = (x>=lambda);
    index_upper = (x<=-lambda);
    x(index_lower) = lambda;
    x(index_upper) = -lambda;
    index_all = index_lower|index_upper;
    if sum(index_all)==0  %all(x == x_old)
        g = g + alpha*q;
    else
        g = ATA*x - ATb; 
    end
    if norm(x-x_k_1)<tol
        break
    end
end
end
