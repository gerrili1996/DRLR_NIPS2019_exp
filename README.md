This folder contains the MATLAB source code for the implementation, and all the experiments in the paper,
"A First_Order Algorithmic Framework for Distributionally Robust  Logistic Regression" (__NeurIPS 2019__)
By Jiajin Li, Sen Huang, Anthony Man-Cho So.
- Contact: Jiajin Li 
- If you have any questions, please feel free to contact "gerrili1996 at gmail.com".  

====================================================================================================
####  Installation Guide 

1. Download YALMIP and IPOPT and add these packages in the root folder "__DRLR_NIPS_exp__" <br>
  Download link for YAMLIP: https://yalmip.github.io/download/ <br>
  Download link for IPOPT from OPTI toolbox:  https://www.inverseproblem.co.nz/OPTI/ 
2. Download the UCI adult dataset (a1a-a9a)<br>
  Download link: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html<br>
  Add them into the folder '/dataset';  
3. To finish installation, run the install.m script to add the dependencies to the working directory of MATLAB. 

#### Utilisation guide
The package comes with 4 main functions to output the experiment results in the paper: 
- Exp1_sythetic.m:  Compare our first-order algorithmic framework with the interior point algorithm (i.e., Yamlip Solver) on sythetic data; 
- Exp1_UCI_adult.m: Compare our first-order algorithmic framework with the interior point algorithm (i.e., Yamlip Solver) on UCI adult a1a- a9a dataset; 
- Exp2_sythetic_plot.m: Compare our adaptive LP_ADMM algorithm with other first-order methods on the beta- subproblem; 
- Exp3_accuracy.m: Output the test performance of DRLR model compare with vanilla LR and Regularized LR;
All the experiment result (i.e.,txt, figure) are in the folder '__/experiment_result__'.

CONTENTS of '__/src__' folder: 
- FO_DRLR: contain all function in our first-order algorithmic framwork;  
- Yamlip_DRLR: apply the Yamlip solver to solver the DRLR; 
- prox_operator: contains all proximal mapping functions;
- LR & RLR: contain the solver appling the proximal-newton method to tackle the regularized LR model;
- subproblem_baseline: contain all first-order method baselines (PDHG, Subgradient, SADMM) in the _Exp2_sythetic_plot_;



