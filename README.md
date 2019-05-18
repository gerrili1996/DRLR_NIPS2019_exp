#### A First-order Algorithmic Framework for Distributionally Robust Logistic Regression
This folder contains the MATLAB source code for the implementation, and all the experiments in the paper.
- Contact: Jiajin LI (gerrili1996@gmail.com)
- If you have any questions, please feel free to contact me.  

===============================================================
####  Installation Guide 

1. Download YALMIP and IPOPT and add these packages in the root folder "__DRLR_NIPS_exp__" 
  --Download link for YAMLIP: https://yalmip.github.io/download/ 
  --Download link for IPOPT from OPTI toolbox:  https://www.inverseproblem.co.nz/OPTI/ 
2. Download the UCI adult dataset (a1a-a9a)
  --Download link: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
  --Add them into the folder '/dataset';  
3. To finish installation, run the install.m script to add the dependencies to the working directory of MATLAB. 

#### Utilisation guide
The package comes with 4 main functions to output the experiment results in the paper: 
- Exp1_sythetic.m:  Compare our first-order algorithm framework with Yamlip Solver on sythetic data; 
- Exp1_UCI_adult.m: Compare our first-order algorithm framework with Yamlip Solver on UCI adult a1a- a9a dataset; 
- Exp2_sythetic_plot.m: Compare our adaptive LP_ADMM algorithm with other first-order method on the beta- subproblem; 
- Exp3_accuracy.m: Output the test performance of DRLR model compare with  LR and Regularized LR;
All the experiment result (i.e.,txt, figure) are in the folder '__/experiment_result__'.

CONTENTS of '__/src__' folder: 
- FO_DRLR: contain all function in our first-order algorithm framwork;  
- Yamlip_DRLR: apply the Yamlip solver to solver the DRLR; 
- prox_operator: contains all proximal mapping function;
- LR & RLR: contain the solver appling the proximal-newton method to tackle the regularized LR model;
- subproblem_baseline: contain all first-order method baseline (PDHG,Subgradient, SADMM) in the _Exp2_sythetic_plot_;

