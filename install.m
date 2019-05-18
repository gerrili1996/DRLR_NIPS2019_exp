%% Install the dependencies packages 
% Install the related package and configure the necessary path
% __author__ = 'Huang Sen'
% __email__ = ' Huangsen1993@gmail.com '


clc
clear all
mkdir experiment_result 
addpath(genpath('src'))
addpath(genpath('experiment_result'))
addpath(genpath('dataset'))
path1 = pwd;
path_src = strcat(path1,filesep,'src');
path_data = strcat(path1,filesep,'dataset');
path(genpath(path_src),path); 
path(genpath(path_data),path);
clear path1 path_src path_data
path1 = pwd;
path_src = strcat(path1,filesep,'src');
path_yamlip = strcat(path1,filesep,'YALMIP-master');
path_OPTI = strcat(path1,filesep,'OPTI-master');
addpath([path_yamlip,'\extras']);
addpath([path_yamlip,'\solvers']);
addpath([path_yamlip,'\modules']);
addpath([path_yamlip,'\modules\parametric']);
addpath([path_yamlip,'\modules\moment']);
addpath([path_yamlip,'\modules\global']);
addpath([path_yamlip,'\modules\sos']);
addpath([path_yamlip,'\operators']);
path(genpath(path_src),path); 
path(genpath(path_yamlip),path); 
path(genpath([path_yamlip,'\extras']),path); 
oldFolder = cd(path_OPTI);
opti_Install
cd(oldFolder)
clear path1 path_src path_yamlip 



