%% read all the mat-filenames in a file
clear;clc;
namelist1 = dir('/home/cxy/Documents/GRN-HSI-Denoising/data/datasets/raw_clean_ICVL_train/*.mat');

len = length(namelist1);
for i = 1:len
    fns{i} = namelist1(i).name;
    fns = fns';
end
save('/home/cxy/Documents/GRN-HSI-Denoising/train_fns.mat','fns') 

clear fns;
namelist2 = dir('/home/cxy/Documents/GRN-HSI-Denoising/data/datasets/raw_clean_ICVL_test/*.mat');
len = length(namelist2);
for i = 1:len
    fns{i} = namelist2(i).name;
    fns = fns';
end
save('/home/cxy/Documents/GRN-HSI-Denoising/test_fns.mat','fns') 


clear fns;
namelist3 = dir('/home/cxy/Documents/GRN-HSI-Denoising/data/datasets/raw_clean_ICVL_validate/*.mat');
len = length(namelist3);
for i = 1:len
    fns{i} = namelist3(i).name;
    fns = fns';
end
save('/home/cxy/Documents/GRN-HSI-Denoising/validate_fns.mat','fns') 


