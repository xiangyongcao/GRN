%% Generate Training Dataset (Stage 3: complex noise)
clc;clear;
rng(0);
addpath(genpath(pwd));

basedir = '/home/cxy/Documents/GlobalNN-HSI/data/ICVL_training/stage3/';
datadir = '/home/cxy/Documents/GlobalNN-HSI/data/datasets/ICVL_Train/';
namedir = '/home/cxy/Documents/GlobalNN-HSI/';
g = load(fullfile(namedir, 'train_fns.mat'));
fns = g.fns;
preprocess = @(x)(rot90(x,1));

% %%% for complex noise (randomly select from case 1 to case 4)
% newdir = fullfile(basedir, ['icvl_','complex','_case1']);
sigmas = [10 30 50 70];
% generate_dataset_mixture_test_case1(datadir, fns, newdir, sigmas, 'rad', preprocess);
% 
% newdir = fullfile(basedir, ['icvl_','complex','_case2']);
% generate_dataset_mixture_test_case2(datadir, fns, newdir, sigmas, 'rad', preprocess);
% 
% newdir = fullfile(basedir, ['icvl_','complex','_case3']);
% generate_dataset_mixture_test_case3(datadir, fns, newdir, sigmas, 'rad', preprocess);

newdir = fullfile(basedir, ['icvl_','complex','_case4']);
generate_dataset_mixture_test_case4(datadir, fns, newdir, sigmas, 'rad', preprocess);

newdir = fullfile(basedir, ['icvl_','complex','_case5']);
generate_dataset_mixture_test_case5(datadir, fns, newdir, sigmas, 'rad', preprocess);
