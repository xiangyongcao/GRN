%% Generate Training Dataset (Stage 2: iid Gaussian: sigma unknown and randomly selected from [30,70])
clc;clear;
rng(0);
addpath(genpath(pwd));

basedir = '/home/cxy/Documents/GlobalNN-HSI/data/ICVL_training/stage2/';
datadir = '/home/cxy/Documents/GlobalNN-HSI/data/datasets/ICVL_Train/';
namedir = '/home/cxy/Documents/GlobalNN-HSI/';
g = load(fullfile(namedir, 'train_fns.mat'));
fns = g.fns;
preprocess = @(x)(rot90(x,1));


newdir = fullfile(basedir, ['icvl_','blind']);
generate_dataset_blind(datadir, fns, newdir, 'rad', preprocess);


