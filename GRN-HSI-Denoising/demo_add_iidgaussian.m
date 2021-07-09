%% Generate Training Dataset (Stage 1: iid Gaussian: sigma = 50)
clc;clear;
rng(0);
addpath(genpath(pwd));

% Training datasets
disp('---------------------add noise to clean train datasets--------------------')
basedir = '/home/cxy/Documents/GRN-HSI-Denoising/data/datasets/ICVL_training_pair_whole_image/stage1/';
datadir = '/home/cxy/Documents/GRN-HSI-Denoising/data/datasets/raw_clean_ICVL_train/';
namedir = '/home/cxy/Documents/GRN-HSI-Denoising/';
g = load(fullfile(namedir, 'train_fns.mat'));
fns = g.fns;
preprocess = @(x)(rot90(x,1));

for sigma = [50]
    newdir = fullfile(basedir, ['icvl_', num2str(sigma)]);
    generate_dataset(datadir, fns, newdir, sigma, 'rad', preprocess);
end


% validation datasets
disp('---------------------add noise to clean validate datasets--------------------')
basedir = '/home/cxy/Documents/GRN-HSI-Denoising/data/datasets/ICVL_validate_pair_whole_image/stage1/';
datadir = '/home/cxy/Documents/GRN-HSI-Denoising/data/datasets/raw_clean_ICVL_validate/';
namedir = '/home/cxy/Documents/GRN-HSI-Denoising/';
g = load(fullfile(namedir, 'validate_fns.mat'));
fns = g.fns;
preprocess = @(x)(rot90(x,1));

for sigma = [50]
    newdir = fullfile(basedir, ['icvl_', num2str(sigma)]);
    generate_dataset(datadir, fns, newdir, sigma, 'rad', preprocess);
end


% testing Dataset 
disp('---------------------add noise to clean testing datasets--------------------')
basedir = '/home/cxy/Documents/GRN-HSI-Denoising/data/datasets/ICVL_test_pair_whole_image/stage1/';
datadir = '/home/cxy/Documents/GRN-HSI-Denoising/data/datasets/raw_clean_ICVL_test/';
namedir = '/home/cxy/Documents/GRN-HSI-Denoising/';
g = load(fullfile(namedir, 'test_fns.mat'));
fns = g.fns;
sz = 512;   %% crop test date to 512 x 512 x 31 due to NGmeet and LLRT can't handle
preprocess = @(x)(center_crop(rot90(x), sz, sz));

for sigma = [50] % you can change the testing sigma
    newdir = fullfile(basedir, ['icvl_', num2str(sigma)]);
    generate_dataset(datadir, fns, newdir, sigma, 'rad', preprocess);
end

