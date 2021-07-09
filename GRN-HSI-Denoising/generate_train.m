clear;
close all;
addpath(genpath(pwd));

% settings
size_input = 64;
size_label = 64;
image_dim =  31;

% load dataGRN-HSI-Denoising
filename = '/home/cxy/Documents/GRN-HSI-Denoising/data/datasets/ICVL_training_pair_whole_image/stage1/icvl_50/';
namelist = dir(fullfile(filename,'/*.mat'));

% generate h5 files, each file contains 500 image patches
num_h5_files = length(namelist);
num_patches = 500;

%% initialization
data = zeros(size_input, size_input, image_dim, 1);
label = zeros(size_label, size_label, image_dim, 1);

for j = 1:num_h5_files
    
    fns = namelist(j).name;
    tmp = load(fullfile(filename,fns));
    
    im_input = tmp.input;  % input: noisy image
    im_label = tmp.gt;  % label: clean image
   
    count = 0;
    disp(['generating ',num2str(j),' h5 file, total ',num2str(num_h5_files),' files']);
    savepath = ['./h5data/ICVL_stage1/train',num2str(j),'.h5'];
    
    for i = 1 : num_patches
        
        orig_img_size = size(im_label);
        x = random('unid', orig_img_size(1) - size_input + 1);
        y = random('unid', orig_img_size(2) - size_input + 1);
        
        subim_input = im_input(x : x+size_input-1, y : y+size_input-1,:);
        subim_label = im_label(x : x+size_input-1, y : y+size_input-1,:);
        
        count=count+1;
        data(:, :, 1:image_dim, count) = flip(imrotate(subim_input,270),2);
        label(:, :, 1:image_dim, count) = flip(imrotate(subim_label,270),2);
        
    end
    
    order = randperm(count);
    data = data(:, :, 1:image_dim, order);
    label = label(:, :, 1:image_dim, order);
    
    %% writing to HDF5
    chunksz = 100;
    created_flag = false;
    totalct = 0;
    
    for batchno = 1:floor(count/chunksz)
        last_read=(batchno-1)*chunksz;
        batchdata = data(:,:,1:image_dim,last_read+1:last_read+chunksz);
        batchlabs = label(:,:,1:image_dim,last_read+1:last_read+chunksz);
        
        startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
        curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz);
        created_flag = true;
        totalct = curr_dat_sz(end);
    end
    h5disp(savepath);
end

