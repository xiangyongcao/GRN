function [noisy_data_all, clean_data_all] = dataload(filename)
namelist = dir(fullfile(filename,'/*.mat'));
len = length(namelist);
noisy_data_all = {};
clean_data_all = {};
for i = 1:len
    fns = namelist(i).name;
    x = load(fullfile(filename,fns));
    noisy_data_all{i} = x.input;
    clean_data_all{i} = x.gt;
end

