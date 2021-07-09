function [ ] = generate_dataset_mixture_test_case4( datadir, fns, newdir, sigmas, gt_key, preprocess )

ratios = [0.1, 0.3, 0.5, 0.7];
min_amount = 0.05;
max_amount = 0.15;

if ~exist(newdir, 'dir')
    mkdir(newdir)
end

for k = 1:length(fns)
    fn = fns{k};
    fprintf('generate data(%d/%d)\n', k, length(fns));
    filepath = fullfile(datadir, fn);
    mat = load(filepath); % contain gt_key
    gt = getfield(mat, gt_key);
    
    if exist('preprocess', 'var')
        gt = preprocess(gt);
    end
    
%     gt = normalized(gt);  % CAVE: comment  ICVL: no comment
%     gt = rot90(gt,1);  % CAVE: comment  ICVL: no comment
    
    Ind = 4;
    % 1: case1 (non-iid Gaussian);
    % 2: case2 (Gaussian + Stripe);
    % 3: case3 (Gaussian + Deadline);
    % 4: case4 (Gaussian + Impulse);
    [H,N,B] = size(gt);
    switch Ind
        case 1
            % add non-iid noise
            idx = randi(length(sigmas), size(gt,3), 1);
            sigma = sigmas(idx);
            %             s = reshape(sigma, 1, 1, length(sigma));
            s = zeros(H,N,B);
            for i = 1:length(sigma)
                s(:,:,i) = repmat(sigma(i),H,N);
            end
            input = gt + s/255 .* randn(size(gt));
            noise_type = 'non-iid Gaussian';
        case 2
            % add non-iid noise
            idx = randi(length(sigmas), size(gt,3), 1);
            sigma = sigmas(idx);
            %             s = reshape(sigma, 1, 1, length(sigma));
            s = zeros(H,N,B);
            for i = 1:length(sigma)
                s(:,:,i) = repmat(sigma(i),H,N);
            end
            input = gt + s/255 .* randn(size(gt));
            
            % add stripe
            all_band = randperm(B);
            b = floor(B/3);
            band = all_band(1:b);
            stripnum = randi([ceil(min_amount * N), ceil(max_amount * N)], length(band), 1);
            fprintf('Stripes:\n');
            disp(stripnum);
            for i=1:length(band)
                loc = randperm(N);
                loc = loc(1:stripnum(i));
                stripe = rand(1,length(loc))*0.5-0.25;
                input(:,loc,band(i)) = input(:,loc,band(i)) - stripe;
            end
            noise_type = 'non-iid Gaussian+Stripe';
        case 3
            % add non-iid noise
            idx = randi(length(sigmas), size(gt,3), 1);
            sigma = sigmas(idx);
            %             s = reshape(sigma, 1, 1, length(sigma));
            s = zeros(H,N,B);
            for i = 1:length(sigma)
                s(:,:,i) = repmat(sigma(i),H,N);
            end
            input = gt + s/255 .* randn(size(gt));
            % add deadline
            all_band = randperm(B);
            b = floor(B/3);
            band = all_band(b+1:2*b);
            deadlinenum = randi([ceil(min_amount * N), ceil(max_amount * N)], length(band), 1);
            fprintf('Deadline:\n');
            disp(deadlinenum);
            for i=1:length(band)
                loc = randperm(N);
                loc = loc(1:deadlinenum(i));
                input(:,loc,band(i)) = 0;
            end
            noise_type = 'non-iid Gaussian+Deadline';
        case 4
            % add non-iid noise
            idx = randi(length(sigmas), size(gt,3), 1);
            sigma = sigmas(idx);
            %             s = reshape(sigma, 1, 1, length(sigma));
            s = zeros(H,N,B);
            for i = 1:length(sigma)
                s(:,:,i) = repmat(sigma(i),H,N);
            end
            input = gt + s/255 .* randn(size(gt));
            % add impulse
            fprintf('impulse:\n');
            all_band = randperm(B);
            b = floor(B/3);
            band = all_band(2*b+1:3*b);
            idx = randi(length(ratios), length(band), 1);
            ratio = ratios(idx);
            disp(ratio);
            for i=1:length(band)
                input(:,:,band(i)) = imnoise(input(:,:,band(i)),'salt & pepper',ratio(i));
            end
            noise_type = 'non-iid Gaussian+impulse';
    end
    save(fullfile(newdir, fn), 'gt', 'input', 'sigma','noise_type');
end

