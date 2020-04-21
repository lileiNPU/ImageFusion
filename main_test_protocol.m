clc;
clear; 
close all;

%% Proposed method different conv layers
original_img_path = '.\data\img_vis';
img_file = dir(fullfile(original_img_path, '*.png'));
img_num = length(img_file) / 2;
fuse_img_path = '.\fuse_results';
fuse_mode = {'_ssim_1_skips_1_convs_256_filters', '_ssim_1_skips_2_convs', '_ssim_1_skips_3_convs', '_ssim_1_skips_4_convs', '_ssim_1_skips_5_convs', '_ssim_1_skips_6_convs', '_ssim_1_skips_7_convs'};
ssim_a_mode_all = [];
Nabf_a_mode_all = [];
FMI_dct_a_mode_all = [];
FMI_w_a_mode_all = [];
for i = 1 : 10%img_num
    if i < 10
        img_index = ['img_0', num2str(i)];
    else
        img_index = ['img_', num2str(i)];
    end
    img_ir = im2double(imread(fullfile(original_img_path, [img_index, '_ir.png'])));
    img_vis = im2double(imread(fullfile(original_img_path, [img_index, '_vis.png'])));
    ssim_a_mode = [];
    Nabf_a_mode = [];
    FMI_dct_a_mode = [];
    FMI_w_a_mode = [];
    for f = 1 : length(fuse_mode)
        fuse_mode_temp = fuse_mode{f};
        fuse_img = im2double(imread(fullfile(fuse_img_path, [img_index, fuse_mode_temp, '.png'])));
        ssim_ir_fuse = ssim(fuse_img, img_ir);
        ssim_vis_fuse = ssim(fuse_img, img_vis);
        ssim_a = (ssim_ir_fuse + ssim_vis_fuse) / 2;
        ssim_a_mode = [ssim_a_mode ssim_a];
        %Nabf
        Nabf = analysis_nabf(fuse_img, img_ir, img_vis);
        Nabf_a_mode = [Nabf_a_mode Nabf];
        %FMI
        FMI_dct = analysis_fmi(img_ir, img_vis, fuse_img, 'dct');
        FMI_dct_a_mode = [FMI_dct_a_mode FMI_dct];
        FMI_w = analysis_fmi(img_ir, img_vis, fuse_img, 'wavelet');
        FMI_w_a_mode = [FMI_w_a_mode FMI_w];
    end
    ssim_a_mode_all = [ssim_a_mode_all; ssim_a_mode];
    Nabf_a_mode_all = [Nabf_a_mode_all; Nabf_a_mode];
    FMI_dct_a_mode_all = [FMI_dct_a_mode_all; FMI_dct_a_mode];
    FMI_w_a_mode_all = [FMI_w_a_mode_all; FMI_w_a_mode];
end
ssim_a_mode_all_sum_layers = mean(ssim_a_mode_all, 1);
Nabf_a_mode_all_sum_layers = mean(Nabf_a_mode_all, 1);
FMI_dct_a_mode_all_sum_layers = mean(FMI_dct_a_mode_all, 1);
FMI_w_a_mode_all_sum_layers = mean(FMI_w_a_mode_all, 1);

%% Proposed method different filters
original_img_path = '.\data\img_vis';
img_file = dir(fullfile(original_img_path, '*.png'));
img_num = length(img_file) / 2;
fuse_img_path = '.\fuse_results';
fuse_mode = {'_ssim_1_skips_1_convs_8_filters', '_ssim_1_skips_1_convs_16_filters', '_ssim_1_skips_1_convs_32_filters', '_ssim_1_skips_1_convs_64_filters', '_ssim_1_skips_1_convs_128_filters', ...
             '_ssim_1_skips_1_convs_256_filters', '_ssim_1_skips_1_convs_512_filters', '_ssim_1_skips_1_convs_1024_filters'};

ssim_a_mode_all = [];
Nabf_a_mode_all = [];
FMI_dct_a_mode_all = [];
FMI_w_a_mode_all = [];
for i = 1 : 10%img_num
    if i < 10
        img_index = ['img_0', num2str(i)];
    else
        img_index = ['img_', num2str(i)];
    end
    img_ir = im2double(imread(fullfile(original_img_path, [img_index, '_ir.png'])));
    img_vis = im2double(imread(fullfile(original_img_path, [img_index, '_vis.png'])));
    ssim_a_mode = [];
    Nabf_a_mode = [];
    FMI_dct_a_mode = [];
    FMI_w_a_mode = [];
    for f = 1 : length(fuse_mode)
        fuse_mode_temp = fuse_mode{f};
        fuse_img = im2double(imread(fullfile(fuse_img_path, [img_index, fuse_mode_temp, '.png'])));
        ssim_ir_fuse = ssim(fuse_img, img_ir);
        ssim_vis_fuse = ssim(fuse_img, img_vis);
        ssim_a = (ssim_ir_fuse + ssim_vis_fuse) / 2;
        ssim_a_mode = [ssim_a_mode ssim_a];
        %Nabf
        Nabf = analysis_nabf(fuse_img, img_ir, img_vis);
        Nabf_a_mode = [Nabf_a_mode Nabf];
        %FMI
        FMI_dct = analysis_fmi(img_ir, img_vis, fuse_img, 'dct');
        FMI_dct_a_mode = [FMI_dct_a_mode FMI_dct];
        FMI_w = analysis_fmi(img_ir, img_vis, fuse_img, 'wavelet');
        FMI_w_a_mode = [FMI_w_a_mode FMI_w];
    end
    ssim_a_mode_all = [ssim_a_mode_all; ssim_a_mode];
    Nabf_a_mode_all = [Nabf_a_mode_all; Nabf_a_mode];
    FMI_dct_a_mode_all = [FMI_dct_a_mode_all; FMI_dct_a_mode];
    FMI_w_a_mode_all = [FMI_w_a_mode_all; FMI_w_a_mode];
end
ssim_a_mode_all_sum_filters = mean(ssim_a_mode_all, 1);
Nabf_a_mode_all_sum_filters = mean(Nabf_a_mode_all, 1);
FMI_dct_a_mode_all_sum_filters = mean(FMI_dct_a_mode_all, 1);
FMI_w_a_mode_all_sum_filters = mean(FMI_w_a_mode_all, 1);

%% Proposed method different loss
original_img_path = '.\data\img_vis';
img_file = dir(fullfile(original_img_path, '*.png'));
img_num = length(img_file) / 2;
fuse_img_path = '.\fuse_results';
fuse_mode = {'_ssim_1_skips_1_convs_256_filters', '_ssim_1_skips_1_convs_256_filters_mse', '_pearson_1_skips_1_convs_256_filters'};

ssim_a_mode_all = [];
Nabf_a_mode_all = [];
FMI_dct_a_mode_all = [];
FMI_w_a_mode_all = [];
for i = 1 : 10%img_num
    if i < 10
        img_index = ['img_0', num2str(i)];
    else
        img_index = ['img_', num2str(i)];
    end
    img_ir = im2double(imread(fullfile(original_img_path, [img_index, '_ir.png'])));
    img_vis = im2double(imread(fullfile(original_img_path, [img_index, '_vis.png'])));
    ssim_a_mode = [];
    Nabf_a_mode = [];
    FMI_dct_a_mode = [];
    FMI_w_a_mode = [];
    for f = 1 : length(fuse_mode)
        fuse_mode_temp = fuse_mode{f};
        fuse_img = im2double(imread(fullfile(fuse_img_path, [img_index, fuse_mode_temp, '.png'])));
        %ssim
        ssim_ir_fuse = ssim(fuse_img, img_ir);
        ssim_vis_fuse = ssim(fuse_img, img_vis);
        ssim_a = (ssim_ir_fuse + ssim_vis_fuse) / 2;
        ssim_a_mode = [ssim_a_mode ssim_a];
        %Nabf
        Nabf = analysis_nabf(fuse_img, img_ir, img_vis);
        Nabf_a_mode = [Nabf_a_mode Nabf];
        %FMI
        FMI_dct = analysis_fmi(img_ir, img_vis, fuse_img, 'dct');
        FMI_dct_a_mode = [FMI_dct_a_mode FMI_dct];
        FMI_w = analysis_fmi(img_ir, img_vis, fuse_img, 'wavelet');
        FMI_w_a_mode = [FMI_w_a_mode FMI_w];
    end
    ssim_a_mode_all = [ssim_a_mode_all; ssim_a_mode];
    Nabf_a_mode_all = [Nabf_a_mode_all; Nabf_a_mode];
    FMI_dct_a_mode_all = [FMI_dct_a_mode_all; FMI_dct_a_mode];
    FMI_w_a_mode_all = [FMI_w_a_mode_all; FMI_w_a_mode];
end

ssim_a_mode_all_sum_loss = mean(ssim_a_mode_all, 1);
Nabf_a_mode_all_sum_loss = mean(Nabf_a_mode_all, 1);
FMI_dct_a_mode_all_sum_loss = mean(FMI_dct_a_mode_all, 1);
FMI_w_a_mode_all_sum_loss = mean(FMI_w_a_mode_all, 1);

%%
Comapred_with_layer_filter_loss.ssim_a_mode_all_sum_layers = ssim_a_mode_all_sum_layers;
Comapred_with_layer_filter_loss.Nabf_a_mode_all_sum_layers = Nabf_a_mode_all_sum_layers;
Comapred_with_layer_filter_loss.FMI_dct_a_mode_all_sum_layers = FMI_dct_a_mode_all_sum_layers;
Comapred_with_layer_filter_loss.FMI_w_a_mode_all_sum_layers = FMI_w_a_mode_all_sum_layers;

Comapred_with_layer_filter_loss.ssim_a_mode_all_sum_filters = ssim_a_mode_all_sum_filters;
Comapred_with_layer_filter_loss.Nabf_a_mode_all_sum_filters = Nabf_a_mode_all_sum_filters;
Comapred_with_layer_filter_loss.FMI_dct_a_mode_all_sum_filters = FMI_dct_a_mode_all_sum_filters;
Comapred_with_layer_filter_loss.FMI_w_a_mode_all_sum_filters = FMI_w_a_mode_all_sum_filters;

Comapred_with_layer_filter_loss.ssim_a_mode_all_sum_loss = ssim_a_mode_all_sum_loss;
Comapred_with_layer_filter_loss.Nabf_a_mode_all_sum_loss = Nabf_a_mode_all_sum_loss;
Comapred_with_layer_filter_loss.FMI_dct_a_mode_all_sum_loss = FMI_dct_a_mode_all_sum_loss;
Comapred_with_layer_filter_loss.FMI_w_a_mode_all_sum_loss = FMI_w_a_mode_all_sum_loss;

save('Comapred_with_layer_filter_loss.mat', 'Comapred_with_layer_filter_loss');