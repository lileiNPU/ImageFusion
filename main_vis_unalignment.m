clc;
clear;
close all;

img_1 = imread('.\data\img_vis\img_03_ir.png');
img_1 = img_1(135:158, 110:150);
img_1 = imresize(img_1, [224 224]);
img_2 = imread('.\data\img_vis\img_03_vis.png');
img_2 = img_2(135:158, 110:150);
img_2 = imresize(img_2, [224 224]);
img_3 = abs(img_1 - img_2);
figure(1);
imshow(img_1);
figure(2);
imshow(img_2);
figure(3);
imshow(img_3);