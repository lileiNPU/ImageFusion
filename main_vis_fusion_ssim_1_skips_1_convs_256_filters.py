import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torchvision import transforms
from PIL import Image
import random
from torch.utils.data import Dataset, DataLoader
import os
import pytorch_ssim
import cv2
from matplotlib import pyplot as plt
import scipy

class Data_myself(Dataset):

    def __init__(self, listroot=None, labelroot=None, shuffle=False):
        self.listroot = listroot
        self.labelroot = labelroot
        self.transform = transforms.ToTensor()
        listfile_root = self.listroot#os.path.join(self.listroot, 'train_img_label.txt')

        with open(listfile_root, 'r') as file:
            self.lines = file.readlines()
        if shuffle:
            random.shuffle(self.lines)
        # self.nSamples = len(self.lines[:30]) if debug else len(self.lines)
        self.nSamples = len(self.lines)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        assert index <= len(self), 'index range error'
        imgpath_labelpath = self.lines[index].rstrip()
        img, label = self.load_data_label(imgpath_labelpath)
        return (img, label)

    def load_data_label(self, imgpath):
        img_ir_path = imgpath.split(" ")[0]
        img_ir = Image.open(img_ir_path)#.convert('RGB')
        img_ir = self.transform(img_ir).float()
        img_vis_path = imgpath.split(" ")[1]
        img_vis = Image.open(img_vis_path)#.convert('RGB')
        img_vis = self.transform(img_vis).float()
        return img_ir, img_vis


class AttentionNet(nn.Module):
    def __init__(self, img_channel):
        super(AttentionNet, self).__init__()
        self.conv_1_1_ir = nn.Conv2d(img_channel, 256, 3, 1, 1)
        self.conv_1_1_vis = nn.Conv2d(img_channel, 256, 3, 1, 1)
        self.conv_1_1_ir_vis = nn.Conv2d(256 * 2, 1, 3, 1, 1)


    def forward(self, img_ir, img_vis):
        # ir block
        # Conv layer - 1
        img_ir_conv_1_1 = nn.functional.relu(self.conv_1_1_ir(img_ir))

        # vis block
        # Conv layer - 1
        img_vis_conv_1_1 = nn.functional.relu(self.conv_1_1_vis(img_vis))

        # concat
        img_ir_vis = torch.cat((img_ir_conv_1_1, img_vis_conv_1_1), 1)
        img_fuse = nn.functional.relu(self.conv_1_1_ir_vis(img_ir_vis))
        img_fuse_2 = torch.cat((img_fuse, img_fuse), 1)
        return img_fuse  # 返回

#individual_id = ["img_01", "img_02", "img_03", "img_04", "img_05", "img_06", "img_07", "img_08", "img_09", "img_10",
                 #"img_11", "img_12", "img_13", "img_14", "img_15", "img_16", "img_17", "img_18", "img_19", "img_20", "img_21"]

individual_id = ["img_01", "img_02", "img_03", "img_04", "img_05", "img_06", "img_07", "img_08", "img_09", "img_10"]

img_transforms = transforms.ToTensor()
# load image and label
batch_size = 1
epoch_num = 20

model_save_path = "./results/model_ssim_1_skips_1_convs_256_filters"
fuse_image_save_path = './fuse_results'
if not os.path.exists(fuse_image_save_path):
    os.makedirs(fuse_image_save_path)

# set network and load parameters
net = AttentionNet(img_channel=1)
net.load_state_dict(torch.load(model_save_path + "/" + "net_epoch_" + str(epoch_num - 1) + ".pkl"))
# gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)

list_path = "./data/list/vis_list.txt"
train_data = Data_myself(listroot=list_path)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
b_counter = 0
for batch_image_ir, batch_image_vis in train_loader:
    b_counter = b_counter + batch_size
    # zero the parameter gradients
    # forward
    batch_image_ir = batch_image_ir.to(device)
    batch_image_vis = batch_image_vis.to(device)
    outputs = net(batch_image_ir, batch_image_vis)
    img_fusion = outputs.cpu().detach().numpy()
    img_fusion = img_fusion.squeeze()

    #plt.imshow(img_fusion, cmap=plt.get_cmap('hot'))
    #plt.colorbar()
    #plt.show()

    cv2.imshow("BLUE", img_fusion)
    cv2.waitKey(500)
    img_fusion = 255 * img_fusion
    if b_counter < 10:
        cv2.imwrite(fuse_image_save_path + '/img_0'+ str(b_counter) + '_ssim_1_skips_1_convs_256_filters.png', img_fusion)
    else:
        cv2.imwrite(fuse_image_save_path + '/img_' + str(b_counter) + '_ssim_1_skips_1_convs_256_filters.png', img_fusion)

print('Finished Visualization')