import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torchvision import transforms
from PIL import Image
import random
from torch.utils.data import Dataset, DataLoader
import os
import pytorch_pearson

class Data_myself(Dataset):

    def __init__(self, listroot=None, labelroot=None, shuffle=True):
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
        return img_fuse_2  # 返回

img_transforms = transforms.ToTensor()
# load image and label
batch_size = 1
epoch_num = 20
#model_save_path = "./results/model_none_norm_stn"
model_save_path = "./results/model_pearson_1_skips_1_convs_256_filters"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# set network
net = AttentionNet(img_channel=1)
# gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)
#criterion = torch.nn.MSELoss()
criterion = pytorch_pearson.PEARSON()
optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
list_path = "./data/list/train_list_stn.txt"
train_data = Data_myself(listroot=list_path)
train_loss = open(model_save_path + "/train_loss_stn.txt", "w")
for epoch in range(epoch_num):
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    b_counter = 0
    for batch_image_ir, batch_image_vis in train_loader:
        b_counter = b_counter + batch_size
        # zero the parameter gradients
        optimizer.zero_grad()  # 要把梯度重新归零，因为反向传播过程中梯度会累加上一次循环的梯度
        # forward + backward + optimize
        batch_image_ir = batch_image_ir.to(device)
        batch_image_vis = batch_image_vis.to(device)
        outputs = net(batch_image_ir, batch_image_vis)
        #outputs.type(torch.LongTensor)
        #outputs = outputs.long()
        batch_ir_vis = torch.cat((batch_image_ir, batch_image_vis), 1)
        pearson_cc = criterion(outputs, batch_ir_vis)
        loss = 1 - abs(pearson_cc)
        #if pearson_cc <= 0:
            #loss = - pearson_cc  # 计算损失值,criterion我们在第三步里面定义了
        #else:
            #loss = 1 - pearson_cc

        loss.backward()
        optimizer.step()
        # print(loss)
        running_loss = loss.item()

        print('[%d, %5d] loss: %.7f' %
                (epoch + 1, b_counter, running_loss))

        train_loss.write("epoch_" + str(epoch) + "_batch_index_" + str(b_counter) + " " +  str(running_loss) + "\n")

    torch.save(net.state_dict(), (model_save_path + "/" + "net_epoch_" + str(epoch) + ".pkl"))

train_loss.close()
print('Finished Training')