import numpy as np
import os
from PIL import Image
import scipy.io as io

def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')]

img_path = './data/img_patches'
img_files = get_imlist(img_path)
imgCount = len(img_files)

list_path = './data/list/'
if not os.path.exists(list_path):
    os.makedirs(list_path)

individual_id = ["img_01", "img_02", "img_03", "img_04", "img_05", "img_06", "img_07", "img_08", "img_09", "img_10",
                 "img_11", "img_12", "img_13", "img_14", "img_15", "img_16", "img_17", "img_18", "img_19", "img_20", "img_21"]

id_test = individual_id[0 : 10]
id_train = individual_id[10 : ]

train_txt_file = open(list_path + "train_list_stn.txt", "w")
test_txt_file = open(list_path + "test_list_stn.txt", "w")

img_index = 0
for img_file in img_files:
    img_index += 1
    print("Prepare data do: " +  " | " + str(imgCount) + " | " + str(img_index))

    # 获取图像路径与标签
    img_file = img_file.replace("\\", "/")
    img_file_temp = img_file
    while img_file_temp.find("/") >= 0:
        imdex_temp = img_file_temp.find("/")
        img_file_temp = list(img_file_temp)
        img_file_temp[imdex_temp] = "_"
        img_file_temp = "".join(img_file_temp)

    img_name = img_file_temp[imdex_temp + 1:]


    img_type = img_name[6 : 10]
    if img_type == "_ir_":
        img_id = img_name[0: 6]
        img_ir_file = img_file
        img_vis_file = img_file.replace("_ir_", "_vis_")
        #train_txt_file.write(img_ir_file + " " + img_vis_file + "\n")

        if img_id in id_train:
            train_txt_file.write(img_ir_file + " " + img_vis_file + "\n")
        elif img_id in id_test:
            test_txt_file.write(img_ir_file + " " + img_vis_file + "\n")

train_txt_file.close()
test_txt_file.close()