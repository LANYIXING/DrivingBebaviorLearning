"""
Author:Lan
功能：pytorch训练所需的dataloader文件
"""
# ***************************一些必要的包的调用********************************
import re
import torch.nn.functional as F
import torch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import string
import torch.optim as optim
import os

# ***************************初始化一些函数********************************
# torch.cuda.set_device(gpu_id)#使用GPU
learning_rate = 0.0001  # 学习率的设置

# *************************************数据集的设置****************************************************************************


# 定义读取文件的格式
def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    # 创建自己的类： MyDataset,这个类是继承的torch.utils.data.Dataset
    # **********************************  #使用__init__()初始化一些需要传入的参数及数据集的调用**********************
    def __init__(self, txt='train_pic.txt', transform=None, target_transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        # 对继承自父类的属性进行初始化
        fh = open(txt, 'r')
        imgs = []
        # 按照传入的路径和txt文本参数，以只读的方式打开这个文本
        for line in fh:  # 迭代该列表#按行循环txt文本中的内
            line = line.strip('\n')
            line = line.rstrip('\n')
            # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()
            # 用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
            imgs.append((words[0], float(words[1])))
            # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
        # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable
        data_tf = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])])
        self.imgs = imgs
        # self.transform = data_tf
        self.transform = data_tf
        self.target_transform = target_transform
        self.loader = loader
        # *************************** #使用__getitem__()对数据进行预处理并返回想要的信息**********************

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]
        img = self.loader(fn)
        # if self.transform is not None :
        #     img = self.transform(img)
        # tf = transforms.ToTensor()
        tf = transforms.Compose([
            # lambda x : Image.open(x).convert("RGB"),  # string path => image data
            # transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
            # transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])
        img = tf(img)
        label = torch.tensor(label)
        return img, label
        # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
        # **********************************  #使用__len__()初始化一些需要传入的参数及数据集的调用**********************

    def __len__(self):
        # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)

class State(Dataset):
    # 创建自己的类： MyDataset,这个类是继承的torch.utils.data.Dataset
    # **********************************  #使用__init__()初始化一些需要传入的参数及数据集的调用**********************
    def __init__(self, txt='train_state.txt', transform=None, target_transform=None, loader=None):
        super(State, self).__init__()
        # 对继承自父类的属性进行初始化
        fh = open(txt, 'r')
        self.state = []
        self.label = []
        # 按照传入的路径和txt文本参数，以只读的方式打开这个文本
        for line in fh:  # 迭代该列表#按行循环txt文本中的内
            line = line.strip('\n')
            line = line.rstrip('\n')
            # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            # words = line.split()
            words = line.split()
            # 用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
            numbers = words[:-1]
            temp = ''.join(numbers)
            temp = temp.lstrip('[').rstrip(']')
            # print(temp)
            temp1 = [float(s) for s in re.findall(r'\d+.\d+', temp)]
            self.state.append(np.array(temp1))
            self.label.append(float(words[-1]))

            # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
        # data_tf = transforms.Compose(
        #     [None,
        #      transforms.Normalize([0], [1])])
        # self.imgs = state_label
        # self.target_transform = target_transform
        # self.loader = loader
        # *************************** #使用__getitem__()对数据进行预处理并返回想要的信息**********************

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        # state = torch.utils.data.TensorDataset(self.state)
        state, label = self.state[index], self.label[index]
        return torch.tensor(state), torch.tensor(label)
        # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
        # **********************************  #使用__len__()初始化一些需要传入的参数及数据集的调用**********************

    def __len__(self):
        # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.state)

class State_Pic(Dataset):
    # 创建自己的类： MyDataset,这个类是继承的torch.utils.data.Dataset
    # **********************************  #使用__init__()初始化一些需要传入的参数及数据集的调用**********************
    def __init__(self, txt='data/train_pic_state.txt', transform=None, target_transform=None, loader=default_loader):
        super(State_Pic, self).__init__()
        # 对继承自父类的属性进行初始化
        fh = open(txt, 'r')
        self.imgs = []
        self.state = []
        self.label = []
        # 按照传入的路径和txt文本参数，以只读的方式打开这个文本
        for line in fh:  # 迭代该列表#按行循环txt文本中的内
            line = line.strip('\n')
            line = line.rstrip('\n')
            # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()
            # 用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
            self.imgs.append(words[0])
            state_label = words[1:]
            numbers = state_label[:-1]
            temp = ''.join(numbers)
            temp = temp.lstrip('[').rstrip(']')
            # print(temp)
            temp1 = [float(s) for s in re.findall(r'\d+.\d+', temp)]
            self.state.append(np.array(temp1))
            self.label.append(float(words[-1]))

            # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
        # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable
        self.target_transform = target_transform
        self.loader = loader
        # *************************** #使用__getitem__()对数据进行预处理并返回想要的信息**********************

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn = self.imgs[index]
        img = self.loader(fn)
        tf = transforms.ToTensor()
        # tf = transforms.Compose([
        #     # lambda x : Image.open(x).convert("RGB"),  # string path => image data
        #     # transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
        #     transforms.RandomRotation(15),
        #     transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        #     # transforms.CenterCrop(self.resize),
        #     transforms.ToTensor(),
        #     # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #     #                      std=[0.229, 0.224, 0.225])
        # ])
        img = tf(img)
        state, label = self.state[index], self.label[index]
        return img, torch.tensor(state), torch.tensor(label)

    def __len__(self):
        # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.state)

class State_timestamp(Dataset):
    # 创建自己的类： MyDataset,这个类是继承的torch.utils.data.Dataset
    # **********************************  #使用__init__()初始化一些需要传入的参数及数据集的调用**********************
    def __init__(self, txt='data/train_lstm_state.txt', transform=None, target_transform=None, loader=None):
        super(State_timestamp, self).__init__()
        # 对继承自父类的属性进行初始化
        fh = open(txt, 'r')
        self.state = []
        self.label = []
        # 按照传入的路径和txt文本参数，以只读的方式打开这个文本
        for line in fh:  # 迭代该列表#按行循环txt文本中的内
            line = line.strip('\n')
            line = line.rstrip('\n')
            # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            # words = line.split()
            words = line.split()
            # 用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
            self.label.append(float(words [-1]))
            numbers = words[:-1]
            temp = ''.join(numbers)
            temp = temp.lstrip('[').rstrip(']')
            # print(temp)
            temp1 = [float(s) for s in re.findall(r'\d+.\d+', temp)]
            self.state.append(np.array(temp1))


            # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
        # data_tf = transforms.Compose(
        #     [None,
        #      transforms.Normalize([0], [1])])
        # self.imgs = state_label
        # self.target_transform = target_transform
        # self.loader = loader
        # *************************** #使用__getitem__()对数据进行预处理并返回想要的信息**********************

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        # state = torch.utils.data.TensorDataset(self.state)
        state, label = self.state[index], self.label[index]
        return torch.tensor(state), torch.tensor(label)
        # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
        # **********************************  #使用__len__()初始化一些需要传入的参数及数据集的调用**********************

    def __len__(self):
        # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.label)

if __name__ == '__main__':

    # train_data = MyDataset(txt='data/train_pic.txt', transform=transforms.ToTensor())
    # state_data = State(txt='data/train_state.txt', transform=transforms.ToTensor())
    # state_data = State_Pic(txt='data/train_pic_state.txt', transform=transforms.ToTensor())
    state_data = State_timestamp(txt='data/train_lstm_state.txt', transform=transforms.ToTensor())
    print('f')
