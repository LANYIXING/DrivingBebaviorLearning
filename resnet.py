"""
Author:Lan
"""
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
import numpy as np
import matplotlib.pyplot as plt
import myDataLoader

show = ToPILImage()


# 用于ResNet18和34的残差块，用的是2个3x3的卷积
class BasicBlock(nn.Module) :
    expansion = 1

    def __init__(self, in_planes, planes, stride=1) :
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion * planes :
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x) :
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 用于ResNet50,101和152的残差块，用的是1x1+3x3+1x1的卷积
class Bottleneck(nn.Module) :
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4

    def __init__(self, in_planes, planes, stride=1) :
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes :
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x) :
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module) :
    def __init__(self, block, num_blocks, num_classes=1) :
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks [0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks [1], stride=2)
        # self.layer3 = self._make_layer(block, 256, num_blocks [2], stride=2)
        # self.layer4 = self._make_layer(block, 512, num_blocks [3], stride=2)
        self.fc1 = nn.Sequential(
            nn.Linear(38400, 1000), nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1000, 50), nn.ReLU(inplace=True),
            nn.Linear(50, 1), nn.Tanh()
        )

    def _make_layer(self, block, planes, num_blocks, stride) :
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides :
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x) :
        insize = x.size(0)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(insize, -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out.view(-1)


def ResNet18() :
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34() :
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50() :
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101() :
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152() :
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test() :
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


def main() :
    batchsize = 3
    train_data = myDataLoader.MyDataset(txt='data/train_pic.txt', transform=transforms.ToTensor())
    train_data_load = torch.utils.data.DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=0)
    if torch.cuda.is_available() :
        net = ResNet18().double().cuda()
    else:
        net = ResNet18().double()
    print(net)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # train
    print("training begin")
    for epoch in range(300) :
        start = time.time()
        running_loss = 0
        for i, data in enumerate(train_data_load, 0) :
            # print (inputs,labels)
            image, label = data
            if torch.cuda.is_available() :
                # print('GPU2')
                image = image.double().cuda()
                label = label.double().cuda()
            else :
                image = image.double()
                label = label.double()
            image = Variable(image)
            label = Variable(label)
            # imshow(torchvision.utils.make_grid(image))
            # plt.show()
            # print (label)
            optimizer.zero_grad()
            outputs = net(image)
            # print (outputs)
            loss = criterion(outputs, label)
            if torch.isnan(loss):
                break
            # print(outputs)
            # print(label)
            # print(loss)
            torch.set_default_tensor_type(torch.FloatTensor)
            loss.backward()
            optimizer.step()
            running_loss += loss.data
            # if i % 5 == 1 :
            if i % 100 == 99 :
                end = time.time()
                print('[epoch %d,imgs %5d] loss: %.7f  time: %0.3f s' % (
                    epoch + 1, (i + 1) * batchsize, running_loss / 100, (end - start)))
                with open("train_log.txt", 'a+') as f1 :
                    line = '[epoch %d,imgs %5d] loss: %.7f  time: %0.3f s' % (
                        epoch + 1, (i + 1) * batchsize, running_loss / 100, (end - start))+'\n'
                    f1.writelines(line)
                start = time.time()
                running_loss = 0
                if epoch % 100 == 99:
                    torch.save(net,"net/epoch%d"%(epoch+1)+'resnet.pkl')
                    print('model has saved')
    print("finish training")
    torch.save(net, 'net/resnet.pkl')  # 保存整个网络

def test_net():
    batchsize = 1
    start = time.time
    test_data = myDataLoader.MyDataset(txt='data/test_pic.txt', transform=transforms.ToTensor())
    test_data_load = torch.utils.data.DataLoader(test_data, batch_size=batchsize, shuffle=True, num_workers=0)
    i = len(test_data)
    if torch.cuda.is_available() :
        # net = torch.load('epoch299resnet.pkl').cuda()
        net = torch.load('net/epoch399resnet.pkl').cuda()
    else:
        # net = torch.load('epoch299resnet.pkl')
        net = torch.load('net/epoch399resnet.pkl')
    net.eval()
    running_loss = 0
    for i, data in enumerate(test_data_load, 0):
        # print (inputs,labels)
        image, label = data
        if torch.cuda.is_available() :
            # print('GPU2')
            image = image.double().cuda()
            label = label.double().cuda()
        else :
            image = image.double()
            label = label.double()
        image = Variable(image)
        label = Variable(label)
        outputs = net(image)
        print('outputs：', outputs)
        print('label:', label)
        criterion = nn.MSELoss()
        loss = criterion(outputs, label)
        if torch.isnan(loss):
            break
        running_loss += loss.data
        # if i % 5 == 1 :
        # if i % 100 == 99 :
        #     end = time.time()
        #     print('[imgs %5d] loss: %.7f  time: %0.3f s' % (
        #         (i + 1) * batchsize, running_loss / 100, (end - start)))
        #     with open("test_log.txt", 'a+') as f1 :
        #         line = '[imgs %5d] loss: %.7f  time: %0.3f s' % (
        #             (i + 1) * batchsize, running_loss / 100, (end - start)) + '\n'
        #         f1.writelines(line)
        #     start = time.time()
        #     running_loss = 0
    running_loss = running_loss/len(test_data)
    print(running_loss)


if __name__ == '__main__' :
    main()
    # test_net()
