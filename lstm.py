"""
Author:Lan
"""
import time
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import myDataLoader


class lstm(nn.Module):
    def __init__(self, input_size=3, hidden_size=4, output_size=1, num_layer=2) :
        super(lstm, self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
        self.layer2 = nn.Sequential(nn.Linear(hidden_size, output_size),
                                    nn.Tanh())

    def forward(self, x):
        x, _ = self.layer1(x)
        s, b, h = x.size()
        x = x.view(s * b, h)
        x = self.layer2(x)
        x = x.view(s, b, -1)
        return x

def train():
    batch_size = 32
    learning_rate = 0.02
    train_data = myDataLoader.State(txt='train_state.txt', transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    model = lstm(8, 4, 1, 2)
    # 开始训练
    # model = net.Activation_Net(28 * 28, 300, 100, 10)
    # model = net.Batch_Net(28 * 28, 300, 100, 10)
    if torch.cuda.is_available() :
        model = model.cuda()

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # 训练模型
    for epoch in range(400) :
        start = time.time()
        running_loss = 0
        for i, data in enumerate(train_loader, 0) :
            img, label = data
            img = img.view(img.size(0), -1).float()
            label = label.float()
            if torch.cuda.is_available():
                img = img.float().cuda()
                label = label.float().cuda()
            else :
                img = Variable(img)
                label = Variable(label)
            img = img.reshape(img.shape [0], 1, img.shape [1])
            out = model(img)
            out = torch.squeeze(out)
            loss = criterion(out, label)
            # print(loss)
            if torch.isnan(loss) :
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.data
            if i % 100 == 99 :
                print('epoch: {}, loss: {:.4}'.format(epoch + 1, running_loss / 100))
                running_loss = 0
    torch.save(model, 'lstm.pkl')  # 保存整个网络

def test():
    batch_size = 1
    model = torch.load('lstm.pkl')
    model = model.eval()  # 转换成测试模式
    test_data = myDataLoader.State(txt='test_state.txt', transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)
    if torch.cuda.is_available() :
        model = model.cuda()
    criterion = nn.MSELoss()
    # 训练模型
    running_loss = 0
    for i, data in enumerate(test_loader, 0) :
        img, label = data
        img = img.view(img.size(0), -1).float()
        label = label.float()
        if torch.cuda.is_available():
            img = img.float().cuda()
            label = label.float().cuda()
        else :
            img = Variable(img)
            label = Variable(label)
        img = img.reshape(img.shape [0], 1, img.shape [1])
        out = model(img)
        out = torch.squeeze(out)
        loss = criterion(out, label)
        running_loss += loss.data
    print("total number:", len(test_data))
    running_loss = running_loss/len(test_data)
    print("average mse:", running_loss)


if __name__ == '__main__':
    # train()
    test()
