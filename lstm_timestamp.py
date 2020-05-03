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
    def __init__(self, input_size=8, hidden_size=20, output_size=1, num_layer=3) :
        super(lstm, self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
        self.layer2 = nn.Sequential(
            nn.Linear(80, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Tanh())

    def forward(self, x):
        insize = x.size(0)
        x, _ = self.layer1(x)
        s, b, h = x.size()
        x = x.view(insize, -1)
        # x = x.view(s *  b, h)
        x = self.layer2(x)
        # x = x.view(s, b, -1)
        return x

def train():
    batch_size = 32
    learning_rate = 0.02
    train_data = myDataLoader.State_timestamp(txt='data/train_lstm_state.txt', transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    model = lstm(8, 20, 1, 3)
    print (model)
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(500) :
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
            # (batch_size, timesteps, input_dim)
            img = img.reshape(-1, 4, 8)
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
    torch.save(model, 'net/lstm_timestamp.pkl')  # 保存整个网络

def test():
    batch_size = 1
    model = torch.load('net/lstm_timestamp.pkl')
    model = model.eval()  # 转换成测试模式
    test_data = myDataLoader.State_timestamp(txt='data/test_lstm_state.txt', transform=transforms.ToTensor())
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
        else:
            img = Variable(img)
            label = Variable(label)
        img = img.reshape(-1, 4, 8)
        out = model(img)
        out = torch.squeeze(out)
        print('out:',out)
        print('label:',label)
        loss = criterion(out, label)
        running_loss += loss.data
    print("total number:", len(test_data))
    running_loss = running_loss/len(test_data)
    print("average mse:", running_loss)


if __name__ == '__main__':
    # train()
    test()
