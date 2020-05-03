"""
Author:Lan
"""
import time
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import myDataLoader


class Batch_Net(nn.Module) :
    """
    """
    def __init__(self, in_dim, n_hidden_1, n_hidden_2) :
        super(Batch_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, 1), nn.Tanh())

    def forward(self, x) :
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


def train():
    batch_size = 16
    learning_rate = 0.02
    train_data = myDataLoader.State(txt='data/train_state.txt', transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    # 选择模型
    model = Batch_Net(8, 300, 100)
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(400):
        running_loss = 0
        for i, data in enumerate(train_loader, 0):
            img, label = data
            img = img.view(img.size(0), -1).float()
            label = label.float()
            if torch.cuda.is_available():
                img = img.float().cuda()
                label = label.float().cuda()
            else:
                img = Variable(img)
                label = Variable(label)
            out = model(img)
            out = torch.squeeze(out)
            loss = criterion(out, label)
            # print(loss)
            if torch.isnan(loss):
                break
            print_loss = loss.data.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.data
            if i % 100 == 99:
                print('epoch: {}, loss: {:.4}'.format(epoch+1, running_loss/100))
                running_loss = 0
    torch.save(model, 'net/mlp.pkl')  # 保存整个网络

def test():
    batchsize = 1
    criterion = nn.MSELoss()
    test_data = myDataLoader.State(txt='data/test_state.txt', transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batchsize, shuffle=True, num_workers=0)
    if torch.cuda.is_available() :
        net = torch.load('net/mlp.pkl').cuda()
    else:
        net = torch.load('net/mlp.pkl')
    net.eval()
    eval_loss = 0
    for data in test_loader:
        img, label = data
        img = img.view(img.size(0), -1).float()
        label = label.float()
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        out = net(img)
        loss = criterion(out, label)
        eval_loss += loss
        print('out:',out)
        print('label:',label)
    print('Test Loss: {:.6f}'.format(
        eval_loss/len(test_data)
    ))


if __name__ == '__main__':
    # train()
    test()
