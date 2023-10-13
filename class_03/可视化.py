import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
from sklearn.datasets import load_iris
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class IrisData(Dataset):

    def __init__(self,X, y):
        self.x_data = X
        self.y_data = y
        # 转化为tensor
        self.x_data = torch.from_numpy(self.x_data).float()
        self.y_data = torch.from_numpy(self.y_data).long()

        self.n_samples = X.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


class IrisNet(torch.nn.Module):

    def __init__(self):
        super(IrisNet, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x



def train(model, opt, EPOCH=100):

    writer = SummaryWriter()
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(EPOCH):
        for batch_id, (x, y) in enumerate(train_loader):
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            writer.add_scalar('train_loss', loss.item(), epoch * len(train_loader) + batch_id)
            writer.flush()
            print('epoch {}, batch_id {}, loss {}'.format(epoch, batch_id, loss.item()))

if __name__ == '__main__':

    iris = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(iris[0], iris[1], test_size=0.2, random_state=32,shuffle=True)
    IrisDataset_train= IrisData(x_train,y_train)
    IrisDataset_test = IrisData(x_test, y_test)

    train_loader = DataLoader(dataset=IrisDataset_train, batch_size=10, shuffle=True)

    net = IrisNet()
    # print(net)
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    train(net, opt, EPOCH=100)
