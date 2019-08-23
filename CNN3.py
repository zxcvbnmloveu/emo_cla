import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import time
import torch.utils.data as Data
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 24, kernel_size=3, stride=2),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Conv1d(24, 16, kernel_size=3, stride=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc1 = nn.Linear(2544, 20)
        self.fc2 = nn.Linear(20, num_classes)

    def forward(self, x):
        out = self.layer(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.fc2(out)
        return out



def data_prepare(X, y, BATCH_SIZE = 40, shuffle = True):

    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    print(X.size(),y.size())
    torch_dataset = Data.TensorDataset(X, y)  # 把数据放在数据库中
    loader = Data.DataLoader(
        # 从dataset数据库中每次抽出batch_size个数据
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,  # 将数据打乱
        num_workers=2,  # 使用两个线程
    )
    return loader

def run(X_train,save_name):

    label_a = np.load('./data/label_a.npy')
    label_v = np.load('./data/label_v.npy')

    y_train = label_a
    X_train = StandardScaler().fit_transform(X_train)
    X_train = X_train.reshape(1280, 1, 1920)
    print(X_train.shape)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=43)
    print(X_train.shape, y_train.shape)
    train_loader = data_prepare(X_train, y_train, BATCH_SIZE=40)
    test_loader = data_prepare(X_test, y_test, shuffle=False, BATCH_SIZE=40)

    num_epochs = 5
    num_classes = 2
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ConvNet(num_classes).to(device)
    # print(model)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-3)
    optimizer2 = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)
    optimizer3 = torch.optim.SGD(model.parameters(), lr=0.00001, weight_decay=1e-6, momentum=0.9, nesterov=True)

    # Train the model
    total_step = len(train_loader)
    print('total_step: %d'%total_step)
    for epoch in range(num_epochs):
        print('epoch : %d'%(epoch+1))
        model.train()
        for i, (input, labels) in enumerate(train_loader):
            input = input.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(input.float())
            loss = criterion(outputs, labels.long())
            # print(loss)
            # Backward and optimize
            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()
            if (i + 1) % 31 == 0:
                print('Epoch [{}/{}], Step [{}/{}] Loss: {}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        # Test the model
        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for input, labels in test_loader:
                input = input.to(device)
                labels = labels.to(device)
                outputs = model(input.float())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted.int() == labels.int()).sum().item()
        print('Test Accuracy of the model on the test data: {} %'.format(100 * correct / total))

        with torch.no_grad():
            correct = 0
            total = 0
            for input, labels in train_loader:
                input = input.to(device)
                labels = labels.to(device)
                outputs = model(input.float())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted.int() == labels.int()).sum().item()
        print('Test Accuracy of the model on the train data: {} %'.format(100 * correct / total))

    torch.save(model.state_dict(), "./data/" + save_name + ".ckpt")

def loop():
    data = np.load('./data/seg_data.npy')
    print(data.shape)
    for ch in range(10):
        for seg in range(4):
            X_train = data[ch][seg]
            save_name = "Ch" + str(ch) + "Seg" + str(seg)
            run(X_train, save_name)

def test2():
    device = torch.device("cuda")
    data = np.load('./data/seg_data.npy')
    X_train = data[0][0][:40]
    print(X_train.shape)
    X_train = StandardScaler().fit_transform(X_train)
    X_train = X_train.reshape(-1, 1, 1920)
    X_train = torch.from_numpy(X_train)
    X_train = X_train.cuda().float()
    print(X_train.device)
    model = ConvNet(2).to(device)
    model.load_state_dict(torch.load("./DNNmodel.ckpt", map_location=device))

    # model.eval()
    outputs = model(X_train)

if __name__ == '__main__':
    loop()
