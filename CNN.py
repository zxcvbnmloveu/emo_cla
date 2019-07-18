import numpy
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
            nn.Conv2d(1, 100, kernel_size=3, stride=1),
            nn.BatchNorm2d(100),
            nn.Tanh(),
            nn.Conv2d(100, 100, kernel_size=3),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout(0.5))
        self.fc1 = nn.Linear(86400, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.layer(x)
        # print(out.size())
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = F.tanh(out)
        out = F.dropout(out, p=0.25, training=self.training)
        out = self.fc2(out)
        return out



def data_prepare(X, y, BATCH_SIZE = 500, shuffle = True):

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

if __name__ == '__main__':

    file_x1 = "./data/features_train1.dat"
    file_y1 = './data/label_class_0_train.dat'
    file_x2 = "./data/features_test1.dat"
    file_y2 = './data/label_class_0_test.dat'

    X_train = numpy.genfromtxt(file_x1, delimiter=' ')
    y_train = numpy.genfromtxt(file_y1, delimiter=' ')
    X_train = StandardScaler().fit_transform(X_train)
    print(X_train.shape)
    X_train = X_train.reshape(-1, 1, 40, 101)
    print(X_train.shape)
    train_loader = data_prepare(X_train, y_train, BATCH_SIZE = 50)

    X_test = numpy.genfromtxt(file_x2, delimiter=' ')
    y_test = numpy.genfromtxt(file_y2, delimiter=' ')
    X_test = StandardScaler().fit_transform(X_test)
    X_test = X_test.reshape(-1, 1, 40, 101)
    test_loader = data_prepare(X_test, y_test, shuffle = False, BATCH_SIZE = 50)

    num_epochs = 50
    num_classes = 2
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ConvNet(num_classes).to(device)
    print(model)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-3)
    optimizer2 = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)
    optimizer3 = torch.optim.SGD(model.parameters(), lr=0.00001, weight_decay  = 1e-6, momentum = 0.9, nesterov =True)

    # Train the model
    total_step = len(train_loader)
    print(total_step)
    for epoch in range(num_epochs):
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
            if (i + 1) % 5 == 0:
                print('Epoch [{}/{}], Step [{}/{}] Loss: {}'
                      .format(epoch + 1, num_epochs, i+1, total_step, loss.item()))

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

    torch.save(model.state_dict(), "./DNNmodel.ckpt")
