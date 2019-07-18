import numpy
import torch
from sklearn.preprocessing import StandardScaler
import time
import torch.utils.data as Data
from torch import nn
import torch.nn.functional as F


# Fully connected neural network with one hidden layer
class DNN(nn.Module):
    def __init__(self, input_size, num_classes = 2):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 5000)
        self.fc2 = nn.Linear(5000, 500)
        self.fc3 = nn.Linear(500, 1000)
        self.fc4 = nn.Linear(1000, num_classes)

    def forward(self, inputs):
        out = F.relu(self.fc1(inputs))
        out = F.dropout(out, p=0.1, training=self.training)
        out = F.relu(self.fc2(out))
        out = F.dropout(out, p=0.25, training=self.training)
        out = F.relu(self.fc3(out))
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.fc4(out)

        return out



def data_prepare(file_x, file_y, BATCH_SIZE = 155, shuffle = True):
    X = numpy.genfromtxt(file_x, delimiter=' ')
    y = numpy.genfromtxt(file_y, delimiter=' ')
    X = StandardScaler().fit_transform(X)

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

    file_x1 = "./data/features_train2.dat"
    file_y1 = './data/label_class_0_train.dat'
    train_loader = data_prepare(file_x1, file_y1)

    file_x2 = "./data/features_test2.dat"
    file_y2 = './data/label_class_0_test.dat'
    test_loader = data_prepare(file_x2, file_y2)

    num_epochs = 100
    num_classes = 2
    input_size =4000
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = DNN(input_size, num_classes).to(device)
    print(model)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer1 = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-3)
    optimizer2 = torch.optim.RMSprop(model.parameters(), lr=0.00001, alpha=0.9)

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
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()
            if (i + 1) % 8 == 0:
                print('Epoch [{}/{}],  Loss: {}'
                      .format(epoch + 1, num_epochs, loss.item()))
        if ((epoch + 1) % 10 == 0):
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

            model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
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