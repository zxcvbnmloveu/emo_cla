import numpy
import torch
from sklearn.preprocessing import StandardScaler
import time
import torch.utils.data as Data
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# Fully connected neural network with one hidden layer
class DNN(nn.Module):
    def __init__(self, input_size, num_classes = 2):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 600)
        self.fc4 = nn.Linear(600, num_classes)

    def forward(self, inputs):
        out = F.relu(self.fc1(inputs))
        # out = F.dropout(out, p=0.25, training=self.training)
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.fc4(out)

        return out



def data_prepare(X, y, BATCH_SIZE = 500, shuffle = True):
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

    file_x1 = "./data/features_train_0123.dat"
    file_y1 = './data/label_class_0_train_0123.dat'

    X = numpy.genfromtxt(file_x1, delimiter=' ')
    y = numpy.genfromtxt(file_y1, delimiter=' ')

    # file_x2 = "./data/features_test_0123.dat"
    # file_y2 = './data/label_class_0_3class_test_0123.dat'

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=43)

    train_loader = data_prepare(X_train, y_train)

    test_loader = data_prepare(X_test, y_test, shuffle = False)

    num_epochs = 10
    num_classes = 2
    input_size = 3072
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = DNN(input_size, num_classes).to(device)
    print(model)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-3)
    optimizer2 = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)

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
            optimizer2.step()
            if (i + 1) % 10 == 0:
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
