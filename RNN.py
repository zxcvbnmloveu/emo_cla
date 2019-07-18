import numpy
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
from sklearn.preprocessing import StandardScaler
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

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

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out




if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters
    sequence_length = 10
    input_size = 768
    hidden_size = 128
    num_layers = 2
    num_classes = 2
    batch_size = 20
    num_epochs = 20
    learning_rate = 0.01

    model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
    print(model)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    filex = "./data/features_train_ch1.dat"
    filey = './data/label_class_0_train_aug.dat'

    x = numpy.genfromtxt(filex, delimiter=' ')
    x = StandardScaler().fit_transform(x)
    x = x.reshape(-1, sequence_length, input_size)
    y = numpy.genfromtxt(filey, delimiter=' ')
    print(x.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=43)
    print(X_train.shape, y_train.shape)

    train_loader = data_prepare(X_train, y_train, BATCH_SIZE = batch_size)
    test_loader = data_prepare(X_test, y_test, shuffle=False, BATCH_SIZE=batch_size)

    total_step = len(train_loader)
    print("total_step:" + str(total_step))
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 6 == 0:
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


    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')