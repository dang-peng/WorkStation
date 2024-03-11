# -*- coding: utf-8 -*-
import datetime
import os
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from DatasetLoader import train_loader, test_loader


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding = 1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding = 1)
        self.pool2 = nn.MaxPool2d(2, 2, padding = 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        
        self.conv5 = nn.Conv2d(128, 128, 3, padding = 1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding = 1)
        self.conv7 = nn.Conv2d(128, 128, 1, padding = 1)
        self.pool3 = nn.MaxPool2d(2, 2, padding = 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        
        self.conv8 = nn.Conv2d(128, 256, 3, padding = 1)
        self.conv9 = nn.Conv2d(256, 256, 3, padding = 1)
        self.conv10 = nn.Conv2d(256, 256, 1, padding = 1)
        self.pool4 = nn.MaxPool2d(2, 2, padding = 1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        
        self.conv11 = nn.Conv2d(256, 512, 3, padding = 1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding = 1)
        self.conv13 = nn.Conv2d(512, 512, 1, padding = 1)
        self.pool5 = nn.MaxPool2d(2, 2, padding = 1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()
        
        self.fc14 = nn.Linear(512 * 4 * 4, 1024)
        self.drop1 = nn.Dropout()
        self.fc15 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.fc16 = nn.Linear(1024, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        # print(" x shape ",x.size())
        x = x.view(-1, 512 * 4 * 4)
        x = F.relu(self.fc14(x))
        x = self.drop1(x)
        x = F.relu(self.fc15(x))
        x = self.drop2(x)
        x = self.fc16(x)
        return x
    
    def Train(self, device):
        num_epochs = 301
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 5e-4)
        scheduler = StepLR(optimizer, step_size = 30, gamma = 0.5)
        # optimizer = optim.Adam(self.parameters(), lr = 0.0001)
        Loss = []
        Test_Accuracy = []
        for epoch in range(num_epochs):
            timestart = time.time()
            epoch_loss = 0.0
            total_test = 0
            correct_test = 0
            for i, train_data in enumerate(train_loader, 0):
                train_images, train_labels = train_data
                train_images, train_labels = train_images.to(device), train_labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward propagation
                y_hat = self(train_images)
                # calculate loss
                loss = loss_func(y_hat, train_labels)
                # backpropagation
                loss.backward()
                # update model parameter
                optimizer.step()
                epoch_loss += loss.item()
            Loss.append(epoch_loss / len(train_loader))
            scheduler.step()
            for test_data in train_loader:
                test_images, test_labels = test_data
                test_images, test_labels = test_images.to(device), test_labels.to(device)
                y_hat = self(test_images)
                _, predicted = torch.max(y_hat.data, 1)
                total_test += test_labels.size(0)
                correct_test += (predicted == test_labels).sum().item()
            Test_Accuracy.append(100.0 * correct_test / total_test)
            print('Epoch {} Cost Time: {:.2f} sec Loss: {:.2f} Test Accuracy: {:.2f} %'.format(epoch,
                                                                                               time.time() - timestart,
                                                                                               Loss[epoch],
                                                                                               100.0 * correct_test / total_test))
            # print('epoch %d cost %3f sec' % (epoch, time.time() - timestart))
            # 动态创建文件名以包含epoch信息
            folder_path = "./TrainData"
            os.makedirs(folder_path, exist_ok = True)
            filename = f"CNN_model_data_epoch_{epoch}.tar"
            file_path = os.path.join(folder_path, filename)
            torch.save({'epoch': epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_loss / len(train_loader)
                        }, file_path)
        current_time = datetime.datetime.now().strftime("%m_%d_%H_%M")
        with open(f'./TrainData/CNN_train_data_{current_time}.pickle', 'wb') as file:
            pickle.dump((Loss, Test_Accuracy), file)
    
    def Test(self, device):
        file_path = './TrainData/CNN_model_data_epoch_99.tar'
        checkpoint = torch.load(file_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        correct = 0
        total = 0
        with torch.no_grad():
            for test_data in test_loader:
                test_images, test_labels = test_data
                test_images, test_labels = test_images.to(device), test_labels.to(device)
                y_hat = self(test_images)
                _, predicted = torch.max(y_hat.data, 1)
                total += test_labels.size(0)
                correct += (predicted == test_labels).sum().item()
        print('Test Accuracy: {:.2f} %'.format(100.0 * correct / total))


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CNN = ConvolutionalNeuralNetwork()
    CNN = CNN.to(device)
    CNN.Train(device)
    CNN.Test(device)
