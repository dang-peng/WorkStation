# -*- coding: utf-8 -*-
import datetime
import math
import os
import pickle
import time

import torch
import torch.nn as nn

from DatasetLoader import train_loader, test_loader


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.updates_num = 0
        self.conv1 = nn.Conv2d(3, 64, 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(True)
        
        self.conv2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(2, 2, padding = 1)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(True)
        
        self.conv4 = nn.Conv2d(128, 128, 3, padding = 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(True)
        self.pool4 = nn.MaxPool2d(2, 2, padding = 1)
        
        self.conv5 = nn.Conv2d(128, 256, 3, padding = 1)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU(True)
        
        self.conv6 = nn.Conv2d(256, 256, 3, padding = 1)
        self.bn6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU(True)
        
        self.conv7 = nn.Conv2d(256, 256, 3, padding = 1)
        self.bn7 = nn.BatchNorm2d(256)
        self.relu7 = nn.ReLU(True)
        self.pool7 = nn.MaxPool2d(2, 2, padding = 1)
        
        self.conv8 = nn.Conv2d(256, 512, 3, padding = 1)
        self.bn8 = nn.BatchNorm2d(512)
        self.relu8 = nn.ReLU(True)
        
        self.conv9 = nn.Conv2d(512, 512, 3, padding = 1)
        self.bn9 = nn.BatchNorm2d(512)
        self.relu9 = nn.ReLU(True)
        
        self.conv10 = nn.Conv2d(512, 512, 3, padding = 1)
        self.bn10 = nn.BatchNorm2d(512)
        self.relu10 = nn.ReLU(True)
        self.pool10 = nn.MaxPool2d(2, 2, padding = 1)
        
        self.conv11 = nn.Conv2d(512, 512, 3, padding = 1)
        self.bn11 = nn.BatchNorm2d(512)
        self.relu11 = nn.ReLU(True)
        
        self.conv12 = nn.Conv2d(512, 512, 3, padding = 1)
        self.bn12 = nn.BatchNorm2d(512)
        self.relu12 = nn.ReLU(True)
        
        self.conv13 = nn.Conv2d(512, 512, 3, padding = 1)
        self.bn13 = nn.BatchNorm2d(512)
        self.relu13 = nn.ReLU(True)
        self.pool13 = nn.MaxPool2d(2, 2, padding = 1)
        
        self.fc14 = nn.Linear(512 * 2 * 2, 4096)
        self.relu14 = nn.ReLU(True)
        self.drop1 = nn.Dropout()
        self.fc15 = nn.Linear(4096, 4096)
        self.relu15 = nn.ReLU(True)
        self.drop2 = nn.Dropout()
        self.fc16 = nn.Linear(4096, 10)
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu7(x)
        x = self.pool7(x)
        
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu8(x)
        
        x = self.conv9(x)
        x = self.bn9(x)
        x = self.relu9(x)
        
        x = self.conv10(x)
        x = self.bn10(x)
        x = self.relu10(x)
        x = self.pool10(x)
        
        x = self.conv11(x)
        x = self.bn11(x)
        x = self.relu11(x)
        
        x = self.conv12(x)
        x = self.bn12(x)
        x = self.relu12(x)
        
        x = self.conv13(x)
        x = self.bn13(x)
        x = self.relu13(x)
        x = self.pool13(x)
        # print(" x shape ", x.size())
        x = x.view(-1, 512 * 2 * 2)
        x = self.fc14(x)
        x = self.relu14(x)
        x = self.drop1(x)
        x = self.fc15(x)
        x = self.relu15(x)
        x = self.drop2(x)
        x = self.fc16(x)
        return x
    
    def Train(self, device):
        # optimizer = optim.SGD(self.parameters(), lr = 0.01, momentum = 0.9)
        # optimizer = optim.Adam(self.parameters(), lr = 0.0001)
        # scheduler = StepLR(optimizer, step_size = 30, gamma = 0.5)
        num_epochs = 301
        loss_func = nn.CrossEntropyLoss()
        initial_lr = 0.01
        lr = initial_lr
        step_size = 30
        gamma = 0.5
        # 初始化动量状态
        momentum_state = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                momentum_state[name] = torch.zeros_like(param.data)
        momentum = 0.9  # 动量因子
        # 累积权重变化和阈值
        accumulated_delta_w = {name: torch.zeros_like(param) for name, param in self.named_parameters()}
        # 设置更新阈值
        threshold = 0
        Loss = []
        Test_Accuracy = []
        for epoch in range(num_epochs):
            timestart = time.time()
            epoch_loss = 0.0
            total_test = 0
            correct_test = 0
            # 检查是否需要调整学习率
            if (epoch + 1) % step_size == 0:
                lr *= gamma  # 更新学习率
            for i, train_data in enumerate(train_loader, 0):
                train_images, train_labels = train_data
                train_images, train_labels = train_images.to(device), train_labels.to(device)
                self.zero_grad()
                y_hat = self(train_images)
                loss = loss_func(y_hat, train_labels)
                loss.backward()
                # 手动更新权重，加入动量优化
                with torch.no_grad():
                    for name, param in self.named_parameters():
                        if param.requires_grad:
                            # 计算动量更新
                            momentum_state[name] = momentum * momentum_state[name] + param.grad.data
                            # 更新参数
                            delta_weights = - lr * momentum_state[name]
                            # 判断层类型是否属于Conv2d或Linear层
                            submodule = self.get_submodule(name.split('.')[0])
                            if isinstance(submodule, (nn.Conv2d, nn.Linear)):
                                accumulated_delta_w[name] += delta_weights
                                # 检查累积权重变化是否达到阈值
                                mask = torch.abs(accumulated_delta_w[name]) > threshold
                                # 只更新超过阈值的权重变化，其他保持不变
                                if mask.any():
                                    weight_before_update = param.data.clone()
                                    update_weights = accumulated_delta_w[name] * mask.float()
                                    param.data += update_weights
                                    weight_after_update = param.data.clone()
                                    not_updated = (weight_before_update == weight_after_update).all()
                                    if not not_updated:
                                        # 直接使用Tensor操作来计算更新的数量
                                        self.updates_num += mask.sum().item()
                                    # 重置已更新权重的累积变化
                                    accumulated_delta_w[name] *= ~mask  # 使用逻辑非操作(~)将超过阈值的位置重置为0
                            elif not isinstance(submodule, (nn.BatchNorm2d, nn.Dropout)):
                                # 对其他层（排除BatchNorm和Dropout）应用即时更新策略
                                param.data += delta_weights
                epoch_loss += loss.item()
            Loss.append(epoch_loss / len(train_loader))
            for test_data in test_loader:
                test_images, test_labels = test_data
                test_images, test_labels = test_images.to(device), test_labels.to(device)
                y_hat = self(test_images)
                _, predicted = torch.max(y_hat.data, 1)
                total_test += test_labels.size(0)
                correct_test += (predicted == test_labels).sum().item()
            Test_Accuracy.append(100.0 * correct_test / total_test)
            print('Epoch: {} Cost Times: {:.2f} sec Loss: {:.2f} Test Accuracy: {:.2f} %'.format(epoch,
                                                                                                 time.time() - timestart,
                                                                                                 Loss[epoch],
                                                                                                 100.0 * correct_test / total_test))
            folder_path = "./TrainCumulativeThreshold"
            os.makedirs(folder_path, exist_ok = True)
            # 只在第300个epoch时保存
            if epoch == 300:
                filename = f"CNN_model_data_epoch_{epoch}.tar"
                file_path = os.path.join(folder_path, filename)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict(),
                    'learning_rate': lr,  # 保存当前学习率
                    'momentum_state': momentum_state,  # 保存动量状态
                    'loss': epoch_loss / len(train_loader)
                }, file_path)
        current_time = datetime.datetime.now().strftime("%m_%d_%H_%M")
        with open(f'./TrainCumulativeThreshold/CNN_train_data_{current_time}.pickle', 'wb') as file:
            pickle.dump((self.updates_num, Loss, Test_Accuracy), file)
    
    def Test(self, device):
        file_path = './TrainCumulativeThreshold/CNN_model_data_epoch_300.tar'
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
    # CNN.Test(device)
