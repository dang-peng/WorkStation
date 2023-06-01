import numpy as np
from matplotlib import pyplot as plt

# from DatasetProcess import (Train_images, Train_labels, Test_images, Test_labels)
from GetDataset import GetDataset


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


class NeuralNetwork(object):
    def __init__(self, layer_sizes, lr):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.learning_rate = lr
        self.weights = [np.random.randn(layer_sizes[i - 1], layer_sizes[i]) for i in range(1, self.num_layers)]
        self.biases = [np.random.randn(1, layer_sizes[i]) for i in range(1, self.num_layers)]
        # self.weights = [np.random.rand(i, j) for i, j in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
        # self.biases = [np.random.rand(1, j) for j in self.layer_sizes[1:]]
    
    def before_update(self):
        for i in range(self.num_layers - 1):
            print("before update weights[{}]={}x{}".format(i, len(self.weights[i]), len(self.weights[i][0])))
            print("before update biases[{}]={}x{}".format(i, len(self.biases[i]), len(self.biases[i][0])))
    
    def after_update(self):
        for i in range(self.num_layers - 1):
            print("after update weights[{}]={}x{}".format(i, len(self.weights[i]), len(self.weights[i][0])))
            print("after update biases[{}]={}x{}".format(i, len(self.biases[i]), len(self.biases[i][0])))
    
    def forward(self, x):
        # 批量化测试
        activations = [x]
        layer_input = []
        for weight, bias in zip(self.weights, self.biases):
            z = np.dot(activations[-1], weight) + bias
            layer_input.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        return activations[-1]
    
    # def gradient_descent(self, epoch, train_images, train_labels):
    #     loss = self.backward(train_images, train_labels)
    #     # print("Epoch: {}   loss = {:.4f}".format(epoch, 100 * loss))
    
    def backward(self, _images, _labels):
        dL_db = [np.zeros(b.shape) for b in self.biases]
        dL_dw = [np.zeros(w.shape) for w in self.weights]
        activation = _images
        activations = [_images]
        layer_input = []
        m = _images.shape[0]
        for weight, bias in zip(self.weights, self.biases):
            z = np.dot(activation, weight) + bias
            layer_input.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        loss, error = self.mse_derivative(activations[-1], _labels)
        delta = error * sigmoid_derivative(layer_input[-1])
        dL_dw[-1] = np.dot(activations[-2].T, delta)
        dL_db[-1] = delta
        for i in range(2, self.num_layers):
            z = layer_input[-i]
            delta = np.dot(delta, self.weights[-i + 1].T) * sigmoid_derivative(z)
            dL_db[-i] = np.sum(delta, axis = 0, keepdims = True)
            dL_dw[-i] = np.dot(activations[-i - 1].T, delta)
        # dL_dw = [np.dot(activations[i].T, delta) / m for i in range(self.num_layers - 1)]
        # dL_db = [np.mean(delta, axis = 1, keepdims = True) for i in range(self.num_layers - 1)]
        self.weights = [weight - learning_rate * dL_dw[i] for i, weight in enumerate(self.weights)]
        self.biases = [bias - learning_rate * dL_db[j] for j, bias in enumerate(self.biases)]
        # self.weights = [self.weights[i] - self.learning_rate * dL_dw[i] for i in range(self.num_layers - 1)]
        # self.biases = [self.biases[i] - self.learning_rate * dL_db[i] for i in range(self.num_layers - 1)]
        return loss
    
    def test(self, test_images, test_labels):
        # if test_labels not split
        # batch_size = 100
        # total_y_hat = []
        # total_labels = []
        # for i in range(0, len(test_images), batch_size):
        #     images_batch = test_images[i:i + batch_size]
        #     labels_batch = test_labels[i:i + batch_size]
        #     y_hat = model.forward(images_batch)
        #     total_y_hat.extend(y_hat)
        #     total_labels.extend(labels_batch)
        total = 0
        y_hat = []
        labels = []
        corrects = 0
        y_pred = self.forward(test_images)
        for sublist in y_pred:
            for sub_sublist in sublist:
                y_hat.append(sub_sublist)
        # y_hat = [sub_sublist for sublist in y_pred for sub_sublist in sublist]
        for sublist in test_labels:
            for sub_sublist in sublist:
                labels.append(sub_sublist)
        # labels = [sub_sublist for sublist in test_labels for sub_sublist in sublist]
        for predict_label, true_label in zip(y_hat, labels):
            total += 1
            # print(np.argmax(predict_label), np.argmax(true_label))
            if np.argmax(predict_label) == np.argmax(true_label):
                corrects += 1
        accuracy = 100 * corrects / total
        # print("Epoch: {}  Test Accuracy = {}/{} = {:.2f}".format(epoch + 1, corrects, len(total), accuracy))
        return corrects, total, accuracy
    
    def mse_derivative(self, x, y):
        error = x - y
        mse = np.mean(np.square(error))
        return mse, error


def train(epochs, train_images, train_labels, test_images, test_labels):
    for epoch in range(epochs):
        for i, (_images_batch, _labels_batch) in enumerate(zip(train_images, train_labels)):
            loss = model.backward(_images_batch, _labels_batch)
            if i % 500 == 0:
                print("Epoch: {} batch:{}  loss = {:.4f}".format(epoch, i, 100 * loss))
        corrects, total, accuracy = model.test(test_images, test_labels)
        print("Epoch: {}  Test Accuracy = {}/{} = {:.2f}".format(epoch + 1, corrects, total, accuracy))


# if __name__ == '__main__':
learning_rate = 0.01
num_epochs = 30
batch_size = 100
input_size = 784
hidden_sizes = [32]
output_size = 10
layer_sizes = [input_size] + hidden_sizes + [output_size]
train_images, train_labels, test_images, test_labels = GetDataset(batch_size).get_data()
model = NeuralNetwork(layer_sizes, lr = learning_rate)
# train(num_epochs, train_images, train_labels, test_images, test_labels)
Loss = []
epochs = []
Test_Accuracy = []
Train_Accuracy = []
for epoch in range(num_epochs):
    for i, (_images_batch, _labels_batch) in enumerate(zip(train_images, train_labels)):
        mse = model.backward(_images_batch, _labels_batch)
        # if i % 600 == 0:
        #     print("Epoch: {} batch:{}  loss = {:.4f}".format(epoch, i, 100 * loss))
        Loss.append(mse)
    epochs.append(epoch)
    corrects, total, test_accuracy = model.test(test_images, test_labels)
    Test_Accuracy.append(test_accuracy)
    print("Epoch: {}  Test Accuracy = {}/{} = {:.2f}".format(epoch + 1, corrects, total, test_accuracy))
    corrects, total, train_accuracy = model.test(test_images, test_labels)
    Train_Accuracy.append(train_accuracy)
    print("Epoch: {}  Test Accuracy = {}/{} = {:.2f}".format(epoch + 1, corrects, total, train_accuracy))
step = 600
loss = Loss[::step]
# 绘制loss和epoch关系图
plt.plot(epochs, loss, 'r-', label = 'Training loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.grid(True)
plt.show()
# 绘制Test Accuracy和epoch关系图
plt.plot(epochs, Train_Accuracy, 'g-', label = 'Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Train_Accuracy')
plt.title('Train_Accuracy vs Epoch')
plt.grid(True)
plt.show()
# 绘制Train Accuracy和epoch关系图
plt.plot(epochs, Test_Accuracy, 'y-', label = 'Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Test_Accuracy')
plt.title('Test_Accuracy vs Epoch')
plt.grid(True)
plt.show()
