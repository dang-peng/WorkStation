import numpy as np

from DatasetLoader import DatasetLoader
from DatasetLoader import (Train_images, Train_labels, Test_images, Test_labels)


class Layer:
    def __init__(self, input_size, output_size, activation, lr):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.learning_rate = lr
        self.weights = 1.0 / np.sqrt(input_size) * np.random.randn(input_size, output_size)
        self.biases = np.zeros((1, output_size))
        self.x = None
        self.z = None
        self.y = None
        self.dy_dz = None
        self.dL_dz = None
        self.dL_db = None
        self.dL_dw = None
        self.dL_dy = None
        self.dL_dz = None
        self.dL_dx = None
    
    def forward(self, x):
        self.x = np.array(x)
        self.z = np.dot(self.x, self.weights) + self.biases
        (self.y, self.dy_dz) = self.activation(self.z)
        return self.y
    
    def backward(self, dL_dy):
        self.dL_dy = np.array(dL_dy)
        self.dL_dz = self.dL_dy * self.dy_dz
        self.dL_dx = np.dot(self.dL_dz, self.weights.transpose())
        return self.dL_dx
    
    def weight_update(self):
        self.dL_dw = np.dot(self.dL_dz.transpose(), self.x).transpose()  # / self.x.shape[0]
        self.dL_db = np.sum(self.dL_dz, axis = 0)
        self.weights = self.weights - self.learning_rate * self.dL_dw
        self.biases = self.biases - self.learning_rate * self.dL_db


def sigmoid(z):
    y = 1.0 / (1.0 + np.exp(-z))
    dy_dz = y * (1.0 - y)
    return (y, dy_dz)


class Network:
    def __init__(self, layer_sizes, lr):
        self.layer_sizes = layer_sizes
        self.learning_rate = lr
        self.layers = []
        self.create_layers()
    
    def create_layers(self):
        for i in range(1, len(layer_sizes)):
            self.layers.append(Layer(layer_sizes[i - 1], layer_sizes[i], activation = sigmoid, lr = self.learning_rate))
    
    def train(self, X, Y):
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].forward(X)
            else:
                self.layers[i].forward(self.layers[i - 1].y)
        for i in reversed(range(len(self.layers))):
            if i == len(self.layers) - 1:
                error = self.layers[i].y - Y
                mse = np.mean(np.square(error))
                # print("Loss: ", mse)
                self.layers[i].backward(error)
            else:
                self.layers[i].backward(self.layers[i + 1].dL_dx)
        for i in range(len(self.layers)):
            self.layers[i].weight_update()
    
    def test(self, X, Y):
        all = 0.0
        result = 0.0
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].forward(X)
            else:
                self.layers[i].forward(self.layers[i - 1].y)
        for predict_label_i, l_i in zip(self.layers[-1].y, Y):
            all += 1
            if np.argmax(predict_label_i) == np.argmax(l_i):
                result += 1
        return 100 * result / all


learning_rate = 0.01
num_epochs = 100
input_size = 784
hidden_sizes = [300]
output_size = 10
layer_sizes = [input_size] + hidden_sizes + [output_size]

# def __init__(self, layer_sizes, lr):
model = Network(layer_sizes, learning_rate)

for epoch, _ in enumerate(range(num_epochs)):
    train_batch = DatasetLoader(Train_images, Train_labels, batch_size = 100)
    train_data = train_batch.data_loader
    for train_images_batch, train_labels_batch in train_data:
        model.train(train_images_batch, train_labels_batch)
    accuracy = model.test(Test_images, Test_labels)
    print("Epoch:", epoch, " Inference Accuracy: {:.2f}".format(accuracy))
