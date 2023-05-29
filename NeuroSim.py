import numpy as np

from DatasetLoader import DatasetLoader
from DatasetLoader import (Train_images, Train_labels)


class NeuralNetwork:
    def __init__(self, layer_sizes, batch_size, lr):
        self.learning_rate = lr
        self.layer_input = None
        self.activations = None
        self.activation_function = None
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.batch_size = batch_size
        self.weights = [np.random.rand(layer_sizes[i - 1], layer_sizes[i]) for i in range(1, self.num_layers)]
        # self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i - 1]) for i in range(1, self.num_layers)]
        self.biases = [np.zeros((self.batch_size, layer_sizes[i])) for i in range(1, self.num_layers)]
        # self.biases = [np.random.rand(y, 1) for y in layer_sizes[1:]]
        # self.weights = [np.random.rand(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        return 1.0 - np.tanh(x) * np.tanh(x)
    
    @staticmethod
    def logistic(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def logistic_derivative(self, x):
        return self.logistic(x) * (1 - self.logistic(x))
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    @staticmethod
    def loss_derivative(activations, y):
        result = [(a.argmax() - y) for a, y in zip(activations, y)]
        return result
    
    @staticmethod
    def calculate_loss(x, y):
        return 0.5 * (x - y) ** 2
    
    def forward(self, x):
        activation = x
        self.activations = [x]  # list to store all the activations, layer by layer
        self.layer_input = []  # list to store all the input vectors, layer by layer
        for i in range(self.num_layers - 1):
            # output = np.dot(self.activations[i], self.weights[i])
            z = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
            self.layer_input.append(z)
            activation = self.sigmoid(z)
            self.activations.append(activation)
        return self.activations[-1]
    
    def softmax(self, y, axis=1):
        y_row_max = y.max(axis = axis)
        y_row_max = y_row_max.reshape(-1, 1)
        y = y - y_row_max
        y_exp = np.exp(y)
        y_sum = np.sum(y_exp, axis = axis, keepdims = True)
        pred = y_exp / y_sum
        return pred
    
    def backward(self, x, y):
        m = x.shape[0]
        error = self.loss_derivative(x, y)
        deltas = [error * self.sigmoid_derivative(self.layer_input[-1])]
        for i in range(self.num_layers - 2, 0, -1):
            delta = np.dot(deltas[-1], self.weights[i].T) * (self.activations[i] * (1 - self.activations[i]))
            deltas.append(delta)
        deltas.reverse()
        dL_dW = [np.dot(self.activations[i].T, deltas[i]) / m for i in range(self.num_layers - 1)]
        dL_db = [np.mean(deltas[i], axis = 1, keepdims = True) for i in range(self.num_layers - 1)]
        self.weights = [self.weights[i] - self.learning_rate * dL_dW[i] for i in range(self.num_layers - 1)]
        self.biases = [self.biases[i] - self.learning_rate * dL_db[i] for i in range(self.num_layers - 1)]
    
    def compute_loss(self, x, y):
        m = x.shape[0]
        log_probs = -np.log(self.forward(x)[y, np.arange(m)])
        loss = np.sum(log_probs) / m
        return loss
    
    def mean_squared_error(self, x, y):
        # mse = np.sum((self.forward(x) - y) * (self.forward(x) - y)) / 2
        error = x - y
        mse = np.mean(np.square(error))
        return mse
    
    def train(self, x_train, y_train, epoch):
        x = self.forward(x_train)
        loss = self.mean_squared_error(x, y_train)
        self.backward(x, y_train)
        print("Epoch {} Loss: {:.4f}".format(epoch + 1, loss))
    
    def predict(self, test_images):
        test_result = np.argmax(np.dot(test_images, self.weights))
        return test_result
    
    def accuracy(self, test_data):
        result = 0
        all = 0
        for data, label in test_data:
            predict_label = self.forward(data[-1])
            # print(np.argmax(predict_label), np.argmax(label))
            for predict_label_i, l_i in zip(predict_label, label):
                all += 1
                # print(np.argmax(predict_label_i), np.argmax(l_i))
                if np.argmax(predict_label_i) == np.argmax(l_i):
                    result += 1
        # print(result, all, result * 100 / all)
        return result / all


learning_rate = 0.1
num_epochs = 10
input_size = 784
hidden_sizes = [10]
output_size = 10
batch_size = 32
layer_sizes = [input_size] + hidden_sizes + [output_size]
model = NeuralNetwork(layer_sizes, batch_size, learning_rate)

i = 1

for epoch, _ in enumerate(range(2)):
    train_batch = DatasetLoader(Train_images, Train_labels, batch_size = 32)
    train_data = train_batch.data_loader
    for train_images_batch, train_labels_batch in train_data:
        model.train(train_images_batch, train_labels_batch, epoch)
        print("i = ", i)
        i = i + 1

test_batch = DatasetLoader(Train_images, Train_labels, batch_size = 32)
test_data = test_batch.data_loader
# for test_images_batch, test_labels_batch in test_data:
accuracy = model.accuracy(test_data)
print(accuracy)
