import pickle

import numpy as np
from matplotlib import pyplot as plt

from GetDataset import GetDataset


def binary_matrix_conversion(activations, random_matrix):
    # for i in range(len(activations)):
    #     for j in range(len(activations[i])):
    #         if activations[i][j] > random_matrix[i][j]:
    #             activations[i][j] = 1
    #         else:
    #             activations[i][j] = 0
    return (activations > random_matrix).astype('float')


def sigmoid(z):
    # y_active = 1.0 / (1.0 + np.exp(-z))
    # shape = y_active.shape
    # binary_y = np.random.randn(*shape)
    # new_y = binary_matrix_conversion(y_active, binary_y)
    # return new_y
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_binary(z):
    y_active = 1.0 / (1.0 + np.exp(-z))
    shape = y_active.shape
    binary_y = np.random.rand(*shape)
    new_y = binary_matrix_conversion(y_active, binary_y)
    return new_y


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis = 1, keepdims = True)


def softmax_binary(z):
    y_active = np.exp(z) / np.sum(np.exp(z), axis = 1, keepdims = True)
    shape = y_active.shape
    binary_y = np.random.rand(*shape)
    new_y = binary_matrix_conversion(y_active, binary_y)
    return new_y


def cross_entropy_loss(y_pred, y_true):
    loss = -np.sum(y_true * np.log(y_pred), axis = 1)
    # 计算平均损失
    mean_loss = np.mean(loss)
    return mean_loss


class NeuralNetwork(object):
    def __init__(self, layer_sizes, lr):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.learning_rate = lr
        self.weights = [1.0 / np.sqrt(layer_sizes[i - 1]) * np.random.randn(layer_sizes[i - 1], layer_sizes[i]) for i in
                        range(1, self.num_layers)]
        self.biases = [np.zeros(layer_sizes[i]) for i in range(1, self.num_layers)]
        # self.weights = [np.random.rand(i, j) for i, j in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
        # self.biases = [np.random.rand(1, j) for j in self.layer_sizes[1:]]
    
    def forward(self, x):
        activations = [x]
        layer_input = []
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(activations[-1], weight) + bias
            layer_input.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        layer_input.append(z)
        activation = softmax(z)
        activations.append(activation)
        return activations[-1]
    
    def forward_binary(self, x):
        activations = [x]
        layer_input = []
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(activations[-1], weight) + bias
            layer_input.append(z)
            activation = sigmoid_binary(z)
            activations.append(activation)
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        layer_input.append(z)
        activation = softmax_binary(z)
        activations.append(activation)
        return activations[-1]
    
    def backward(self, _images, _labels):
        dL_db = [np.zeros(b.shape) for b in self.biases]
        dL_dw = [np.zeros(w.shape) for w in self.weights]
        activation = _images
        activations = [_images]
        layer_input = []
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(activations[-1], weight) + bias
            layer_input.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        layer_input.append(z)
        activation = softmax(z)
        activations.append(activation)
        delta = activations[-1] - _labels
        loss = cross_entropy_loss(activations[-1], _labels)
        dL_dw[-1] = np.dot(activations[-2].T, delta)
        dL_db[-1] = np.sum(delta, axis = 0, keepdims = True)
        for i in range(2, self.num_layers):
            z = layer_input[-i]
            delta = np.dot(delta, self.weights[-i + 1].T) * sigmoid_derivative(z)
            dL_db[-i] = np.sum(delta, axis = 0, keepdims = True)
            dL_dw[-i] = np.dot(activations[-i - 1].T, delta)
        # dL_dw = [np.dot(activations[i].T, delta) / m for i in range(self.num_layers - 1)]
        # dL_db = [np.mean(delta, axis = 1, keepdims = True) for i in range(self.num_layers - 1)]
        self.weights = [weight - learning_rate * dL_dw[i] for i, weight in enumerate(self.weights)]
        # self.biases = [bias - learning_rate * dL_db[j] for j, bias in enumerate(self.biases)]
        # self.find_max_min()
        return loss
    
    def test(self, test_images, test_labels):
        total = 0
        y_hat = []
        labels = []
        corrects = 0
        y_pred = []
        true_labels = []
        for _images_batch, _labels_batch in zip(test_images, test_labels):
            y_forward = self.forward(_images_batch)
            labels.append(_labels_batch)
            y_hat.append(y_forward)
        for sublist1 in y_hat:
            for sublist2 in sublist1:
                y_pred.append(sublist2)
        for sublist1 in labels:
            for sublist2 in sublist1:
                true_labels.append(sublist2)
        for predict_label, true_label in zip(y_pred, true_labels):
            total += 1
            # print(np.argmax(predict_label), np.argmax(true_label))
            if np.argmax(predict_label) == np.argmax(true_label):
                corrects += 1
        accuracy = 100 * corrects / total
        # print("Epoch: {}  Test Accuracy = {}/{} = {:.2f}".format(epoch + 1, corrects, len(total), accuracy))
        return accuracy
    
    def test_binary(self, times, test_images, test_labels):
        corrects = 0
        total = 0
        true_labels = []
        M, N, Z = np.array(test_labels).shape
        test_results = np.zeros((M * N, Z))
        for i in range(times):
            y_hat = []
            labels = []
            y_pred = []
            for _images_batch, _labels_batch in zip(test_images, test_labels):
                y_forward = self.forward_binary(_images_batch)
                labels.append(_labels_batch)
                y_hat.append(y_forward)
            for sublist1 in y_hat:
                for sublist2 in sublist1:
                    y_pred.append(sublist2)
            for sublist1 in labels:
                for sublist2 in sublist1:
                    true_labels.append(sublist2)
            test_results = test_results + np.array(y_pred)
        for predict_label, true_label in zip(test_results, true_labels):
            total += 1
            # print(np.argmax(predict_label), np.argmax(true_label))
            if np.argmax(predict_label) == np.argmax(true_label):
                corrects += 1
        accuracy = 100 * corrects / total
        # print("Epoch: {}  Test Accuracy = {}/{} = {:.2f}".format(epoch + 1, corrects, len(total), accuracy))
        return accuracy
    
    def mse_derivative(self, x, y):
        error = x - y
        mse = np.mean(np.square(error))
        return mse, error


# if __name__ == '__main__':
learning_rate = 0.01
num_epochs = 100
batch_size = 100
input_size = 784
hidden_sizes = [500, 300]
output_size = 10
layer_sizes = [input_size] + hidden_sizes + [output_size]
train_images, train_labels, test_images, test_labels, batches = GetDataset(batch_size).get_data()
model = NeuralNetwork(layer_sizes, lr = learning_rate)
Loss = []
loss = []
epochs = []
Test_Accuracy = []
Test_Accuracy_binary = []
Test_Accuracy_hardware = []
Train_Accuracy = []
Test_times = 10
for epoch in range(num_epochs):
    for i, (_images_batch, _labels_batch) in enumerate(zip(train_images, train_labels)):
        loss = model.backward(_images_batch, _labels_batch)
        if i % batches == 0:
            # print("Epoch: {} batch:{}  loss = {:.4f}".format(epoch, i, 100 * loss))
            Loss.append(loss)
    epochs.append(epoch)
    # train_accuracy = model.test(train_images, train_labels)
    # Train_Accuracy.append(train_accuracy)
    # print("Epoch: {}  Loss: {:.2f}  Train Accuracy: {:.2f}".format(epoch + 1, loss[epoch], train_accuracy))
    test_accuracy = model.test(test_images, test_labels)
    Test_Accuracy.append(test_accuracy)
    #
    # test_accuracy_binary = model.test_binary(Test_times, test_images, test_labels)
    # Test_Accuracy_binary.append(test_accuracy_binary)
    
    # UsingHardwareInference = HardwareSimulation(model.weights)
    # test_accuracy_hardware = UsingHardwareInference.Test_Hardware(Test_times, test_images, test_labels)
    # Test_Accuracy_hardware.append(test_accuracy_hardware)
    print("Epoch: {}  Loss: {:.2f}    Test Accuracy: {:.2f}    ".format(epoch + 1, Loss[epoch], test_accuracy))
# original_list = model.weights
# filename = "weight_data.txt"
# with open(filename, "w") as file:
#     for array in original_list:
#         for row in array:
#             file.write(" ".join(map(str, row)) + "\n")
#         file.write("---\n")\
filename = "./weights_biases.pickle"
with open(filename, 'wb') as file:
    pickle.dump((model.weights, model.biases), file)

# 绘制loss和epoch关系图
plt.plot(epochs, Loss, 'r-', label = 'Training loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.grid(True)
plt.show()
# 绘制Train Accuracy和epoch关系图
# plt.plot(epochs, Train_Accuracy, 'g-', label = 'Training Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Train_Accuracy')
# plt.title('Train_Accuracy vs Epoch')ll
# plt.grid(True)
# plt.show()
# 绘制Test Accuracy和epoch关系图
# plt.plot(epochs, Test_Accuracy, 'b-', label = 'Test Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Test_Accuracy')
# plt.title('Test_Accuracy vs Epoch')
# plt.grid(True)
# plt.show()l
# 绘制Test Accuracy和epoch关系图
# plt.plot(epochs, Test_Accuracy_binary, 'b-', label = 'Test Accuracy binary')
# plt.xlabel('Epoch')
# plt.ylabel('Test_Accuracy_Binary')
# plt.title('Test_Accuracy_Binary vs Epoch')
# plt.grid(True)
# plt.show()
# 绘制Test Accuracy和epoch关系图
# plt.plot(epochs, Test_Accuracy_hardware, 'b-', label = 'Test Accuracy hardware')
# plt.xlabel('Epoch')
# plt.ylabel('Test_Accuracy_Hardware')
# plt.title('Test_Accuracy_Hardware vs Epoch')
# plt.grid(True)
# plt.show()
