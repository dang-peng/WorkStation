import numpy as np

# 加载MNIST数据集
from MNIST import (x_train, y_train, x_test, y_test)


# 定义神经网络模型类
class NeuralNetwork:
    def __init__(self, layer_sizes, lr):
        self.learning_rate = lr
        self.layer_input = None
        self.activations = None
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i - 1]) for i in range(1, self.num_layers)]
        self.biases = [np.zeros((layer_sizes[i], 1)) for i in range(1, self.num_layers)]
    
    # 激活函数
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    # Sigmoid的导数
    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    @staticmethod
    def loss_derivative(activations, y):
        # ∂L/∂y = y-Y
        result = [(a - y) for a, y in zip(activations, y)]
        return result
    
    @staticmethod
    def calculate_loss(x, y):
        return 0.5 * (x - y) ** 2
    
    # 前向传播
    def forward(self, x):
        activation = x
        self.activations = [x]
        self.layer_input = []  # 存储除第一层之外的每一层输入
        for i in range(self.num_layers - 1):
            z = np.dot(self.weights[i], self.activations[i]) + self.biases[i]
            self.layer_input.append(z)
            activation = self.sigmoid(z)
            self.activations.append(activation)
        return self.activations[-1]
    
    # 反向传播和参数更新
    def backward(self, x, y):
        m = x.shape[0]
        # 𝛿 = (y-Y)*y*(1-y)
        # error = self.loss_derivative(self.activations[-1], y)
        deltas = [self.loss_derivative(self.activations[-1], y) * self.sigmoid_derivative(self.layer_input[-1])]
        # delta = error * self.sigmoid_derivative(self.layer_input[-1])
        for i in range(self.num_layers - 2, 0, -1):
            delta = np.dot(self.weights[i].T, deltas[-1]) * (self.activations[i] * (1 - self.activations[i]))
            deltas.append(delta)
        deltas.reverse()
        # ∂Lⱼ/∂wᵢ = 𝛿*X
        dL_dW = [np.dot(deltas[i], self.activations[i].T) / m for i in range(self.num_layers - 1)]
        dL_db = [np.mean(deltas[i], axis = 1, keepdims = True) for i in range(self.num_layers - 1)]
        # 参数更新
        self.weights = [self.weights[i] - self.learning_rate * dL_dW[i] for i in range(self.num_layers - 1)]
        self.biases = [self.biases[i] - self.learning_rate * dL_db[i] for i in range(self.num_layers - 1)]
    
    # 交叉熵损失函数
    def compute_loss(self, x, y):
        m = x.shape[0]  # 样本数量
        log_probs = -np.log(self.forward(x)[y, np.arange(m)])  # 计算每个样本对应类别的负对数概率
        loss = np.sum(log_probs) / m  # 计算平均损失
        return loss * 10 ** 8
    
    # 计算均方误差损失函数
    def mean_squared_error(self, x, y):
        m = x.shape[0]  # 样本数量
        test_result = self.forward(x)[y, np.arange(m)]
        return np.sum([0.5 * (x - y) ** 2 for (x, y) in test_result])
    
    # 模型训练
    def train(self, x_train, y_train):
        for i in range(len(x_train)):
            self.forward(x_train[i])
            self.backward(x_train[i], y_train[i])
            # 计算并输出损失
            loss = self.compute_loss(x_train[i], y_train[i])
            print("Epoch {} Loss: {:.4f}".format(epoch + 1, loss))
        # mse = self.mean_squared_error(x_train, y_train)
        # print("Epoch {} Loss: {:.4f}".format(epoch + 1, mse))
    
    # 预测
    def predict(self, test_input):
        test_result = [self.forward(x) for x in test_input]
        return test_result
    
    def accuracy(self, x_test, y_test):
        count = 0
        for i in range(len(x_test)):
            y_hat = self.forward(x_test[i])
            y_pred = np.argmax(y_hat, axis = 0)
            for j in range(len(x_test[i])):
                if y_pred[i] == y_test[i][j]:
                    count = count + 1
        sum_samples = len(x_test) * len(x_test[0])
        accuracy = 100 * count / sum_samples
        return accuracy


# 设置超参数
learning_rate = 0.01
num_epochs = 10
# 创建神经网络模型
input_size = 100
hidden_sizes = 64  # 可根据需求设置隐藏层的大小和层数
output_size = 10
layer_sizes = [input_size] + [hidden_sizes] + [output_size]
model = NeuralNetwork(layer_sizes, learning_rate)
# 训练模型
# from MNIST import (x_train, y_train, x_test, y_test)
for epoch in range(1000):
    model.train(x_train, y_train)
accuracy = model.accuracy(x_test, y_test)
print("Inference Accuracy: {:.2f}".format(accuracy))
