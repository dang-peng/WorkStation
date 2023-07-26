import numpy as np

from GetDataset import GetDataset
import numpy as np

from GetDataset import GetDataset


class ConvolutionalLayer:
    def __init__(self, num_filters, kernel_size):
        self.last_input = None
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.filters = np.random.randn(num_filters, kernel_size, kernel_size) / (kernel_size * kernel_size)
    
    def iterate_regions(self, image):
        h, w = image.shape
        for i in range(h - self.kernel_size + 1):
            for j in range(w - self.kernel_size + 1):
                region = image[i:i + self.kernel_size, j:j + self.kernel_size]
                yield region, i, j
    
    def forward(self, input):
        self.last_input = input
        h, w = input.shape
        output = np.zeros((h - self.kernel_size + 1, w - self.kernel_size + 1, self.num_filters))
        for region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(region * self.filters, axis = (1, 2))
        return output
    
    def backward(self, d_L_d_out, learn_rate):
        d_L_d_filters = np.zeros(self.filters.shape)
        for region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * region
        self.filters -= learn_rate * d_L_d_filters
        return None


class MaxPoolingLayer:
    def __init__(self, size):
        self.last_input = None
        self.size = size
    
    def iterate_regions(self, image):
        h, w, _ = image.shape
        new_h = h // self.size
        new_w = w // self.size
        for i in range(new_h):
            for j in range(new_w):
                region = image[i * self.size:(i * self.size + self.size), j * self.size:(j * self.size + self.size)]
                yield region, i, j
    
    def forward(self, input):
        self.last_input = input
        h, w, num_filters = input.shape
        output = np.zeros((h // self.size, w // self.size, num_filters))
        for region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(region, axis = (0, 1))
        return output
    
    def backward(self, d_L_d_out):
        d_L_d_input = np.zeros(self.last_input.shape)
        for region, i, j in self.iterate_regions(self.last_input):
            h, w, f = region.shape
            amax = np.amax(region, axis = (0, 1))
            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        if region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i * self.size + i2, j * self.size + j2, f2] = d_L_d_out[i, j, f2]
        return d_L_d_input


class FullyConnectedLayer:
    def __init__(self, num_inputs, num_outputs):
        self.last_input = None
        self.last_input_shape = None
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.weights = np.random.randn(num_inputs, num_outputs) / num_inputs
        self.biases = np.zeros(num_outputs)
    
    def forward(self, input):
        self.last_input_shape = input.shape
        input = input.flatten()
        self.last_input = input
        output = np.dot(input, self.weights) + self.biases
        return output
    
    def backward(self, d_L_d_out, learn_rate):
        d_out_d_weights = self.last_input
        d_out_d_input = self.weights
        
        d_L_d_weights = np.dot(d_out_d_weights[np.newaxis].T, d_L_d_out[np.newaxis])
        d_L_d_biases = d_L_d_out
        d_L_d_input = np.dot(d_out_d_input, d_L_d_out)
        
        self.weights -= learn_rate * d_L_d_weights
        self.biases -= learn_rate * d_L_d_biases
        return d_L_d_input.reshape(self.last_input_shape)


class ConvNet:
    def __init__(self):
        self.conv1 = ConvolutionalLayer(8, 3)
        self.pool1 = MaxPoolingLayer(2)
        self.conv2 = ConvolutionalLayer(16, 3)
        self.pool2 = MaxPoolingLayer(2)
        self.dense1 = FullyConnectedLayer(7 * 7 * 16, 32)
        self.dense2 = FullyConnectedLayer(32, 10)
    
    def forward(self, input):
        output = self.conv1.forward(input)
        output = self.pool1.forward(output)
        output = self.conv2.forward(output)
        output = self.pool2.forward(output)
        output = self.dense1.forward(output)
        output = self.dense2.forward(output)
        return output
    
    def backward(self, d_L_d_out, learn_rate):
        d_L_d_out = self.dense2.backward(d_L_d_out, learn_rate)
        d_L_d_out = self.dense1.backward(d_L_d_out, learn_rate)
        d_L_d_out = self.pool2.backward(d_L_d_out)
        d_L_d_out = self.conv2.backward(d_L_d_out, learn_rate)
        d_L_d_out = self.pool1.backward(d_L_d_out)
        d_L_d_out = self.conv1.backward(d_L_d_out, learn_rate)


def main():
    # 加载MNIST数据集
    batch_size = 100
    train_images, train_labels, test_images, test_labels, batches = GetDataset(batch_size).get_data()
    # 创建模型
    model = ConvNet()
    # 训练模型
    epochs = 5
    learn_rate = 0.01
    for epoch in range(epochs):
        print("Epoch", epoch + 1)
        for i in range(batches):
            # 前向传播
            output = model.forward(train_images)
            # 计算损失
            loss = np.mean((output - train_labels) ** 2)
            # 反向传播
            model.backward(output - train_labels, learn_rate)
            if i % 100 == 0:
                print("Loss:", loss)
    
    # 在测试集上评估模型
    num_correct = 0
    for i in range(len(test_images)):
        output = model.forward(test_images[i])
        predicted_label = np.argmax(output)
        true_label = np.argmax(test_labels[i])
        if predicted_label == true_label:
            num_correct += 1
    accuracy = num_correct / len(test_images)
    print("Test accuracy:", accuracy)


if __name__ == '__main__':
    main()
