import os
import struct

import numpy as np
from matplotlib import pyplot as plt


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    # use in windows
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte/%s-labels.idx1-ubyte' % (kind, kind))
    images_path = os.path.join(path, '%s-images.idx3-ubyte/%s-images.idx3-ubyte' % (kind, kind))
    # use in linux
    # labels_path = os.path.join(path,
    #                            '%s-labels-idx1-ubyte'
    #                            % kind)
    # images_path = os.path.join(path,
    #                            '%s-images-idx3-ubyte'
    #                            % kind)
    with open(labels_path, 'rb') as dbpath:
        print("Extracting  %s" % labels_path)
        magic, n = struct.unpack('>II',
                                 dbpath.read(8))
        labels = np.fromfile(dbpath,
                             dtype = np.uint8)
    
    with open(images_path, 'rb') as imgpath:
        print("Extracting  %s" % images_path)
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype = np.uint8).reshape(len(labels), 784)
    
    return images, labels


(X_Tr, Y_Tr) = load_mnist("f:/Code/practice/Learning/Python/dataset/", kind = 'train')
print("X_Train:", X_Tr)
print("y_train:", Y_Tr)
(X_Te, Y_Te) = load_mnist("f:/Code/practice/Learning/Python/dataset/", kind = 't10k')
print('X_Test:  ', X_Te.shape)
print('Y_Test:  ', Y_Te.shape)
print('X_Train: ', X_Tr.shape)
print('Y_Train: ', Y_Tr.shape)
# 数据预处理
X_Train = X_Tr / 255.0 - 0.5
Y_Train = Y_Tr
a = X_Train.shape
b = Y_Train.shape
# 数据预处理
X_Test = X_Te / 255.0 - 0.5
Y_Test = Y_Te
c = X_Test.shape
d = Y_Test.shape
# 设置超参数
batch_size = 100
num_train_samples = X_Train.shape[0]
num_test_samples = X_Test.shape[0]
# 随机打乱数据
permutation = np.random.permutation(num_train_samples)
shuffled_X_Train = X_Train[permutation]
shuffled_Y_Train = Y_Train[permutation]
x_train = []
y_train = []
x_test = []
y_test = []
for batch_start in range(0, num_train_samples, batch_size):
    batch_end = batch_start + batch_size
    X_train_batch = shuffled_X_Train[batch_start:batch_end]
    y_train_batch = shuffled_Y_Train[batch_start:batch_end]
    x_train.append(X_train_batch)
    y_train.append(y_train_batch)
for batch_start in range(0, num_test_samples, batch_size):
    batch_end = batch_start + batch_size
    X_test_batch = X_Test[batch_start:batch_end]
    y_test_batch = Y_Test[batch_start:batch_end]
    x_test.append(X_test_batch)
    y_test.append(y_test_batch)
print('finish')
# for i in range(X_Tr.shape[0]):
#     for j in range(X_Tr.shape[1]):
#         print(X_Tr[i][j])
# print(X_Tr.shape[0])
# print(X_Tr.shape[1])
# print(Y_Tr.shape[0])
# print(X_Test.shape[0])
# print(X_Test.shape[0])
# print(Y_Test.shape[0])
# plt.figure()
# for i in range(10):
#     im = X_Tr[i].reshape(28, 28)  # 训练数据集的第i张图，将其转化为28x28格式
#     plt.imshow(im)
#     plt.pause(0.1)  # 暂停时间
# plt.imshow(im)
# plt.show()

# np.savetxt('train_img.csv', X_Train,
#            fmt = '%i', delimiter = ',')
# np.savetxt('train_labels.csv', Y_Train,
#            fmt = '%i', delimiter = ',')
# np.savetxt('test_img.csv', X_Test,
#            fmt = '%i', delimiter = ',')
# np.savetxt('test_labels.csv', Y_Test,
#            fmt = '%i', delimiter = ',')
# X_train = np.genfromtxt('train_img.csv',
#                         dtype=int, delimiter=',')
# y_train = np.genfromtxt('train_labels.csv',
#                         dtype=int, delimiter=',')
# X_test = np.genfromtxt('test_img.csv',
#                        dtype=int, delimiter=',')
# y_test = np.genfromtxt('test_labels.csv',
#                        dtype=int, delimiter=',')
