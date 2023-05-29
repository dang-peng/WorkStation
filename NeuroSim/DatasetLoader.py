import os
import struct

import numpy as np


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
        magic, n = struct.unpack('>II', dbpath.read(8))
        labels = np.fromfile(dbpath, dtype = np.uint8)
    
    with open(images_path, 'rb') as imgpath:
        print("Extracting  %s" % images_path)
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype = np.uint8).reshape(len(labels), 784)
    
    return images, labels


class DatasetLoader:
    def __init__(self, images, labels, batch_size):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        # self.batch_images = None
        # self.batch_labels = None
        self.num_samples = images.shape[0]
        self.num_batches = self.num_samples // batch_size
        self.current_batch_index = 0
        self.preprocess_data()
        self.data_loader = self.create_data_loader()
    
    def preprocess_data(self):
        self.images = self.normalize_images()
        self.labels = self.labels.astype(np.int64)
        # return self.images, self.labels
    
    def normalize_images(self):
        return self.images.astype(np.float32) / 255.0
    
    def astype(self, float32):
        pass
    
    def create_data_loader(self):
        num_samples = len(self.images)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        for i in range(0, num_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_images = self.images[batch_indices]
            batch_labels = self.labels[batch_indices]
            yield batch_images, batch_labels


(Train_images, Train_Y) = load_mnist("f:/Code/practice/Learning/Python/dataset/", kind = 'train')
(Test_images, Test_Y) = load_mnist("f:/Code/practice/Learning/Python/dataset/", kind = 't10k')
# 将标签转换为one-hot向量
num_classes = 10
Train_labels = np.eye(num_classes)[Train_Y]
Test_labels = np.eye(num_classes)[Test_Y]

print('Train_images: ', Train_images.shape)
print('Train_labels: ', Train_labels.shape)
print('Test_images:  ', Test_images.shape)
print('Test_labels:  ', Test_labels.shape)
# (Train_images, Train_labels), (Test_images, Test_labels)
# train_loader = DatasetLoader(Train_images, Train_labels, batch_size = 32)
