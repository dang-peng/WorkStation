import os
import struct

import numpy as np


class DatasetLoader:
    def __init__(self, path, kind):
        self.path = path
        self.kind = kind
        self.raw_images, self.raw_labels = self.load_mnist()
    
    def load_mnist(self):
        """Load MNIST data from `path`"""
        # use in windows
        labels_path = os.path.join(self.path, '%s-labels.idx1-ubyte/%s-labels.idx1-ubyte' % (self.kind, self.kind))
        images_path = os.path.join(self.path, '%s-images.idx3-ubyte/%s-images.idx3-ubyte' % (self.kind, self.kind))
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


class DatasetProcess:
    def __init__(self, images, labels, batch_size):
        self.raw_images = images
        self.raw_labels = labels
        self.batch_size = batch_size
        self.num_samples = images.shape[0]
        self.num_batches = self.num_samples // batch_size
        self.preprocess_data()
        self.images, self.labels = self.data_batch()
    
    def one_hot(self):
        num_classes = 10
        return np.eye(num_classes)[self.raw_labels]
    
    def preprocess_data(self):
        self.raw_images = self.normalize_images()
        self.raw_labels = self.raw_labels.astype(np.int64)
        self.raw_labels = self.one_hot()
    
    def normalize_images(self):
        return self.raw_images.astype(np.float32) / 255.0
    
    def astype(self, float32):
        pass
    
    def data_batch(self):
        data_iterator = self.split_batch()
        _images = []
        _labels = []
        for images_batch, labels_batch in data_iterator:
            _images.append(images_batch)
            _labels.append(labels_batch)
        return _images, _labels
    
    def split_batch(self):
        num_samples = self.num_batches * self.batch_size
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        for i in range(0, num_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            images_batch = self.raw_images[batch_indices]
            labels_batch = self.raw_labels[batch_indices]
            yield images_batch, labels_batch


class GetDataset(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        # self.train_images, self.train_labels, self.test_images, self.test_labels = self.get_data()
    
    def get_data(self):
        # load dataset
        path = "f:/Code/practice/Learning/Python/dataset/"
        kind_train = 'train'
        kind_test = 't10k'
        train_raw_data = DatasetLoader(path, kind_train)
        test_raw_data = DatasetLoader(path, kind_test)
        print('Train_images: ', train_raw_data.raw_images.shape)
        print('Train_labels: ', train_raw_data.raw_labels.shape)
        print('Test_images:  ', test_raw_data.raw_images.shape)
        print('Test_labels:  ', test_raw_data.raw_labels.shape)
        # preprocess dataset
        Train_data = DatasetProcess(train_raw_data.raw_images, train_raw_data.raw_labels, self.batch_size)
        Test_data = DatasetProcess(test_raw_data.raw_images, test_raw_data.raw_labels, self.batch_size)
        train_images = Train_data.images
        train_labels = Train_data.labels
        test_images = Test_data.images
        test_labels = Test_data.labels
        return train_images, train_labels, test_images, test_labels
