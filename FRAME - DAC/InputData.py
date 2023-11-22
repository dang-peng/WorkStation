import numpy as np


class InputData:
    def __init__(self):
        self.rows = 100
        self.cols = 784
        self.input = []
        self.data_generator()
        self.print_shape()
    
    def data_generator(self):
        self.input = [np.random.randint(2, size = (self.rows, self.cols)) for _ in range(100)]
    
    def print_shape(self):
        print("input data shape:", self.input[0].shape)
        print("input data length:", len(self.input))


data = InputData()
# data.print_shape()
