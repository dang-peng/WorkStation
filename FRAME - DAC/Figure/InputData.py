import numpy as np

# 10 20 70 100
np.random.seed(10)


class InputData:
    def __init__(self):
        self.rows = 100
        self.cols = 1
        self.input = []
        self.weights = []
        self.data_generator()
        self.print_shape()
    
    def data_generator(self):
        self.weights = np.random.normal(0, 1, (100, 10))
        # self.input = [np.random.randint(2, size = (self.rows, self.cols)) for _ in range(100)]
        self.input = np.random.randint(2, size = (self.rows, self.cols))
    
    def print_shape(self):
        print("input data shape:", self.input[0].shape)
        print("input data length:", len(self.input))


data = InputData()
# data.print_shape()
