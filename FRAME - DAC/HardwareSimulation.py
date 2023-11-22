import numpy as np
import pickle
from GetDataset import GetDataset

np.random.seed(10)


class HardwareSimulation:
    def __init__(self, weights):
        self.K = 1.380649e-23
        self.T = 300
        self.delta_f = 1e10
        self.N = 4 * self.K * self.T * self.delta_f * 0.01
        self.Gmax = 10e-6
        self.Gmin = 1e-6
        self.R_TIA = 100e3
        self.Wmin = -1
        self.Wmax = 1
        self.Vr = 0.05
        self.Vth = 3.9
        self.input_data = None
        self.Weights = weights
        self.Iref = None
        self.Iref_noise = None
        self.total_Iref = None
        self.Scale_factor = (self.Gmax - self.Gmin) / (self.Wmax - self.Wmin)
        self.Gref = (self.Wmax * self.Gmin - self.Wmin * self.Gmax) / (self.Wmax - self.Wmin)
        self.G0 = (self.Gmax - self.Gmin) / (self.Wmax - self.Wmin)
        self.A = self.Vr * self.G0 / np.sqrt(2 * 100 * self.N * self.Gref)
        self.RRAM_Array = None
        self.RefArray = None
        self.GetRefArray()
        self.GrefArray = [np.ones((shape, 1)) * self.Gref for shape in self.RefArray]
        self.printA()
        self.WeightToConductance()
        # self.resistanceOn = 240e3
        # self.resistanceOff = 240e3 * 100
        # self.Gmax = 1 / self.resistanceOn
        # self.Gmin = 1 / self.resistanceOff
        # self.find_max_min()
    
    def printA(self):
        print("a = ", self.A)
    
    def GetInput(self, data):
        threshold = 0.5
        for array in data:
            array[array >= threshold] = 1
            array[array < threshold] = 0
        return data
    
    def GetRefArray(self):
        self.RefArray = []
        for arr in self.Weights:
            self.RefArray.append(arr.shape[0])
    
    def WeightToConductance(self, ):
        Temp_Weights = self.Weights.copy()
        _Weights = np.array(Temp_Weights, dtype = object) * self.Scale_factor + self.Gref
        temp_weights = [np.clip(array, self.Gmin, self.Gmax) for array in _Weights]
        self.RRAM_Array = list(temp_weights)
        print("Weights To RRAM Conductance Complete...")
    
    def SoftMax_Hardware(self, I, V_activation, G_Array, Gref):
        total_Iref = self.GetIref(V_activation, Gref)
        Array_noise = G_Array.copy()
        for i in range(Array_noise.shape[0]):
            for j in range(Array_noise.shape[1]):
                Array_noise[i, j] = np.random.normal(0, np.sqrt(self.N * Array_noise[i, j]), (1, 1))
        I_noise = np.sum(Array_noise, axis = 0)
        I_total = I + np.array(I_noise)
        V = self.R_TIA * I_total
        yb = np.where(V > self.Vth, 1, 0)
        return yb
    
    # def sigmoid(self, z):
    #     return 1.0 / (1.0 + np.exp(-z))
    #
    # def softmax(self, z):
    #     return np.exp(z) / np.sum(np.exp(z), axis = 1, keepdims = True)
    
    def SoftMax(self, I, V_activation, G_Array, Gref):
        num_experiment = 100
        yb_sum = np.zeros_like(I)
        # y = np.zeros((I.shape[0]))
        for i in range(I.shape[0]):
            yb = []
            for j in range(num_experiment):
                y = 0
                num_samples = 0
                while y == 0 and num_samples < 1000:
                    yb[:] = np.zeros_like(yb)
                    Array_noise = G_Array.copy()
                    for m in range(Array_noise.shape[0]):
                        for n in range(Array_noise.shape[1]):
                            Array_noise[m, n] = np.random.normal(0, np.sqrt(self.N * Array_noise[m, n]), (1, 1))
                    I_noise = np.sum(Array_noise, axis = 0)
                    # Array_noise = np.random.normal(0, np.sqrt(self.N * G_Array), size = G_Array.shape)
                    I_total = I[i] + np.array(I_noise)
                    V = self.R_TIA * I_total
                    yb = np.where(V > self.Vth, 1, 0)
                    y = np.sum(yb, axis = 0)
                    num_samples += 1
                yb_sum[i] = yb_sum[i] + np.array(yb)
                # print("num_samples:", num_samples
            yb_sum[i] = yb_sum[i] / num_experiment
        return yb_sum
    
    def GetIref(self, Vref, Gref):
        Iref = np.dot(Vref, Gref)
        refArray_noise = Gref.copy()
        for i in range(len(refArray_noise)):
            refArray_noise[i] = np.random.normal(0, np.sqrt(self.N * self.Gref), (1, 1))
        Iref_noise = np.sum(refArray_noise, axis = 0)
        total_Iref = Iref_noise + Iref
        return total_Iref
    
    def Sigmoid(self, I, V_activation, G_Array, Gref):
        num_experiment = 10
        yb_total = np.zeros_like(I)
        for i in range(num_experiment):
            Array_noise = G_Array.copy()
            for i in range(Array_noise.shape[0]):
                for j in range(Array_noise.shape[1]):
                    Array_noise[i, j] = np.random.normal(0, np.sqrt(self.N * Array_noise[i, j]), (1, 1))
            Iref_total = self.GetIref(V_activation, Gref)
            I_noise = np.sum(Array_noise, axis = 0)
            I_total = I + I_noise
            yb = np.where(I_total > Iref_total, 1, 0)
            yb_total = yb_total + yb
            # z = (I - Iref_total) / (self.Vr * self.G0)
            # print("Experiment Number:", i + 1)
        return yb_total / num_experiment
    
    def Forward_Hardware(self, x):
        activations = [x]
        for i, (G_Array, Gref) in enumerate(zip(self.RRAM_Array[:-1], self.GrefArray[:-1])):
            # V_act = np.array(activations[-1]) * self.Vr
            V_act = np.array(activations[-1])
            I = np.dot(V_act, G_Array)
            # activation = self.sigmoid(I)
            activation = self.Sigmoid(I, V_act, G_Array, Gref)
            activations.append(activation)
        V_last_act = np.array(activations[-1]) * self.Vr
        # V_last_act = np.array(activations[-1])
        I = np.dot(V_last_act, self.RRAM_Array[-1])
        # activation = self.softmax(I)
        activation = self.SoftMax(I, V_last_act, self.RRAM_Array[-1], self.GrefArray[-1])
        activations.append(activation)
        return activations[-1]
    
    def Test_Hardware(self, times, test_images, test_labels):
        images = self.GetInput(test_images)
        corrects = 0
        total = 0
        true_labels = []
        M, N, Z = np.array(test_labels).shape
        test_results = np.zeros((M * N, Z))
        print("Start Testing...")
        for i in range(times):
            y_hat = []
            labels = []
            y_pred = []
            for _images_batch, _labels_batch in zip(images, test_labels):
                y_forward = self.Forward_Hardware(_images_batch)
                labels.append(_labels_batch)
                y_hat.append(y_forward)
            # print("{}th Forward Propagation Completed...".format(i + 1))
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
            print("Times: {}  Test Accuracy = {:.2f}".format(i + 1, accuracy))
        return accuracy


batch_size = 100
train_images, train_labels, test_images, test_labels, batches = GetDataset(batch_size).get_data()
Test_times = 10
# file_path = "Weight_data.txt"
filename = "weight_data.txt"
Weights_list = []
current_array = []
with open(filename, "r") as file:
    for line in file:
        if line.strip() == "---":
            Weights_list.append(np.array(current_array))
            current_array = []
        else:
            row = list(map(float, line.strip().split()))
            current_array.append(row)

# loaded_list.append(np.array(current_array))
UsingHardwareInference = HardwareSimulation(Weights_list)
test_accuracy_hardware = UsingHardwareInference.Test_Hardware(Test_times, test_images, test_labels)
# Test_Accuracy_hardware.append(test_accuracy_hardware)
print("Test Accuracy: {:.2f}    ".format(test_accuracy_hardware))
