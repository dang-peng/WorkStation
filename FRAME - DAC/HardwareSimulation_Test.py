import pickle

import numpy as np

from GetDataset import GetDataset

np.random.seed(10)


class HardwareSimulation:
    def __init__(self, weights, biases):
        self.G_biasesArray = None
        self.BiasesArray = None
        self.K = 1.380649e-23
        self.T = 300
        self.delta_f = 1e10
        self.N = 4 * self.K * self.T * self.delta_f
        self.resistanceOn = 240e3
        self.resistanceOff = 240e3 * 100
        self.Gmax = 1 / self.resistanceOn
        self.Gmin = 1 / self.resistanceOff
        # self.Vr_list = [0.16, 0.13, 0.01]
        self.Vr_list = [0.13, 0.09, 0.1]
        # self.Gmax = 100e-6
        # self.Gmin = 1e-6
        self.R_TIA = 100e3
        self.Wmin = -1
        self.Wmax = 1
        self.Vth = 0.1
        self.Weights = weights
        self.biases = biases
        self.Iref = None
        self.Iref_noise = None
        self.total_Iref = None
        self.Scale_factor = (self.Gmax - self.Gmin) / (self.Wmax - self.Wmin)
        self.Gref = (self.Wmax * self.Gmin - self.Wmin * self.Gmax) / (self.Wmax - self.Wmin)
        self.G0 = (self.Gmax - self.Gmin) / (self.Wmax - self.Wmin)
        self.A = []
        self.RRAM_Array = None
        self.RefArray = None
        self.G_refArray = None
        self.getRefArray()
        self.printA()
        self.WeightToConductance()
        # self.getBiasesArray()
    
    def printA(self):
        for weight, Vr in zip(self.Weights, self.Vr_list):
            a = Vr * self.G0 / np.sqrt(2 * weight.shape[0] * self.N * self.Gref)
            print("a = ", a)
            self.A.append(a)
    
    def getInput(self, data):
        threshold = 0.5
        for array in data:
            array[array >= threshold] = 1.0
            array[array < threshold] = 0.0
        return data
    
    def getRefArray(self):
        self.RefArray = []
        for arr in self.Weights:
            self.RefArray.append(arr.shape[0])
        self.G_refArray = [np.ones((shape, 1)) * self.Gref for shape in self.RefArray]
    
    # def getBiasesArray(self):
    #     self.BiasesArray = []
    #     for bia in self.biases:
    #         self.BiasesArray.append(bia.shape[1])
    # self.biasesArray = [np.ones((1, element)) * self.Gref for element in self.BiasesArray]
    
    def WeightToConductance(self, ):
        Temp_biases = self.biases.copy()
        for i in range(len(Temp_biases)):
            Temp_biases[i] = Temp_biases[i] * self.Scale_factor + self.Gref
        _biases = [np.clip(array, self.Gmin, self.Gmax) for array in Temp_biases]
        self.G_biasesArray = list(_biases)
        Temp_Weights = self.Weights.copy()
        _Weights = np.array(Temp_Weights, dtype = object) * self.Scale_factor + self.Gref
        temp_weights = [np.clip(array, self.Gmin, self.Gmax) for array in _Weights]
        self.RRAM_Array = list(temp_weights)
        print("Weights To RRAM Conductance Complete...")
    
    def SoftMax(self, I, V_activation, G_Array, Gref):
        yb_sum = np.zeros_like(I)
        for i in range(I.shape[0]):
            yb = []
            y = 0
            num_samples = 0
            while y == 0 and num_samples < 1000:
                yb[:] = np.zeros_like(yb)
                Array_noise = G_Array.copy()
                # for m in range(Array_noise.shape[0]):
                #     for n in range(Array_noise.shape[1]):
                #         Array_noise[m, n] = np.random.normal(0, np.sqrt(self.N * Array_noise[m, n]), (1, 1))
                Array_noise = np.random.normal(0, np.sqrt(self.N * Gref), size = G_Array.shape)
                I_noise = np.sum(Array_noise, axis = 0)
                I_total = I[i] + np.array(I_noise)
                Iref_total = self.Computing_Iref(V_activation, Gref)
                V = self.R_TIA * (I_total - Iref_total[i])
                yb = np.where(V > self.Vth, 1, 0)
                y = np.sum(yb, axis = 0)
                num_samples += 1
            yb_sum[i] = np.array(yb)
            # print("num_samples:", num_samples)
        return yb_sum
    
    def Computing_Iref(self, Vref, Gref):
        Iref = np.dot(Vref, Gref)
        # refArray_noise = Gref.copy()
        # for i in range(len(refArray_noise)):
        #     refArray_noise[i] = np.random.normal(0, np.sqrt(self.N * self.Gref), (1, 1))
        refArray_noise = np.random.normal(0, np.sqrt(self.N * self.Gref), Gref.shape)
        Iref_noise = np.sum(refArray_noise, axis = 0)
        total_Iref = Iref_noise + Iref
        return total_Iref
    
    def Sigmoid(self, I, V_activation, G_Array, Gref):
        Array_noise = G_Array.copy()
        # for i in range(Array_noise.shape[0]):
        #     for j in range(Array_noise.shape[1]):
        #         Array_noise[i, j] = np.random.normal(0, np.sqrt(self.N * Array_noise[i, j]), (1, 1))
        Array_noise = np.random.normal(0, np.sqrt(self.N * Gref), Array_noise.shape)
        Iref_total = self.Computing_Iref(V_activation, Gref)
        I_noise = np.sum(Array_noise, axis = 0)
        I_total = I + I_noise
        yb = np.where(I_total > Iref_total, 1, 0)
        return yb
    
    def Forward_Hardware(self, x):
        activations = [x]
        for i, (G_Array, Gref, Vr, G_bia) in enumerate(
                zip(self.RRAM_Array[:-1], self.G_refArray[:-1], self.Vr_list[:-1], self.G_biasesArray[:-1])):
            V_act = np.array(activations[-1]) * Vr
            I = np.dot(V_act, G_Array) + np.dot(Vr, G_bia) - Vr * self.Gref
            activation = self.Sigmoid(I, V_act, G_Array, Gref)
            activations.append(activation)
        V_last_act = np.array(activations[-1]) * self.Vr_list[-1]
        I = np.dot(V_last_act, self.RRAM_Array[-1]) + np.dot(self.Vr_list[-1], self.G_biasesArray[-1]) - self.Vr_list[
            -1] * self.Gref
        activation = self.SoftMax(I, V_last_act, self.RRAM_Array[-1], self.G_refArray[-1])
        activations.append(activation)
        return activations[-1]
    
    def Test_Hardware(self, times, test_images, test_labels):
        images = self.getInput(test_images)
        M, N, Z = np.array(test_labels).shape
        test_results = np.zeros((M * N, Z))
        accuracy = 0
        print("Start Testing...")
        for i in range(times):
            corrects = 0
            total = 0
            y_hat = []
            labels = []
            y_pred = []
            true_labels = []
            for _images_batch, _labels_batch in zip(images, test_labels):
                y_forward = self.Forward_Hardware(_images_batch)
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
            print("Times: {}  Test Accuracy = {:.2f}".format(i + 1, accuracy))
        return accuracy


batch_size = 100
train_images, train_labels, test_images, test_labels, batches = GetDataset(batch_size).get_data()
Test_times = 100
filename = "./weights_biases.pickle"
with open(filename, 'rb') as file:
    (weights, biases) = pickle.load(file)

# SoftwareInference = NeuralNetworkTest(weights, biases)
# test_accuracy_software = SoftwareInference.test_binary(Test_times, test_images, test_labels)
# print("Test Accuracy: {:.2f}    ".format(test_accuracy_software))

# loaded_list.append(np.array(current_array))
UsingHardwareInference = HardwareSimulation(weights, biases)
test_accuracy_hardware = UsingHardwareInference.Test_Hardware(Test_times, test_images[:10], test_labels[:10])
# Test_Accuracy_hardware.append(test_accuracy_hardware)
print("Test Accuracy: {:.2f}".format(test_accuracy_hardware))
