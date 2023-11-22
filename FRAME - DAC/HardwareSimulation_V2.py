import pickle

import numpy as np

from GetDataset import GetDataset

np.random.seed(10)


class HardwareSimulation:
    def __init__(self, weights, biases):
        self.Wmin = -0.4
        self.Wmax = 0.4
        self.Bmin = -3.3
        self.Bmax = 3.3
        self.N_list = None
        self.K = 1.380649e-23
        self.T = 300
        self.delta_f = 1e10
        self.N = 4 * self.K * self.T * self.delta_f
        self.resistanceOn = 240e3
        self.resistanceOff = 240e3 * 100
        self.Gmax = 1 / self.resistanceOn
        self.Gmin = 1 / self.resistanceOff
        # self.Vr_list = [0.06, 0.055, 0.05]
        self.Vr_list = [0.13, 0.09, 0.05]
        self.R_TIA = 100e3
        self.Wmin = -1
        self.Wmax = 1
        self.Vth = 0.17
        self.Weights = weights
        self.biases = biases
        self.Gref = (self.Wmax * self.Gmin - self.Wmin * self.Gmax) / (self.Wmax - self.Wmin)
        self.A = []
        self.G0 = (self.Gmax - self.Gmin) / (self.Wmax - self.Wmin)
        self.G0_list = [self.G0, self.G0, self.G0]
        self.B0 = (self.Gmax - self.Gmin) / (self.Bmax - self.Bmin)
        self.B0_list = [self.B0, self.B0, self.B0]
        self.Gref_W = (self.Wmax * self.Gmin - self.Wmin * self.Gmax) / (self.Wmax - self.Wmin)
        self.Gref_list = [self.Gref_W, self.Gref_W, self.Gref_W]
        self.Gref_B = (self.Bmax * self.Gmin - self.Bmin * self.Gmax) / (self.Bmax - self.Bmin)
        self.Bref_list = [self.Gref_B, self.Gref_B, self.Gref_B]
        self.RRAM_Array_Biases = None
        self.RRAM_Array_Weights = None
        self.Bref_Array = None
        self.Gref_Array = None
        self.Iref = None
        self.Iref_noise = None
        self.total_Iref = None
        self.getRefArray()
        # self.printA()
        self.WeightToConductance()
    
    def printA(self):
        for i, (weight, Vr) in enumerate(zip(self.Weights, self.Vr_list)):
            a = Vr * self.G0_list[i] / np.sqrt(2 * weight.shape[0] * self.N * self.Gref_list[i])
            # print("a = ", a)
            self.A.append(a)
    
    def getInput(self, data):
        threshold = 0.5
        for array in data:
            array[array >= threshold] = 1.0
            array[array < threshold] = 0.0
        return data
    
    def getRefArray(self):
        RefArray = []
        for arr in self.Weights:
            RefArray.append(arr.shape[0])
        self.Gref_Array = [np.ones((shape, 1)) * self.Gref_list[i] for i, shape in enumerate(RefArray)]
        B_RefArray = []
        for arr in self.biases:
            B_RefArray.append(arr.shape[0])
        # self.Bref_Array = [np.ones((1, shape)) * self.Gref_list[i] for i, shape in enumerate(B_RefArray)]
        self.Bref_Array = [np.ones((1, shape)) * self.Bref_list[i] for i, shape in enumerate(B_RefArray)]

    def WeightToConductance(self, ):
        Temp_biases = self.biases.copy()
        for i in range(len(Temp_biases)):
            Temp_biases[i] = Temp_biases[i] * self.B0_list[i] + self.Bref_list[i]
            # Temp_biases[i] = Temp_biases[i] * self.G0_list[i] + self.Gref_list[i]
        _biases = [np.clip(array, self.Gmin, self.Gmax) for array in Temp_biases]
        self.RRAM_Array_Biases = list(_biases)
        Temp_Weights = self.Weights.copy()
        for i in range(len(Temp_Weights)):
            Temp_Weights[i] = Temp_Weights[i] * self.G0_list[i] + self.Gref_list[i]
        temp_weights = [np.clip(array, self.Gmin, self.Gmax) for array in Temp_Weights]
        self.RRAM_Array_Weights = list(temp_weights)
        print("Weights To RRAM Conductance Complete...")
    
    def SoftMax(self, I, V_activation, G_Array, Gref):
        yb_sum = np.zeros_like(I)
        for i in range(I.shape[0]):
            yb = []
            y = 0
            num_samples = 0
            while y == 0 and num_samples < 1000:
                yb[:] = np.zeros_like(yb)
                # Array_noise = G_Array.copy()
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
        refArray_noise = np.random.normal(0, np.sqrt(self.N * Gref), Gref.shape)
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
        for i, (G_Array, Gref, Vr, G_bia, Bref) in enumerate(
                zip(self.RRAM_Array_Weights[:-1], self.Gref_Array[:-1], self.Vr_list[:-1], self.RRAM_Array_Biases[:-1],
                    self.Bref_Array[:-1])):
            V_act = np.array(activations[-1]) * Vr
            I = np.dot(V_act, G_Array) + np.dot(Vr, G_bia) - Vr * Bref
            activation = self.Sigmoid(I, V_act, G_Array, Gref)
            activations.append(activation)
        V_last_act = np.array(activations[-1]) * self.Vr_list[-1]
        I = np.dot(V_last_act, self.RRAM_Array_Weights[-1]) + np.dot(self.Vr_list[-1], self.RRAM_Array_Biases[-1]) - \
            self.Vr_list[-1] * self.Bref_Array[-1]
        activation = self.SoftMax(I, V_last_act, self.RRAM_Array_Weights[-1], self.Gref_Array[-1])
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
Test_times = 300
filename = "./weights_biases.pickle"
with open(filename, 'rb') as file:
    (weights, biases) = pickle.load(file)

# SoftwareInference = NeuralNetworkTest(weights, biases)
# test_accuracy_software = SoftwareInference.test_binary(Test_times, test_images, test_labels)
# print("Test Accuracy: {:.2f}    ".format(test_accuracy_software))

# loaded_list.append(np.array(current_array))
UsingHardwareInference = HardwareSimulation(weights, biases)
# test_accuracy_hardware = UsingHardwareInference.Test_Hardware(Test_times, test_images[84:85], test_labels[84:85])
test_accuracy_hardware = UsingHardwareInference.Test_Hardware(Test_times, test_images[:1], test_labels[:1])
# Test_Accuracy_hardware.append(test_accuracy_hardware)
print("Test Accuracy: {:.2f}".format(test_accuracy_hardware))
