import datetime
import os
import pickle

import numpy as np
from matplotlib import pyplot as plt

from GetDataset import GetDataset


# np.random.seed(0)


def softmax_binary(z):
    y_active = np.exp(z) / np.sum(np.exp(z), axis = 1, keepdims = True)
    shape = y_active.shape
    binary_y = np.random.rand(*shape)
    new_y = binary_matrix_conversion(y_active, binary_y)
    return new_y


def binary_matrix_conversion(activations, random_matrix):
    return (activations > random_matrix).astype('float')


def sigmoid_binary(z):
    y_active = 1.0 / (1.0 + np.exp(-z))
    shape = y_active.shape
    binary_y = np.random.rand(*shape)
    new_y = binary_matrix_conversion(y_active, binary_y)
    return new_y


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis = 1, keepdims = True)


class SoftwareSimulation:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases
    
    def forward(self, x):
        activations = [x]
        layer_input = []
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(activations[-1], weight) + bias
            layer_input.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        layer_input.append(z)
        activation = softmax(z)
        activations.append(activation)
        return activations[-1]
    
    def forward_binary(self, x):
        activations = [x]
        layer_input = []
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(activations[-1], weight) + bias
            layer_input.append(z)
            activation = sigmoid_binary(z)
            activations.append(activation)
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        layer_input.append(z)
        activation = softmax_binary(z)
        activations.append(activation)
        return activations[-1]
    
    def test_binary(self, times, test_images, test_labels):
        true_labels = []
        accuracy_list = []
        M, N, Z = np.array(test_labels).shape
        test_results = np.zeros((M * N, Z))
        for i in range(times):
            y_hat = []
            labels = []
            y_pred = []
            for _images_batch, _labels_batch in zip(test_images, test_labels):
                y_forward = self.forward_binary(_images_batch)
                labels.append(_labels_batch)
                y_hat.append(y_forward)
            for sublist1 in y_hat:
                for sublist2 in sublist1:
                    y_pred.append(sublist2)
            for sublist1 in labels:
                for sublist2 in sublist1:
                    true_labels.append(sublist2)
            test_results = test_results + np.array(y_pred)
            corrects = 0
            total = 0
            for predict_label, true_label in zip(test_results, true_labels):
                total += 1
                # print(np.argmax(predict_label), np.argmax(true_label))
                if np.argmax(predict_label) == np.argmax(true_label):
                    corrects += 1
            accuracy = 100 * corrects / total
            accuracy_list.append(accuracy)
            print("Times: {}  Using Hardware Inference Test Accuracy: {:.2f}".format(i + 1, accuracy))
        # print("Epoch: {}  Test Accuracy = {}/{} = {:.2f}".format(epoch + 1, corrects, len(total), accuracy))
        return accuracy_list
    
    def test(self, test_images, test_labels):
        total = 0
        y_hat = []
        labels = []
        corrects = 0
        y_pred = []
        true_labels = []
        for _images_batch, _labels_batch in zip(test_images, test_labels):
            y_forward = self.forward(_images_batch)
            labels.append(_labels_batch)
            y_hat.append(y_forward)
        for sublist1 in y_hat:
            for sublist2 in sublist1:
                y_pred.append(sublist2)
        for sublist1 in labels:
            for sublist2 in sublist1:
                true_labels.append(sublist2)
        for predict_label, true_label in zip(y_pred, true_labels):
            total += 1
            # print(np.argmax(predict_label), np.argmax(true_label))
            if np.argmax(predict_label) == np.argmax(true_label):
                corrects += 1
        accuracy = 100 * corrects / total
        # print("Epoch: {}  Test Accuracy = {}/{} = {:.2f}".format(epoch + 1, corrects, len(total), accuracy))
        return accuracy


class HardwareSimulation:
    def __init__(self, Vr, weights, biases):
        self.Wmin_1 = -0.4
        self.Wmax_1 = 0.4
        self.Wmin_2 = -0.5
        self.Wmax_2 = 0.5
        self.Wmin_3 = -2
        self.Wmax_3 = 2
        self.G_biases_refArray = None
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
        # self.Vr_list = [0.16, 0.13, 0.05]
        self.Vr_list = [0.05]
        self.R_TIA = 100e3
        self.Wmin = -1
        self.Wmax = 1
        # self.Vth = Vth
        self.Vth = 0.1739
        self.Weights = weights
        self.biases = biases
        self.Iref = None
        self.Iref_noise = None
        self.total_Iref = None
        self.Scale_factor = (self.Gmax - self.Gmin) / (self.Wmax - self.Wmin)
        self.Gref = (self.Wmax * self.Gmin - self.Wmin * self.Gmax) / (self.Wmax - self.Wmin)
        self.G0 = (self.Gmax - self.Gmin) / (self.Wmax - self.Wmin)
        self.A = []
        self.Gref_1 = (self.Wmax_1 * self.Gmin - self.Wmin_1 * self.Gmax) / (self.Wmax_1 - self.Wmin_1)
        self.Gref_2 = (self.Wmax_2 * self.Gmin - self.Wmin_2 * self.Gmax) / (self.Wmax_2 - self.Wmin_2)
        self.Gref_3 = (self.Wmax_3 * self.Gmin - self.Wmin_3 * self.Gmax) / (self.Wmax_3 - self.Wmin_3)
        self.Gref_list = [self.Gref_1, self.Gref_2, self.Gref_3]
        self.G0_1 = (self.Gmax - self.Gmin) / (self.Wmax_1 - self.Wmin_1)
        self.G0_2 = (self.Gmax - self.Gmin) / (self.Wmax_2 - self.Wmin_2)
        self.G0_3 = (self.Gmax - self.Gmin) / (self.Wmax_3 - self.Wmin_3)
        self.G0_list = [self.G0_1, self.G0_2, self.G0_3]
        self.RRAM_Array = None
        self.RefArray = None
        self.G_refArray = None
        self.getRefArray()
        self.tuning_Vr(Vr)
        self.printA()
        self.WeightToConductance()
    
    def tuning_Vr(self, Vr):
        # a = 1
        for i, (weight) in enumerate(self.Weights[:-1]):
            # Vr = a * np.sqrt(2 * self.N * weight.shape[0] * self.Gref_list[i]) / self.G0_list[i]
            self.Vr_list.insert(i, Vr)
    
    def printA(self):
        for i, (weight, Vr) in enumerate(zip(self.Weights, self.Vr_list)):
            a = Vr * self.G0_list[i] / np.sqrt(2 * weight.shape[0] * self.N * self.Gref_list[i])
            # print("a{} = ".format(i), a)
            # if i == 2:
            #     a = 0.05 * self.G0_list[i] / np.sqrt(2 * self.N * weight.shape[0] * self.Gref_list[i])
            #     print("i = {} a{} = ".format(i, i), a)
            self.A.append(a)
    
    @staticmethod
    def getInput(data):
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
    
    def WeightToConductance(self, ):
        Temp_biases = self.biases.copy()
        for i in range(len(Temp_biases)):
            Temp_biases[i] = Temp_biases[i] * self.G0_list[i] + self.Gref_list[i]
        _biases = [np.clip(array, self.Gmin, self.Gmax) for array in Temp_biases]
        self.G_biasesArray = list(_biases)
        Temp_biases_ref = self.biases.copy()
        for i in range(len(Temp_biases_ref)):
            Temp_biases_ref[i] = Temp_biases_ref[i] * self.G0_list[i] + self.Gref_list[i]
        _biases_ref = [np.clip(array, self.Gmin, self.Gmax) for array in Temp_biases_ref]
        self.G_biases_refArray = list(_biases_ref)
        Temp_Weights = self.Weights.copy()
        for i in range(len(Temp_Weights)):
            Temp_Weights[i] = Temp_Weights[i] * self.G0_list[i] + self.Gref_list[i]
        temp_weights = [np.clip(array, self.Gmin, self.Gmax) for array in Temp_Weights]
        self.RRAM_Array = list(temp_weights)
        # print("Weights To RRAM Conductance Complete...")
    
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
        accuracy_list = []
        # print("Start Testing...")
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
            accuracy_list.append(accuracy)
            print("Times: {}  Using Hardware Inference Test Accuracy: {:.2f}  Vth = {} V".format(i + 1, accuracy,
                                                                                                 self.Vth))
            # print("Times: {}  Test Accuracy = {:.2f}".format(i + 1, accuracy))
        return accuracy_list


batch_size = 100
train_images, train_labels, test_images, test_labels, batches = GetDataset(batch_size).get_data()
Test_times = 101
filename = "./weights_biases.pickle"
with open(filename, 'rb') as file:
    (weights, biases) = pickle.load(file)

TestNumber = []
SoftwareInferenceTest = []
SoftwareInference = SoftwareSimulation(weights, biases)
for i in range(Test_times):
    print("{}th Testing...".format(i))
    TestNumber.append(i)
    test_accuracy_software = SoftwareInference.test(test_images, test_labels)
    SoftwareInferenceTest.append(test_accuracy_software)
    print("Times: {}  Software Inference Test Accuracy: {:.2f}".format(i + 1, test_accuracy_software))

Vr1 = 0.08
Vr2 = 0.20
Vr3 = 0.06
UsingHardwareInferenceVr1 = HardwareSimulation(Vr1, weights, biases)
UsingHardwareInferenceVr2 = HardwareSimulation(Vr2, weights, biases)
UsingHardwareInferenceVr3 = HardwareSimulation(Vr3, weights, biases)
SoftwareBinaryTest = SoftwareInference.test_binary(Test_times, test_images, test_labels)
UsingHardwareTest1 = UsingHardwareInferenceVr1.Test_Hardware(Test_times, test_images, test_labels)
UsingHardwareTest2 = UsingHardwareInferenceVr2.Test_Hardware(Test_times, test_images, test_labels)
UsingHardwareTest3 = UsingHardwareInferenceVr3.Test_Hardware(Test_times, test_images, test_labels)
# SoftwareInferenceTest.append(SoftwareInferenceTest[-1])
# SoftwareInferenceBinaryTest.append(SoftwareInferenceBinaryTest[-1])
# UsingHardwareTest1.append(UsingHardwareTest1[-1])
# UsingHardwareTest2.append(UsingHardwareTest2[-1])
# UsingHardwareTest3.append(UsingHardwareTest3[-1])
filename1 = "./inference_data_X86_Vr.pickle"
with open(filename1, 'wb') as file:
    pickle.dump((TestNumber, SoftwareInferenceTest, SoftwareBinaryTest, UsingHardwareTest1, UsingHardwareTest2,
                 UsingHardwareTest3), file)
plt.plot(TestNumber, SoftwareInferenceTest, linewidth = 3, label = 'Software Inference')
plt.plot(TestNumber, SoftwareBinaryTest, linewidth = 3, label = 'Binary Software Inference')
plt.plot(TestNumber, UsingHardwareTest1, linewidth = 3, label = 'Hardware Inference' + ' Vr = {} V'.format(Vr1))
plt.plot(TestNumber, UsingHardwareTest2, linewidth = 3, label = 'Hardware Inference' + ' Vr = {} V'.format(Vr2))
plt.plot(TestNumber, UsingHardwareTest3, linewidth = 3, label = 'Hardware Inference' + ' Vr = {} V'.format(Vr3))
plt.legend(prop = {'size': 14, 'family': 'Times New Roman'}, loc = 'lower right')
plt.xlabel('Test Times', fontname = "Times New Roman", fontsize = 20, color = 'black')
plt.ylabel('Accuracy', fontname = "Times New Roman", fontsize = 20, color = 'black')
plt.xticks(fontname = "Times New Roman", fontsize = 18)
plt.yticks(fontname = "Times New Roman", fontsize = 18)
plt.xlim(0, Test_times - 1)
plt.subplots_adjust(top = 0.94, bottom = 0.148)
plt.subplots_adjust(left = 0.15, right = 0.93)
# plt.ylim(60, 100)
# plt.title('Test Accuracy')
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
folder_path = "./data"
os.makedirs(folder_path, exist_ok = True)
current_time = datetime.datetime.now().strftime("%Y_%m%d_%H%M")
filename = f"Test_Accuracy_{current_time}.svg"
file_path = os.path.join(folder_path, filename)
plt.savefig(file_path, format = 'svg')
# plt.savefig('output.svg', format = 'svg')
plt.show()
