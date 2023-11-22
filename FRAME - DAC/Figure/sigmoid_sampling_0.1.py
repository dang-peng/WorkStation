import datetime
import os

import numpy as np
from matplotlib import pyplot as plt

from InputData import data

np.random.seed(1)

rows = 100
cols = 1

K = 1.380649e-23
T = 300
delta_f = 1e10
N1 = 4 * K * T * delta_f
N2 = 4 * K * T * delta_f
N3 = 4 * K * T * delta_f
# a1 = 0.39054999012403124
a1 = 0.42
# N1 = 4 * K * T * delta_f * 0.5
# N2 = 4 * K * T * delta_f * 5e-2
# N3 = 4 * K * T * delta_f * 5
resistanceOn = 240e3
resistanceOff = 240e3 * 100
Gmax = 1 / resistanceOn
Gmin = 1 / resistanceOff
Wmax = 1
Wmin = -1
R_TIA = 100e3
Gref = (Wmax * Gmin - Wmin * Gmax) / (Wmax - Wmin)
G0 = (Gmax - Gmin) / (Wmax - Wmin)
# a1 = Vr * G0 / np.sqrt(4 * N1 * rows * Gref)
Vr1 = a1 * np.sqrt(4 * N1 * rows * Gref) / G0
print("a = ", a1)
print("Vr1 = ", Vr1)
Vr2 = 0.2
print("Vr2 = ", Vr2)
Vr3 = 0.03
print("Vr3 = ", Vr3)
Num_Sampling_list = []
Y_list = []
yb_list = []
V_list = []
Vref_list = []

num_experiment = 1000

for n in range(101):
    X = data.input
    V1 = np.array(X) * Vr1
    V2 = np.array(X) * Vr2
    V3 = np.array(X) * Vr3
    # Weights = weights[0]
    Weights = np.random.uniform(-1, 1, size = (rows, cols))
    G_Array = Weights * G0 + Gref
    G_Array = np.clip(G_Array, Gmin, Gmax)
    Gref_Col = np.ones((rows, 1)) * Gref
    I1 = np.dot(V1.T, G_Array)
    Iref1 = np.dot(V1.T, Gref_Col)
    I2 = np.dot(V2.T, G_Array)
    Iref2 = np.dot(V2.T, Gref_Col)
    I3 = np.dot(V3.T, G_Array)
    Iref3 = np.dot(V3.T, Gref_Col)
    yb = 0
    for i in range(num_experiment):
        I_devices_noise = G_Array.copy()
        for G_device in np.nditer(I_devices_noise, op_flags = ['readwrite']):
            device_I_noise = np.random.normal(0, np.sqrt(N1 * G_device), (1, 1))
            G_device[...] = device_I_noise
        I_noise = np.sum(I_devices_noise, axis = 0)
        Iref_device_noise = G_Array.copy()
        for G_ref in np.nditer(Iref_device_noise, op_flags = ['readwrite']):
            deviceRef_noise = np.random.normal(0, np.sqrt(N1 * G_ref), (1, 1))
            G_ref[...] = deviceRef_noise
        Iref_noise = np.sum(Iref_device_noise, axis = 0)
        if (I1 + I_noise) > (Iref1 + Iref_noise):
            yb += 1
        V = (I1 + I_noise) * R_TIA
        V_list.append(V[0][0])
        Vref = (Iref1 + Iref_noise) * R_TIA
        Vref_list.append(Vref[0][0])
        Num_Sampling_list.append(i)
    # print("Times: {}  ".format(n))
    if round(yb / num_experiment, 1) == 0.1:
        yb_list.append(0.1)
    else:
        yb_list.append(0)

Vref_list = Vref_list[:len(Vref_list) // 1000]
V_list = V_list[:len(V_list) // 1000]
Num_Sampling_list = Num_Sampling_list[:len(Num_Sampling_list) // 1000]
# 第一个子图
plt.subplot(3, 1, 1)
# plt.plot(Num_Sampling_list, V_list, '#FF7F0E', linewidth = 2, label = 'V')
plt.plot(Num_Sampling_list, V_list, 'r-', linewidth = 2, label = 'V')
plt.xlim(0, 100)
plt.ylim(0.7, 0.85)
plt.ylabel('V', fontname = "Times New Roman", fontsize = 16, color = 'black')
plt.yticks(fontname = "Times New Roman", fontsize = 16, color = 'black')
plt.legend(prop = {'family': 'Times New Roman'}, loc = 'lower left')
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.axhline(0, color = 'gray', linestyle = '--', linewidth = 0.5)
plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
# plt.autoscale(enable = True, axis = 'y', tight = True)  # 自动调整y轴范围
# 第二个子图
plt.subplot(3, 1, 2)
plt.plot(Num_Sampling_list, Vref_list, '#2CA02C', linewidth = 2, label = 'Vref')
plt.xlim(0, 100)
plt.ylim(0.7, 0.85)
plt.xlabel('')
plt.ylabel('Vref', fontname = "Times New Roman", fontsize = 16, color = 'black')
plt.yticks(fontname = "Times New Roman", fontsize = 16, color = 'black')
plt.legend(prop = {'family': 'Times New Roman'}, loc = 'lower left')
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.axhline(0, color = 'gray', linestyle = '--', linewidth = 0.5)
plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
# plt.autoscale(enable = True, axis = 'y', tight = True)  # 自动调整y轴范围
# 第三个子图
non_zero_indexes = np.nonzero(yb_list)
Num_Sampling_non_zero = np.take(Num_Sampling_list, non_zero_indexes)
yb_non_zero = np.take(yb_list, non_zero_indexes)
plt.subplot(3, 1, 3)
plt.scatter(Num_Sampling_non_zero, yb_non_zero, color = '#1F77B4', s = 30, label = 'P = 0.1')
plt.xlabel('Sampling Times', fontname = "Times New Roman", fontsize = 16, color = 'black')
plt.ylabel('Probability', fontname = "Times New Roman", fontsize = 16, color = 'black')
plt.yticks(fontname = "Times New Roman", fontsize = 16, color = 'black')
plt.legend(prop = {'family': 'Times New Roman'}, loc = 'lower left')
plt.xticks(fontname = "Times New Roman", fontsize = 16)
plt.yticks(fontname = "Times New Roman", fontsize = 16)
plt.xlim(0, 100)
plt.ylim(0, 1)
# plt.autoscale(enable = True, axis = 'y', tight = True)
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.subplots_adjust(top = 0.9, bottom = 0.125)
folder_path = "data"
os.makedirs(folder_path, exist_ok = True)
current_time = datetime.datetime.now().strftime("%Y_%m%d_%H%M")
filename = f"sigmoid_Sample_0.1_{current_time}.svg"
file_path = os.path.join(folder_path, filename)
plt.savefig(file_path, format = 'svg')
plt.show()
