import datetime
import os

import numpy as np
from matplotlib import pyplot as plt


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis = 1, keepdims = True)


# 10 20 70 100
np.random.seed(1)
rows = 100
cols = 1
K = 1.380649e-23
T = 300
delta_f = 1e10
Vr = 0.05
N1 = 10 * 4 * K * T * delta_f

# Vth = 0.58
Vth = 0.05

R_TIA = 100e3
resistanceOn = 240e3
resistanceOff = 240e3 * 100
Gmax = 1 / resistanceOn
Gmin = 1 / resistanceOff
Weights = np.random.normal(0, 1, (rows, 10))
Wmax = np.max(Weights)
Wmin = np.min(Weights)
Gref = (Wmax * Gmin - Wmin * Gmax) / (Wmax - Wmin)
G0 = (Gmax - Gmin) / (Wmax - Wmin)
a1 = Vr * G0 / np.sqrt(2 * N1 * 100 * Gref)
print("a1 = ", a1)

num_experiment = 101
X = np.random.randint(2, size = (rows, cols))
V = X * Vr
# Weights = np.random.normal(0, 1, (rows, 10))
G_Array = Weights * G0 + Gref
G_Array = np.clip(G_Array, Gmin, Gmax)
I0 = np.dot(V.T, G_Array)
Gref_Col = np.ones((rows, 1)) * Gref
Iref = np.dot(V.T, Gref_Col)
yb_sum = np.zeros((1, 10))
yb = []
Num_Sampling_list = []
yb_list = []
V_list = []
Vref_list = []
Vth_list = []
for j in range(num_experiment):
    y = 0
    num_samples = 0
    Vref = 0
    while y == 0 and num_samples < 1000:
        yb[:] = np.zeros_like(yb)
        Iref_device_noise = G_Array.copy()
        for G_ref in np.nditer(Iref_device_noise, op_flags = ['readwrite']):
            deviceRef_noise = np.random.normal(0, np.sqrt(N1 * G_ref), (1, 1))
            G_ref[...] = deviceRef_noise
        Iref_noise = np.sum(Iref_device_noise, axis = 0)
        Array_noise = G_Array.copy()
        for G_device in np.nditer(Array_noise, op_flags = ['readwrite']):
            device_I_noise = np.random.normal(0, np.sqrt(N1 * G_device), (1, 1))
            G_device[...] = device_I_noise
        I_noise = np.sum(Array_noise, axis = 0)
        I_total = I0 + I_noise
        V = R_TIA * I_total
        Vref = R_TIA * (Iref + Iref_noise)
        yb = np.where((V - Vref) > Vth, 1, 0)
        y = np.sum(yb, axis = 1)
        num_samples += 1
    V_list.append(V[0][0])
    Vref_list.append(Vref[0][0])
    yb_sum = yb_sum + np.array(yb)
    yb_list.append(yb)
    Vth_list.append(Vth)
    Num_Sampling_list.append(j)
    print("num_samples:", num_samples)
# 第一个子图
plt.subplot(4, 1, 1)
plt.plot(Num_Sampling_list, V_list, '#EC7728', linewidth = 2, label = 'V')
# plt.plot(Num_Sampling_list, V_list, 'r-', linewidth = 2, label = 'V')
plt.ylabel('V, Vref [V]', fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.yticks(fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.legend(prop = {'family': 'Times New Roman'}, loc = 'lower left')
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.axhline(0, color = 'gray', linestyle = '--', linewidth = 0.5)
plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
# plt.autoscale(enable = True, axis = 'y', tight = True)  # 自动调整y轴范围
plt.plot(Num_Sampling_list, Vref_list, '#2CA02C', linewidth = 2, label = 'Vref')
plt.xlim(0, 100)
# plt.ylim(0.7, 0.85)
plt.xlabel('')
plt.yticks(fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.legend(prop = {'family': 'Times New Roman'}, loc = 'lower left')
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.axhline(0, color = 'gray', linestyle = '--', linewidth = 0.5)
plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.gca().yaxis.set_label_coords(-0.11, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
# plt.autoscale(enable = True, axis = 'y', tight = True)  # 自动调整y轴范围
# 第二个子图
plt.subplot(4, 1, 2)
# plt.plot(Num_Sampling_list, np.array(V_list) - np.array(Vref_list), '#2CA02C', linewidth = 2, label = 'Vref')
plt.plot(Num_Sampling_list, np.array(V_list) - np.array(Vref_list), '#F8766D', linewidth = 2, label = 'V - Vref')
plt.xlim(0, 100)
# plt.ylim(0.7, 0.85)
plt.xlabel('')
plt.ylabel('V - Vref [V]', fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.yticks(fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.legend(prop = {'family': 'Times New Roman'}, loc = 'lower left')
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.axhline(0, color = 'gray', linestyle = '--', linewidth = 0.5)
plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.gca().yaxis.set_label_coords(-0.11, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
# plt.autoscale(enable = True, axis = 'y', tight = True)  # 自动调整y轴范围
# 第三个子图
plt.subplot(4, 1, 3)
# plt.plot(Num_Sampling_list, np.array(V_list) - np.array(Vref_list), '#2CA02C', linewidth = 2, label = 'Vref')
plt.plot(Num_Sampling_list, Vth_list, '#479CD1', linewidth = 2, label = 'Vth')
plt.xlim(0, 100)
plt.ylim(0.03, 0.06)
plt.xlabel('')
plt.ylabel('Vth', fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.yticks(fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.legend(prop = {'family': 'Times New Roman'}, loc = 'lower left')
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.axhline(0, color = 'gray', linestyle = '--', linewidth = 0.5)
plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.gca().yaxis.set_label_coords(-0.11, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
# 第四个子图
# non_zero_indexes = np.nonzero(yb_list)
# print(yb_list)
# Num_Sampling_non_zero = np.take(Num_Sampling_list, non_zero_indexes)
# yb_non_zero = np.take(yb_list, non_zero_indexes)
plt.subplot(4, 1, 4)
plt.xlabel('')
plt.scatter(Num_Sampling_list, yb_list, color = '#1F77B4', s = 30, label = 'P = ')
plt.xlabel('Number of Experiments', fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.ylabel('y_b', fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.yticks(fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.legend(prop = {'family': 'Times New Roman'}, loc = 'lower left')
plt.xticks(fontname = "Times New Roman", fontsize = 14)
plt.yticks(fontname = "Times New Roman", fontsize = 14)
plt.xlim(0, 100)
# plt.ylim(0, 1)
# plt.autoscale(enable = True, axis = 'y', tight = True)
# plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.gca().yaxis.set_label_coords(-0.11, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)

# plt.tight_layout()
folder_path = "data"
os.makedirs(folder_path, exist_ok = True)
current_time = datetime.datetime.now().strftime("%Y_%m%d_%H%M")
filename = f"softmax_Sampling_{current_time}.svg"
file_path = os.path.join(folder_path, filename)
plt.savefig(file_path, format = 'svg')
plt.show()
