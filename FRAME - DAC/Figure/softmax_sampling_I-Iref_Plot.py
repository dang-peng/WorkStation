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
I_ref = np.dot(V.T, Gref_Col)
yb_sum = np.zeros((1, 10))
yb = []
Num_Sampling_list = []
I0_list = []
I1_list = []
I2_list = []
I3_list = []
I4_list = []
I5_list = []
I6_list = []
I7_list = []
I8_list = []
I9_list = []
Iref0_list = []
Iref1_list = []
Iref2_list = []
Iref3_list = []
Iref4_list = []
Iref5_list = []
Iref6_list = []
Iref7_list = []
Iref8_list = []
Iref9_list = []
yb_list = []
Vref_list = []
Ith_list = []
for j in range(num_experiment):
    y = 0
    num_samples = 0
    Vref = 0
    Iref = 0
    I = 0
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
        I = (I0 + I_noise) * 1e6
        V = R_TIA * (I0 + I_noise)
        Iref = (I_ref + Iref_noise) * 1e6
        Vref = R_TIA * (I_ref + Iref_noise)
        yb = np.where((V - Vref) > Vth, 1, 0)
        y = np.sum(yb, axis = 1)
        num_samples += 1
    Vref_list.append(Vref[0][0])
    yb_sum = yb_sum + np.array(yb)
    # yb = [item for sublist in yb for item in sublist]
    I0_list.append(I[0][0])
    I1_list.append(I[0][1])
    I2_list.append(I[0][2])
    I3_list.append(I[0][3])
    I4_list.append(I[0][4])
    I5_list.append(I[0][5])
    I6_list.append(I[0][6])
    I7_list.append(I[0][7])
    I8_list.append(I[0][8])
    I9_list.append(I[0][9])
    Iref0_list.append(Iref[0][0])
    Iref1_list.append(Iref[0][1])
    Iref2_list.append(Iref[0][2])
    Iref3_list.append(Iref[0][3])
    Iref4_list.append(Iref[0][4])
    Iref5_list.append(Iref[0][5])
    Iref6_list.append(Iref[0][6])
    Iref7_list.append(Iref[0][7])
    Iref8_list.append(Iref[0][8])
    Iref9_list.append(Iref[0][9])
    Ith_list.append(Vth * 1e6 / R_TIA)
    Num_Sampling_list.append(j)
    print("num_samples:", num_samples)
I0 = (np.array(I0_list) - np.array(Iref0_list))
I1 = (np.array(I1_list) - np.array(Iref1_list))
I2 = (np.array(I2_list) - np.array(Iref2_list))
I3 = (np.array(I3_list) - np.array(Iref3_list))
I4 = (np.array(I4_list) - np.array(Iref4_list))
I5 = (np.array(I5_list) - np.array(Iref5_list))
I6 = (np.array(I6_list) - np.array(Iref6_list))
I7 = (np.array(I7_list) - np.array(Iref7_list))
I8 = (np.array(I8_list) - np.array(Iref8_list))
I9 = (np.array(I9_list) - np.array(Iref9_list))

fig = plt.figure()
# 调整图形的尺寸
fig.set_size_inches(12, 10)  # 设置宽度为8英寸，高度为6英寸
plt.subplot(10, 1, 10)
# plt.plot(Num_Sampling_list, Vref_list, '#F8766D', linewidth = 2, label = 'Vref')
plt.plot(Num_Sampling_list, I0, color = '#00A8CC', linewidth = 2, label = 'I0 - Iref')
plt.plot(Num_Sampling_list, Ith_list, color = '#F8766D', linewidth = 2, label = 'Ith')
plt.xlabel('Number of Experiments', fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.ylabel('I0', fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.yticks(fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.legend(prop = {'family': 'Times New Roman'}, loc = 'lower left')
plt.xticks(np.arange(0, 101, 10), fontname = "Times New Roman", fontsize = 14)
plt.yticks(fontname = "Times New Roman", fontsize = 14)
# plt.xlabel('')
plt.xlim(0, 100)
# plt.ylim(0, 1)
# plt.autoscale(enable = True, axis = 'y', tight = True)
# plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.gca().yaxis.set_label_coords(-0.05, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
# plt.tight_layout()
plt.subplot(10, 1, 9)
# plt.plot(Num_Sampling_list, V1, color = '#00A8CC', linewidth = 2, label = 'V1 - Vref')
plt.plot(Num_Sampling_list, I1, color = '#00A8CC', linewidth = 2, label = 'I1 - Iref')
plt.plot(Num_Sampling_list, Ith_list, color = '#F8766D', linewidth = 2, label = 'Ith')
plt.xlabel('Number of Experiments', fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.ylabel('I1', fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.yticks(fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.legend(prop = {'family': 'Times New Roman'}, loc = 'lower left')
plt.xticks(np.arange(0, 101, 10), fontname = "Times New Roman", fontsize = 14)
plt.yticks(fontname = "Times New Roman", fontsize = 14)
plt.xlim(0, 100)
# plt.ylim(0, 1)
plt.xlabel('')
# plt.autoscale(enable = True, axis = 'y', tight = True)
plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.gca().yaxis.set_label_coords(-0.05, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
# plt.tight_layout()
plt.subplot(10, 1, 8)
plt.plot(Num_Sampling_list, I2, color = '#00A8CC', linewidth = 2, label = 'I2 - Iref')
plt.plot(Num_Sampling_list, Ith_list, color = '#F8766D', linewidth = 2, label = 'Ith')
plt.xlabel('Number of Experiments', fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.ylabel('I2', fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.yticks(fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.legend(prop = {'family': 'Times New Roman'}, loc = 'lower left')
plt.xticks(np.arange(0, 101, 10), fontname = "Times New Roman", fontsize = 14)
plt.yticks(fontname = "Times New Roman", fontsize = 14)
plt.xlabel('')
plt.xlim(0, 100)
# plt.ylim(0, 1)
# plt.autoscale(enable = True, axis = 'y', tight = True)
plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.gca().yaxis.set_label_coords(-0.05, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
# plt.tight_layout()
plt.subplot(10, 1, 7)
plt.plot(Num_Sampling_list, I3, color = '#00A8CC', linewidth = 2, label = 'I3 - Iref')
plt.plot(Num_Sampling_list, Ith_list, color = '#F8766D', linewidth = 2, label = 'Ith')
plt.xlabel('Number of Experiments', fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.ylabel('I3', fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.yticks(fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.legend(prop = {'family': 'Times New Roman'}, loc = 'lower left')
plt.xticks(np.arange(0, 101, 10), fontname = "Times New Roman", fontsize = 14)
plt.yticks(fontname = "Times New Roman", fontsize = 14)
plt.xlabel('')
plt.xlim(0, 100)
# plt.ylim(0, 1)
# plt.autoscale(enable = True, axis = 'y', tight = True)
plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.gca().yaxis.set_label_coords(-0.05, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
# plt.tight_layout()
plt.subplot(10, 1, 6)
plt.plot(Num_Sampling_list, I4, color = '#00A8CC', linewidth = 2, label = 'I4 - Iref')
plt.plot(Num_Sampling_list, Ith_list, color = '#F8766D', linewidth = 2, label = 'Ith')
plt.xlabel('Number of Experiments', fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.ylabel('I4', fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.yticks(fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.legend(prop = {'family': 'Times New Roman'}, loc = 'lower left')
plt.xticks(np.arange(0, 101, 10), fontname = "Times New Roman", fontsize = 14)
plt.yticks(fontname = "Times New Roman", fontsize = 14)
plt.xlabel('')
plt.xlim(0, 100)
# plt.ylim(0, 1)
# plt.autoscale(enable = True, axis = 'y', tight = True)
plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.gca().yaxis.set_label_coords(-0.05, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
# plt.tight_layout()
plt.subplot(10, 1, 5)
plt.plot(Num_Sampling_list, I5, color = '#00A8CC', linewidth = 2, label = 'I5 - Iref')
plt.plot(Num_Sampling_list, Ith_list, color = '#F8766D', linewidth = 2, label = 'Ith')
plt.xlabel('Number of Experiments', fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.ylabel('I5', fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.yticks(fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.legend(prop = {'family': 'Times New Roman'}, loc = 'lower left')
plt.xticks(np.arange(0, 101, 10), fontname = "Times New Roman", fontsize = 14)
plt.yticks(fontname = "Times New Roman", fontsize = 14)
plt.xlabel('')
plt.xlim(0, 100)
# plt.ylim(0, 1)
# plt.autoscale(enable = True, axis = 'y', tight = True)
plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.gca().yaxis.set_label_coords(-0.05, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
# plt.tight_layout()
plt.subplot(10, 1, 4)
plt.plot(Num_Sampling_list, I6, color = '#00A8CC', linewidth = 2, label = 'I6 - Iref')
plt.plot(Num_Sampling_list, Ith_list, color = '#F8766D', linewidth = 2, label = 'Ith')
plt.xlabel('Number of Experiments', fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.ylabel('I6', fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.yticks(fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.legend(prop = {'family': 'Times New Roman'}, loc = 'lower left')
plt.xticks(np.arange(0, 101, 10), fontname = "Times New Roman", fontsize = 14)
plt.yticks(fontname = "Times New Roman", fontsize = 14)
plt.xlabel('')
plt.xlim(0, 100)
# plt.ylim(0, 1)
# plt.autoscale(enable = True, axis = 'y', tight = True)
plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.gca().yaxis.set_label_coords(-0.05, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
# plt.tight_layout()
plt.subplot(10, 1, 3)
plt.plot(Num_Sampling_list, I7, color = '#00A8CC', linewidth = 2, label = 'I7 - Iref')
plt.plot(Num_Sampling_list, Ith_list, color = '#F8766D', linewidth = 2, label = 'Ith')
plt.xlabel('Number of Experiments', fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.ylabel('I7', fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.yticks(fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.legend(prop = {'family': 'Times New Roman'}, loc = 'lower left')
plt.xticks(np.arange(0, 101, 10), fontname = "Times New Roman", fontsize = 14)
plt.yticks(fontname = "Times New Roman", fontsize = 14)
plt.xlabel('')
plt.xlim(0, 100)
# plt.ylim(0, 1)
# plt.autoscale(enable = True, axis = 'y', tight = True)
plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.gca().yaxis.set_label_coords(-0.05, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
# plt.tight_layout()
plt.subplot(10, 1, 2)
plt.plot(Num_Sampling_list, I8, color = '#00A8CC', linewidth = 2, label = 'I8 - Iref')
plt.plot(Num_Sampling_list, Ith_list, color = '#F8766D', linewidth = 2, label = 'Ith')
plt.xlabel('Number of Experiments', fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.ylabel('I8', fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.yticks(fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.legend(prop = {'family': 'Times New Roman'}, loc = 'lower left')
plt.xticks(np.arange(0, 101, 10), fontname = "Times New Roman", fontsize = 14)
plt.yticks(fontname = "Times New Roman", fontsize = 14)
plt.xlabel('')
plt.xlim(0, 100)
# plt.ylim(0, 1)
# plt.autoscale(enable = True, axis = 'y', tight = True)
plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.gca().yaxis.set_label_coords(-0.05, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
# plt.tight_layout()
plt.subplot(10, 1, 1)
plt.plot(Num_Sampling_list, I9, color = '#00A8CC', linewidth = 2, label = 'I9 - Iref')
plt.plot(Num_Sampling_list, Ith_list, color = '#F8766D', linewidth = 2, label = 'Ith')
plt.xlabel('Number of Experiments', fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.ylabel('I9  [\u03BCA]', fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.yticks(fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.legend(prop = {'family': 'Times New Roman'}, loc = 'lower left')
plt.xticks(np.arange(0, 101, 10), fontname = "Times New Roman", fontsize = 14)
plt.yticks(fontname = "Times New Roman", fontsize = 14)
plt.xlabel('')
plt.xlim(0, 100)
# plt.ylim(0, 1)
# plt.autoscale(enable = True, axis = 'y', tight = True)
plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.gca().yaxis.set_label_coords(-0.05, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
# plt.tight_layout()

folder_path = "data"
os.makedirs(folder_path, exist_ok = True)
current_time = datetime.datetime.now().strftime("%Y_%m%d_%H%M")
filename = f"softmax_Sampling_I-Iref_{current_time}.svg"
file_path = os.path.join(folder_path, filename)
plt.savefig(file_path, format = 'svg')
plt.show()
