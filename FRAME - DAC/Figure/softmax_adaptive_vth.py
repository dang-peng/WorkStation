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
Vth0 = 0.25
Vdd = 2.5
Vth = Vdd

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

num_experiment = 4
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
V0_list = []
V1_list = []
V2_list = []
V3_list = []
V4_list = []
V5_list = []
V6_list = []
V7_list = []
V8_list = []
V9_list = []
Vref0_list = []
Vref1_list = []
Vref2_list = []
Vref3_list = []
Vref4_list = []
Vref5_list = []
Vref6_list = []
Vref7_list = []
Vref8_list = []
Vref9_list = []
yb_list = []
Vref_list = []
Vth_list = []
sampling = 0

Rth = 1e3
Cth = 1e-9
dt = 1e-7

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
        
        Q = (Vth - Vth0) * Cth
        dQ = (Vth - Vth0) / Rth * dt
        Vth = (Q - dQ) / Cth + Vth0
        
        yb = np.where((V - Vref) > Vth, 1, 0)
        y = np.sum(yb, axis = 1)
        num_samples += 1
        
        V0_list.append(V[0][0])
        V1_list.append(V[0][1])
        V2_list.append(V[0][2])
        V3_list.append(V[0][3])
        V4_list.append(V[0][4])
        V5_list.append(V[0][5])
        V6_list.append(V[0][6])
        V7_list.append(V[0][7])
        V8_list.append(V[0][8])
        V9_list.append(V[0][9])
        Vref0_list.append(Vref[0][0])
        Vref1_list.append(Vref[0][1])
        Vref2_list.append(Vref[0][2])
        Vref3_list.append(Vref[0][3])
        Vref4_list.append(Vref[0][4])
        Vref5_list.append(Vref[0][5])
        Vref6_list.append(Vref[0][6])
        Vref7_list.append(Vref[0][7])
        Vref8_list.append(Vref[0][8])
        Vref9_list.append(Vref[0][9])
        Vth_list.append(Vth)
        
        sampling = sampling + 1
        Num_Sampling_list.append(sampling)
    yb_sum = yb_sum + np.array(yb)
    Vth = Vdd
    
    print("num_samples:", num_samples)
V0 = (np.array(V0_list) - np.array(Vref0_list))
V1 = (np.array(V1_list) - np.array(Vref1_list))
V2 = (np.array(V2_list) - np.array(Vref2_list))
V3 = (np.array(V3_list) - np.array(Vref3_list))
V4 = (np.array(V4_list) - np.array(Vref4_list))
V5 = (np.array(V5_list) - np.array(Vref5_list))
V6 = (np.array(V6_list) - np.array(Vref6_list))
V7 = (np.array(V7_list) - np.array(Vref7_list))
V8 = (np.array(V8_list) - np.array(Vref8_list))
V9 = (np.array(V9_list) - np.array(Vref9_list))

fig = plt.figure()
# 调整图形的尺寸
fig.set_size_inches(12, 10)  # 设置宽度为8英寸，高度为6英寸
# 设置Matplotlib的LaTeX渲染器为True
plt.rcParams['mathtext.fontset'] = 'stix'
# plt.rcParams['font.size'] = 22
y_labels = [r'$\mathrm{V}_\mathrm{' + str(i) + '}$' for i in range(10)]  # 设置纵坐标标签
legend_labels = r'$\mathrm{V}_\mathrm{j}$' + ' - ' + r'$\mathrm{V}_\mathrm{ref}$'
Vth_labels = r'$\mathrm{V}_\mathrm{th}$'
plt.subplot(10, 1, 10)
# plt.plot(Num_Sampling_list, Vref_list, '#00B0F0', linewidth = 3, label = 'Vref')
plt.plot(Num_Sampling_list, V0, color = '#FF4C4C', linewidth = 3, label = legend_labels)
plt.plot(Num_Sampling_list, Vth_list, color = '#00B0F0', linewidth = 3, label = Vth_labels)
intersection_x = []
intersection_y = []
for i in range(len(Num_Sampling_list)):
    if V0[i] >= Vth_list[i]:
        intersection_x.append(Num_Sampling_list[i])
        intersection_y.append(V0[i])
if intersection_x:
    plt.scatter(intersection_x, intersection_y, color = '#00B050', marker = 'o', s = 250, label = 'Intersection',
                zorder = 2)
plt.xlabel('Number of Sampling', fontname = "Times New Roman", fontsize = 30, color = 'black')
plt.ylabel(y_labels[0], fontname = "Times New Roman", fontsize = 30, color = 'black')
# plt.legend(prop = {'family': 'Times New Roman', 'size': 20}, loc = 'lower left')
plt.xticks(np.arange(0, 301, 50), fontname = "Times New Roman", fontsize = 26)
plt.yticks(fontname = "Times New Roman", fontsize = 26)
# plt.xlabel('')
plt.xlim(0, len(Num_Sampling_list))
# plt.ylim(0, 1)
# plt.autoscale(enable = True, axis = 'y', tight = True)
# plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.gca().yaxis.set_label_coords(-0.04, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
# plt.tight_layout()
plt.subplot(10, 1, 9)
# plt.plot(Num_Sampling_list, V1, color = '#FF4C4C', linewidth = 3, label = 'V1 - Vref')
plt.plot(Num_Sampling_list, V1, color = '#FF4C4C', linewidth = 3, label = legend_labels)
plt.plot(Num_Sampling_list, Vth_list, color = '#00B0F0', linewidth = 3, label = Vth_labels)
intersection_x = []
intersection_y = []
for i in range(len(Num_Sampling_list)):
    if V1[i] >= Vth_list[i]:
        intersection_x.append(Num_Sampling_list[i])
        intersection_y.append(V1[i])
if intersection_x:
    plt.scatter(intersection_x, intersection_y, color = '#00B050', marker = 'o', s = 250, label = 'Intersection',
                zorder = 2)
plt.xlabel('Number of Experiments', fontname = "Times New Roman", fontsize = 30, color = 'black')
plt.ylabel(y_labels[1], fontname = "Times New Roman", fontsize = 30, color = 'black')
# plt.legend(prop = {'family': 'Times New Roman', 'size': 20}, loc = 'lower left')
plt.xticks(np.arange(0, 301, 50), fontname = "Times New Roman", fontsize = 26)
plt.yticks(fontname = "Times New Roman", fontsize = 26)
plt.xlim(0, len(Num_Sampling_list))
# plt.ylim(0, 1)
plt.xlabel('')
# plt.autoscale(enable = True, axis = 'y', tight = True)
plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.gca().yaxis.set_label_coords(-0.04, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
# plt.tight_layout()
plt.subplot(10, 1, 8)
plt.plot(Num_Sampling_list, V2, color = '#FF4C4C', linewidth = 3, label = legend_labels)
plt.plot(Num_Sampling_list, Vth_list, color = '#00B0F0', linewidth = 3, label = Vth_labels)
intersection_x = []
intersection_y = []
for i in range(len(Num_Sampling_list)):
    if V2[i] >= Vth_list[i]:
        intersection_x.append(Num_Sampling_list[i])
        intersection_y.append(V2[i])
if intersection_x:
    plt.scatter(intersection_x, intersection_y, color = '#00B050', marker = 'o', s = 250, label = 'Intersection',
                zorder = 2)
plt.xlabel('Number of Experiments', fontname = "Times New Roman", fontsize = 30, color = 'black')
plt.ylabel(y_labels[2], fontname = "Times New Roman", fontsize = 30, color = 'black')
# plt.legend(prop = {'family': 'Times New Roman', 'size': 20}, loc = 'lower left')
plt.xticks(np.arange(0, 301, 50), fontname = "Times New Roman", fontsize = 26)
plt.yticks(fontname = "Times New Roman", fontsize = 26)
plt.xlabel('')
plt.xlim(0, len(Num_Sampling_list))
# plt.ylim(0, 1)
# plt.autoscale(enable = True, axis = 'y', tight = True)
plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.gca().yaxis.set_label_coords(-0.04, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
# plt.tight_layout()
plt.subplot(10, 1, 7)
plt.plot(Num_Sampling_list, V3, color = '#FF4C4C', linewidth = 3, label = legend_labels)
plt.plot(Num_Sampling_list, Vth_list, color = '#00B0F0', linewidth = 3, label = Vth_labels)
intersection_x = []
intersection_y = []
for i in range(len(Num_Sampling_list)):
    if V3[i] >= Vth_list[i]:
        intersection_x.append(Num_Sampling_list[i])
        intersection_y.append(V3[i])
if intersection_x:
    plt.scatter(intersection_x, intersection_y, color = '#00B050', marker = 'o', s = 250, label = 'Intersection',
                zorder = 2)
plt.xlabel('Number of Experiments', fontname = "Times New Roman", fontsize = 30, color = 'black')
plt.ylabel(y_labels[3], fontname = "Times New Roman", fontsize = 30, color = 'black')
# plt.legend(prop = {'family': 'Times New Roman', 'size': 20}, loc = 'lower left')
plt.xticks(np.arange(0, 301, 50), fontname = "Times New Roman", fontsize = 26)
plt.yticks(fontname = "Times New Roman", fontsize = 26)
plt.xlabel('')
plt.xlim(0, len(Num_Sampling_list))
# plt.ylim(0, 1)
# plt.autoscale(enable = True, axis = 'y', tight = True)
plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.gca().yaxis.set_label_coords(-0.04, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
# plt.tight_layout()
plt.subplot(10, 1, 6)
plt.plot(Num_Sampling_list, V4, color = '#FF4C4C', linewidth = 3, label = legend_labels)
plt.plot(Num_Sampling_list, Vth_list, color = '#00B0F0', linewidth = 3, label = Vth_labels)
intersection_x = []
intersection_y = []
for i in range(len(Num_Sampling_list)):
    if V4[i] >= Vth_list[i]:
        intersection_x.append(Num_Sampling_list[i])
        intersection_y.append(V4[i])
if intersection_x:
    plt.scatter(intersection_x, intersection_y, color = '#00B050', marker = 'o', s = 250, label = 'Intersection',
                zorder = 2)
plt.xlabel('Number of Experiments', fontname = "Times New Roman", fontsize = 30, color = 'black')
plt.ylabel(y_labels[4], fontname = "Times New Roman", fontsize = 30, color = 'black')
# plt.legend(prop = {'family': 'Times New Roman', 'size': 20}, loc = 'lower left')
plt.xticks(np.arange(0, 301, 50), fontname = "Times New Roman", fontsize = 26)
plt.yticks(fontname = "Times New Roman", fontsize = 26)
plt.xlabel('')
plt.xlim(0, len(Num_Sampling_list))
# plt.ylim(0, 1)
# plt.autoscale(enable = True, axis = 'y', tight = True)
plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.gca().yaxis.set_label_coords(-0.04, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
# plt.tight_layout()
plt.subplot(10, 1, 5)
plt.plot(Num_Sampling_list, V5, color = '#FF4C4C', linewidth = 3, label = legend_labels)
plt.plot(Num_Sampling_list, Vth_list, color = '#00B0F0', linewidth = 3, label = Vth_labels)
intersection_x = []
intersection_y = []
for i in range(len(Num_Sampling_list)):
    if V5[i] >= Vth_list[i]:
        intersection_x.append(Num_Sampling_list[i])
        intersection_y.append(V5[i])
if intersection_x:
    plt.scatter(intersection_x, intersection_y, color = '#00B050', marker = 'o', s = 250, label = 'Intersection',
                zorder = 2)
plt.xlabel('Number of Experiments', fontname = "Times New Roman", fontsize = 30, color = 'black')
plt.ylabel(y_labels[5], fontname = "Times New Roman", fontsize = 30, color = 'black')
# plt.legend(prop = {'family': 'Times New Roman', 'size': 20}, loc = 'lower left')
plt.xticks(np.arange(0, 301, 50), fontname = "Times New Roman", fontsize = 26)
plt.yticks(fontname = "Times New Roman", fontsize = 26)
plt.xlabel('')
plt.xlim(0, len(Num_Sampling_list))
# plt.ylim(0, 1)
# plt.autoscale(enable = True, axis = 'y', tight = True)
plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.gca().yaxis.set_label_coords(-0.04, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
# plt.tight_layout()
plt.subplot(10, 1, 4)
plt.plot(Num_Sampling_list, V6, color = '#FF4C4C', linewidth = 3, label = legend_labels)
plt.plot(Num_Sampling_list, Vth_list, color = '#00B0F0', linewidth = 3, label = Vth_labels)
intersection_x = []
intersection_y = []
for i in range(len(Num_Sampling_list)):
    if V6[i] >= Vth_list[i]:
        intersection_x.append(Num_Sampling_list[i])
        intersection_y.append(V6[i])
if intersection_x:
    plt.scatter(intersection_x, intersection_y, color = '#00B050', marker = 'o', s = 250, label = 'Intersection',
                zorder = 2)
plt.xlabel('Number of Experiments', fontname = "Times New Roman", fontsize = 30, color = 'black')
plt.ylabel(y_labels[6], fontname = "Times New Roman", fontsize = 30, color = 'black')
# plt.legend(prop = {'family': 'Times New Roman', 'size': 20}, loc = 'lower left')
plt.xticks(np.arange(0, 301, 50), fontname = "Times New Roman", fontsize = 26)
plt.yticks(fontname = "Times New Roman", fontsize = 26)
plt.xlabel('')
plt.xlim(0, len(Num_Sampling_list))
# plt.ylim(0, 1)
# plt.autoscale(enable = True, axis = 'y', tight = True)
plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.gca().yaxis.set_label_coords(-0.04, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
# plt.tight_layout()
plt.subplot(10, 1, 3)
plt.plot(Num_Sampling_list, V7, color = '#FF4C4C', linewidth = 3, label = legend_labels)
plt.plot(Num_Sampling_list, Vth_list, color = '#00B0F0', linewidth = 3, label = Vth_labels)
intersection_x = []
intersection_y = []
for i in range(len(Num_Sampling_list)):
    if V7[i] >= Vth_list[i]:
        intersection_x.append(Num_Sampling_list[i])
        intersection_y.append(V7[i])
if intersection_x:
    plt.scatter(intersection_x, intersection_y, color = '#00B050', marker = 'o', s = 250, label = 'Intersection',
                zorder = 2)
plt.xlabel('Number of Experiments', fontname = "Times New Roman", fontsize = 30, color = 'black')
plt.ylabel(y_labels[7], fontname = "Times New Roman", fontsize = 30, color = 'black')
# plt.legend(prop = {'family': 'Times New Roman', 'size': 20}, loc = 'lower left')
plt.xticks(np.arange(0, 301, 50), fontname = "Times New Roman", fontsize = 26)
plt.yticks(fontname = "Times New Roman", fontsize = 26)
plt.xlabel('')
plt.xlim(0, len(Num_Sampling_list))
# plt.ylim(0, 1)
# plt.autoscale(enable = True, axis = 'y', tight = True)
plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.gca().yaxis.set_label_coords(-0.04, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
# plt.tight_layout()
plt.subplot(10, 1, 2)
plt.plot(Num_Sampling_list, V8, color = '#FF4C4C', linewidth = 3, label = legend_labels)
plt.plot(Num_Sampling_list, Vth_list, color = '#00B0F0', linewidth = 3, label = Vth_labels)
intersection_x = []
intersection_y = []
for i in range(len(Num_Sampling_list)):
    if V8[i] >= Vth_list[i]:
        intersection_x.append(Num_Sampling_list[i])
        intersection_y.append(V8[i])
intersection_x.pop()
intersection_y.pop()
if intersection_x:
    plt.scatter(intersection_x, intersection_y, color = '#00B050', marker = 'o', s = 250, label = 'Intersection',
                zorder = 2)
plt.xlabel('Number of Experiments', fontname = "Times New Roman", fontsize = 30, color = 'black')
plt.ylabel(y_labels[8], fontname = "Times New Roman", fontsize = 30, color = 'black')
# plt.legend(prop = {'family': 'Times New Roman', 'size': 20}, loc = 'lower left')
plt.xticks(np.arange(0, 301, 50), fontname = "Times New Roman", fontsize = 26)
plt.yticks(fontname = "Times New Roman", fontsize = 26)
plt.xlabel('')
plt.xlim(0, len(Num_Sampling_list))
# plt.ylim(0, 1)
# plt.autoscale(enable = True, axis = 'y', tight = True)
plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.gca().yaxis.set_label_coords(-0.04, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
# plt.tight_layout()
plt.subplot(10, 1, 1)
plt.plot(Num_Sampling_list, V9, color = '#FF4C4C', linewidth = 3, label = legend_labels)
plt.plot(Num_Sampling_list, Vth_list, color = '#00B0F0', linewidth = 3, label = Vth_labels)
intersection_x = []
intersection_y = []
for i in range(len(Num_Sampling_list)):
    if V9[i] >= Vth_list[i]:
        intersection_x.append(Num_Sampling_list[i])
        intersection_y.append(V9[i])
if intersection_x:
    plt.scatter(intersection_x, intersection_y, color = '#00B050', marker = 'o', s = 250, label = 'Intersection',
                zorder = 2)
plt.xlabel('Number of Experiments', fontname = "Times New Roman", fontsize = 30, color = 'black')
plt.ylabel('    ' + y_labels[9] + ' [V]', fontname = "Times New Roman", fontsize = 30, color = 'black')
plt.legend(prop = {'family': 'Times New Roman', 'size': 20}, loc = 'lower left')
plt.xticks(np.arange(0, 301, 50), fontname = "Times New Roman", fontsize = 26)
plt.yticks(fontname = "Times New Roman", fontsize = 26)
legend = plt.legend(prop = {'family': 'Times New Roman', 'size': 22}, loc = 'upper left')
legend.get_frame().set_alpha(1.0)
plt.xlabel('')
plt.xlim(0, len(Num_Sampling_list))
# plt.ylim(0, 1)
# plt.autoscale(enable = True, axis = 'y', tight = True)
plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.gca().yaxis.set_label_coords(-0.04, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
plt.subplots_adjust(top=0.9)  # 将顶部位置向上调整10%的子图高度
# plt.tight_layout()
folder_path = "data"
os.makedirs(folder_path, exist_ok = True)
current_time = datetime.datetime.now().strftime("%Y_%m%d_%H%M")
filename = f"softmax_Sampling_Vth_{current_time}.svg"
file_path = os.path.join(folder_path, filename)
plt.savefig(file_path, format = 'svg')
plt.show()
