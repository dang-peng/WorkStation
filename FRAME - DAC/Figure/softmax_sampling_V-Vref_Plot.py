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
Vth = 0.25

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
        I = I0 + I_noise
        V = R_TIA * I
        Iref = I_ref + Iref_noise
        Vref = R_TIA * Iref
        yb = np.where((V - Vref) > Vth, 1, 0)
        y = np.sum(yb, axis = 1)
        num_samples += 1
    Vref_list.append(Vref[0][0])
    yb_sum = yb_sum + np.array(yb)
    # yb = [item for sublist in yb for item in sublist]
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
    Num_Sampling_list.append(j)
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
fig.set_size_inches(12, 10)
# 设置Matplotlib的LaTeX渲染器为True
plt.rcParams['mathtext.fontset'] = 'stix'
# plt.rcParams['font.size'] = 22
y_labels = [r'$\mathrm{V}_\mathrm{' + str(i) + '}$' for i in range(10)]  # 设置纵坐标标签
legend_labels = r'$\mathrm{V}_\mathrm{j}$' + ' - ' + r'$\mathrm{V}_\mathrm{ref}$'
Vth_labels = r'$\mathrm{V}_\mathrm{th}$'
# # 使用数学字体表示刻度
# formatter = ticker.ScalarFormatter(useMathText = True)
# formatter.set_scientific(False)  # 禁用科学计数法
# plt.gca().xaxis.set_major_formatter(formatter)
plt.subplot(10, 1, 10)
# plt.plot(Num_Sampling_list, V0, color = '#00B0F0', linewidth = 3, label = 'V0 - Vref')
plt.plot(Num_Sampling_list, V0, color = '#00B0F0', linewidth = 3, label = legend_labels)
plt.plot(Num_Sampling_list, Vth_list, color = '#FF4C4C', linewidth = 3, label = Vth_labels)
plt.xlabel('Number of Experiments', fontname = "Times New Roman", fontsize = 30, color = 'black')
plt.gca().set_yticks([-0.2, 0, 0.2])  # 设置y轴刻度为-0.25、0、0.25
plt.gca().set_yticklabels([-0.25, '', 0.25])  # 设置刻度标签为-0.25、空字符串、0.25
# plt.ylabel('V0', fontname = "Times New Roman", fontsize = 26, color = 'black')
plt.ylabel(y_labels[0], fontname = "Times New Roman", fontsize = 30, color = 'black')
plt.yticks(fontname = "Times New Roman", fontsize = 22, color = 'black')
# plt.legend(prop = {'family': 'Times New Roman'}, loc = 'lower left')
plt.xticks(np.arange(0, 101, 10), fontname = "Times New Roman", fontsize = 26)
plt.yticks(fontname = "Times New Roman", fontsize = 22)
# plt.xlabel('')
plt.xlim(0, 100)
# plt.ylim(0, 1)
# plt.autoscale(enable = True, axis = 'y', tight = True)
# plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.gca().yaxis.set_label_coords(-0.08, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
plt.subplot(10, 1, 9)
# plt.plot(Num_Sampling_list, V1, color = '#00B0F0', linewidth = 3, label = 'V1 - Vref')
plt.plot(Num_Sampling_list, V1, color = '#00B0F0', linewidth = 3, label = legend_labels)
plt.plot(Num_Sampling_list, Vth_list, color = '#FF4C4C', linewidth = 3, label = Vth_labels)
plt.xlabel('Number of Experiments', fontname = "Times New Roman", fontsize = 26, color = 'black')
plt.ylabel(y_labels[1], fontname = "Times New Roman", fontsize = 30, color = 'black')
plt.yticks(fontname = "Times New Roman", fontsize = 22, color = 'black')
plt.gca().set_yticks([-0.2, 0, 0.2])  # 设置y轴刻度为-0.25、0、0.25
plt.gca().set_yticklabels([-0.25, '', 0.25])  # 设置刻度标签为-0.25、空字符串、0.25
# plt.legend(prop = {'family': 'Times New Roman'}, loc = 'lower left')
plt.xticks(np.arange(0, 101, 10), fontname = "Times New Roman", fontsize = 26)
plt.yticks(fontname = "Times New Roman", fontsize = 22)
plt.xlim(0, 100)
# plt.ylim(0, 1)
plt.xlabel('')
# plt.autoscale(enable = True, axis = 'y', tight = True)
plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.gca().yaxis.set_label_coords(-0.08, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
# plt.tight_layout()
plt.subplot(10, 1, 8)
plt.plot(Num_Sampling_list, V2, color = '#00B0F0', linewidth = 3, label = legend_labels)
plt.plot(Num_Sampling_list, Vth_list, color = '#FF4C4C', linewidth = 3, label = Vth_labels)
plt.xlabel('Number of Experiments', fontname = "Times New Roman", fontsize = 26, color = 'black')
plt.ylabel(y_labels[2], fontname = "Times New Roman", fontsize = 30, color = 'black')
plt.yticks(fontname = "Times New Roman", fontsize = 22, color = 'black')
plt.gca().set_yticks([-0.2, 0, 0.2])  # 设置y轴刻度为-0.25、0、0.25
plt.gca().set_yticklabels([-0.25, '', 0.25])  # 设置刻度标签为-0.25、空字符串、0.25
# plt.legend(prop = {'family': 'Times New Roman'}, loc = 'lower left')
plt.xticks(np.arange(0, 101, 10), fontname = "Times New Roman", fontsize = 26)
plt.yticks(fontname = "Times New Roman", fontsize = 22)
plt.xlabel('')
plt.xlim(0, 100)
# plt.ylim(0, 1)
# plt.autoscale(enable = True, axis = 'y', tight = True)
plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.gca().yaxis.set_label_coords(-0.08, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
# plt.tight_layout()
plt.subplot(10, 1, 7)
plt.plot(Num_Sampling_list, V3, color = '#00B0F0', linewidth = 3, label = legend_labels)
plt.plot(Num_Sampling_list, Vth_list, color = '#FF4C4C', linewidth = 3, label = Vth_labels)
plt.xlabel('Number of Experiments', fontname = "Times New Roman", fontsize = 26, color = 'black')
plt.ylabel(y_labels[3], fontname = "Times New Roman", fontsize = 30, color = 'black')
plt.yticks(fontname = "Times New Roman", fontsize = 22, color = 'black')
plt.gca().set_yticks([-0.2, 0, 0.2])  # 设置y轴刻度为-0.25、0、0.25
plt.gca().set_yticklabels([-0.25, '', 0.25])  # 设置刻度标签为-0.25、空字符串、0.25
# plt.legend(prop = {'family': 'Times New Roman'}, loc = 'lower left')
plt.xticks(np.arange(0, 101, 10), fontname = "Times New Roman", fontsize = 26)
plt.yticks(fontname = "Times New Roman", fontsize = 22)
plt.xlabel('')
plt.xlim(0, 100)
# plt.ylim(0, 1)
# plt.autoscale(enable = True, axis = 'y', tight = True)
plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.gca().yaxis.set_label_coords(-0.08, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
# plt.tight_layout()
plt.subplot(10, 1, 6)
plt.plot(Num_Sampling_list, V4, color = '#00B0F0', linewidth = 3, label = legend_labels)
plt.plot(Num_Sampling_list, Vth_list, color = '#FF4C4C', linewidth = 3, label = Vth_labels)
plt.xlabel('Number of Experiments', fontname = "Times New Roman", fontsize = 26, color = 'black')
plt.ylabel(y_labels[4], fontname = "Times New Roman", fontsize = 30, color = 'black')
plt.gca().set_yticks([-0.2, 0, 0.2])  # 设置y轴刻度为-0.25、0、0.25
plt.gca().set_yticklabels([-0.25, '', 0.25])  # 设置刻度标签为-0.25、空字符串、0.25
# plt.legend(prop = {'family': 'Times New Roman'}, loc = 'lower left')
plt.xticks(np.arange(0, 101, 10), fontname = "Times New Roman", fontsize = 26)
plt.yticks(fontname = "Times New Roman", fontsize = 22, color = 'black')
plt.xlabel('')
plt.xlim(0, 100)
# plt.ylim(0, 1)
# plt.autoscale(enable = True, axis = 'y', tight = True)
plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.gca().yaxis.set_label_coords(-0.08, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
# plt.tight_layout()
plt.subplot(10, 1, 5)
plt.plot(Num_Sampling_list, V5, color = '#00B0F0', linewidth = 3, label = legend_labels)
plt.plot(Num_Sampling_list, Vth_list, color = '#FF4C4C', linewidth = 3, label = Vth_labels)
plt.xlabel('Number of Experiments', fontname = "Times New Roman", fontsize = 26, color = 'black')
plt.ylabel(y_labels[5], fontname = "Times New Roman", fontsize = 30, color = 'black')
plt.yticks(fontname = "Times New Roman", fontsize = 22, color = 'black')
plt.gca().set_yticks([-0.2, 0, 0.2])  # 设置y轴刻度为-0.25、0、0.25
plt.gca().set_yticklabels([-0.25, '', 0.25])  # 设置刻度标签为-0.25、空字符串、0.25
# plt.legend(prop = {'family': 'Times New Roman'}, loc = 'lower left')
plt.xticks(np.arange(0, 101, 10), fontname = "Times New Roman", fontsize = 26)
plt.xlabel('')
plt.xlim(0, 100)
# plt.ylim(0, 1)
# plt.autoscale(enable = True, axis = 'y', tight = True)
plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.gca().yaxis.set_label_coords(-0.08, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
# plt.tight_layout()
plt.subplot(10, 1, 4)
plt.plot(Num_Sampling_list, V6, color = '#00B0F0', linewidth = 3, label = legend_labels)
plt.plot(Num_Sampling_list, Vth_list, color = '#FF4C4C', linewidth = 3, label = Vth_labels)
plt.xlabel('Number of Experiments', fontname = "Times New Roman", fontsize = 26, color = 'black')
plt.ylabel(y_labels[6], fontname = "Times New Roman", fontsize = 30, color = 'black')
plt.yticks(fontname = "Times New Roman", fontsize = 22, color = 'black')
plt.gca().set_yticks([-0.2, 0, 0.2])  # 设置y轴刻度为-0.25、0、0.25
plt.gca().set_yticklabels([-0.25, '', 0.25])  # 设置刻度标签为-0.25、空字符串、0.25
# plt.legend(prop = {'family': 'Times New Roman'}, loc = 'lower left')
plt.xticks(np.arange(0, 101, 10), fontname = "Times New Roman", fontsize = 26)
plt.xlabel('')
plt.xlim(0, 100)
# plt.ylim(0, 1)
# plt.autoscale(enable = True, axis = 'y', tight = True)
plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.gca().yaxis.set_label_coords(-0.08, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
# plt.tight_layout()
plt.subplot(10, 1, 3)
plt.plot(Num_Sampling_list, V7, color = '#00B0F0', linewidth = 3, label = legend_labels)
plt.plot(Num_Sampling_list, Vth_list, color = '#FF4C4C', linewidth = 3, label = Vth_labels)
plt.xlabel('Number of Experiments', fontname = "Times New Roman", fontsize = 26, color = 'black')
plt.ylabel(y_labels[7], fontname = "Times New Roman", fontsize = 30, color = 'black')
plt.yticks(fontname = "Times New Roman", fontsize = 22, color = 'black')
plt.gca().set_yticks([-0.2, 0, 0.2])  # 设置y轴刻度为-0.25、0、0.25
plt.gca().set_yticklabels([-0.25, '', 0.25])  # 设置刻度标签为-0.25、空字符串、0.25
# plt.legend(prop = {'family': 'Times New Roman'}, loc = 'lower left')
plt.xticks(np.arange(0, 101, 10), fontname = "Times New Roman", fontsize = 26)
plt.xlabel('')
plt.xlim(0, 100)
# plt.ylim(0, 1)
# plt.autoscale(enable = True, axis = 'y', tight = True)
plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.gca().yaxis.set_label_coords(-0.08, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
# plt.tight_layout()
plt.subplot(10, 1, 2)
plt.plot(Num_Sampling_list, V8, color = '#00B0F0', linewidth = 3, label = legend_labels)
plt.plot(Num_Sampling_list, Vth_list, color = '#FF4C4C', linewidth = 3, label = Vth_labels)
plt.xlabel('Number of Experiments', fontname = "Times New Roman", fontsize = 26, color = 'black')
plt.ylabel(y_labels[8], fontname = "Times New Roman", fontsize = 30, color = 'black')
plt.yticks(fontname = "Times New Roman", fontsize = 22, color = 'black')
plt.gca().set_yticks([-0.2, 0, 0.2])  # 设置y轴刻度为-0.25、0、0.25
plt.gca().set_yticklabels([-0.25, '', 0.25])  # 设置刻度标签为-0.25、空字符串、0.25
# plt.legend(prop = {'family': 'Times New Roman'}, loc = 'lower left')
plt.xticks(np.arange(0, 101, 10), fontname = "Times New Roman", fontsize = 26)
plt.xlabel('')
plt.xlim(0, 100)
# plt.ylim(0, 1)
# plt.autoscale(enable = True, axis = 'y', tight = True)
plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.gca().yaxis.set_label_coords(-0.08, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
# plt.tight_layout()
plt.subplot(10, 1, 1)
plt.plot(Num_Sampling_list, V9, color = '#00B0F0', linewidth = 3, label = legend_labels)
plt.plot(Num_Sampling_list, Vth_list, color = '#FF4C4C', linewidth = 3, label = Vth_labels)
plt.xlabel('Number of Experiments', fontname = "Times New Roman", fontsize = 30, color = 'black')
plt.ylabel('    '+y_labels[9] + ' [V]', fontname = "Times New Roman", fontsize = 30, color = 'black')
plt.yticks(fontname = "Times New Roman", fontsize = 22, color = 'black')
legend = plt.legend(prop = {'family': 'Times New Roman', 'size': 20}, loc='upper left')
legend.get_frame().set_alpha(1.0)
# plt.gca().set_yticks([-0.25, 0.25])  # 设置y轴刻度为只有-0.25和0.25
plt.gca().set_yticks([-0.2, 0, 0.2])  # 设置y轴刻度为-0.25、0、0.25
plt.gca().set_yticklabels([-0.25, '', 0.25])  # 设置刻度标签为-0.25、空字符串、0.25
plt.xticks(np.arange(0, 101, 10), fontname = "Times New Roman", fontsize = 26)
plt.xlabel('')
plt.xlim(0, 100)
# plt.ylim(0, 1)
# plt.autoscale(enable = True, axis = 'y', tight = True)
plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.gca().yaxis.set_label_coords(-0.08, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
plt.subplots_adjust(top=0.9)  # 将顶部位置向上调整10%的子图高度

# plt.subplots_adjust(hspace=0.3)  # 设置纵向间距为子图高度的50%
# plt.tight_layout()
folder_path = "data"
os.makedirs(folder_path, exist_ok = True)
current_time = datetime.datetime.now().strftime("%Y_%m%d_%H%M%S")
filename = f"softmax_Sampling_V-Vref_{current_time}.svg"
file_path = os.path.join(folder_path, filename)
plt.savefig(file_path, format = 'svg')
plt.show()
