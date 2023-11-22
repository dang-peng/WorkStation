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

num_experiment = 100
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
Vref_list = []
Vth_list = []
V_list = []
for j in range(num_experiment):
    y = 0
    num_samples = 0
    Vref = 0
    while y == 0 and num_samples < 1000:
        # yb = np.zeros((1, 10))
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
    yb_list.append(yb.copy())
    Vth_list.append(Vth)
    Num_Sampling_list.append(j)
    print("num_samples:", num_samples)

# 列出所有为1的点的横纵坐标
coordinates = []
for i, yb in enumerate(yb_list):
    ones_indices = np.where(yb == 1)[1]
    for idx in ones_indices:
        coordinates.append((i, idx))

# 分离横纵坐标
x_coords, y_coords = zip(*coordinates)
y_coords = [coord + 0.5 for coord in y_coords]
plt.rcParams['mathtext.fontset'] = 'stix'
# mpl.rcParams['font.family'] = 'STIX'
plt.rcParams['font.size'] = 30
# 设置绘图尺寸和比例
plt.figure(figsize = (12, 10))
plt.axis([0, 100, 0, 10])
plt.xticks(range(0, 101, 10))
plt.yticks(range(0, 11))
# plt.rcParams['mathtext.fontset'] = 'stix'
# 绘制每个点
plt.plot(x_coords, y_coords, marker = 'o', markersize = 10, color = '#4CA5E2', linestyle = '')
# 设置标题和轴标签
plt.xlabel('Number of Experiments', fontname = "Times New Roman", fontsize = 30, color = 'black')
plt.ylabel('Output Layer Results', fontname = "Times New Roman", fontsize = 30, color = 'black')
plt.xticks(fontname = "Times New Roman", fontsize = 26, color = 'black')
# plt.yticks(fontsize = 18, color = 'black')
# 设置纵坐标刻度和标签
y_ticks = np.arange(0.5, 10.5, 1)  # 设置纵坐标刻度值
# y_tick_labels = ['y{}'.format(i) for i in range(10)]  # 设置纵坐标标签
y_tick_labels = [r'$\mathrm{y}_\mathrm{' + str(i) + '}^\mathrm{b}$' for i in range(10)]  # 设置纵坐标标签
plt.yticks(y_ticks, y_tick_labels)
# plt.legend(prop = {'family': 'Times New Roman'}, loc = 'lower left')
# # plt.setp(plt.gca().get_xticklabels(), visible = False)  # 隐藏横坐标标签
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
plt.gca().yaxis.set_label_coords(-0.07, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
plt.subplots_adjust(top=0.9)  # 将顶部位置向上调整10%的子图高度

# plt.tight_layout()
folder_path = "data"
os.makedirs(folder_path, exist_ok = True)
current_time = datetime.datetime.now().strftime("%Y_%m%d_%H%M")
filename = f"softmax_Roster_plot_{current_time}.svg"
file_path = os.path.join(folder_path, filename)
plt.savefig(file_path, format = 'svg')
plt.show()
