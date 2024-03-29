import datetime
import os

import numpy as np
from matplotlib import pyplot as plt


def softmax(z):
    z -= np.max(z, axis = 1, keepdims = True)
    return np.exp(z) / np.sum(np.exp(z), axis = 1, keepdims = True)


# 10 20 70 100
SEED = 1
# SEED = 100
np.random.seed(SEED)
rows = 100
cols = 1
K = 1.380649e-23
T = 300
delta_f = 1e10
Vr = 0.05
N = 4 * K * T * delta_f

Vth = 0.1
# Vth = 0.1 # Vth = 0.1
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
a1 = Vr * G0 / np.sqrt(2 * N * 100 * Gref)
print("a1 = ", a1)
num_experiment = 100
X = np.random.randint(2, size = (rows, cols))
V = X * Vr
G_Array = Weights * G0 + Gref
G_Array = np.clip(G_Array, Gmin, Gmax)
I0 = np.dot(V.T, G_Array)
Gref_Col = np.ones((rows, 1)) * Gref
Iref = np.dot(V.T, Gref_Col)

yb_sum = np.zeros((1, 10))
yb = []
yb_list = []
I_noise = 0
for j in range(num_experiment):
    y = 0
    num_samples = 0
    while y == 0 and num_samples < 2000:
        yb[:] = np.zeros_like(yb)
        Iref_device_noise = G_Array.copy()
        for G_ref in np.nditer(Iref_device_noise, op_flags = ['readwrite']):
            deviceRef_noise = np.random.normal(0, np.sqrt(N * G_ref), (1, 1))
            G_ref[...] = deviceRef_noise
        Iref_noise = np.sum(Iref_device_noise, axis = 0)
        Array_noise = G_Array.copy()
        for G_device in np.nditer(Array_noise, op_flags = ['readwrite']):
            device_I_noise = np.random.normal(0, np.sqrt(N * G_device), (1, 1))
            G_device[...] = device_I_noise
        I_noise = np.sum(Array_noise, axis = 0)
        I_total = I0 + I_noise
        V = R_TIA * I_total
        Vref = R_TIA * (Iref + Iref_noise)
        yb = np.where((V - Vref) > Vth, 1, 0)
        y = np.sum(yb, axis = 1)
        num_samples += 1
    yb_sum = yb_sum + np.array(yb)
    yb_list.append(yb)
    print("num_samples:", num_samples)
yb_prob_experiment = yb_sum / num_experiment
yb_prob = softmax(I0 / (Vr * G0))
yb_prob_list = yb_prob.flatten().tolist()
yb_prob_index_list = list(range(len(yb_prob_list)))
yb_prob_experiment_list = yb_prob_experiment.flatten().tolist()
yb_prob_experiment_index_list = list(range(len(yb_prob_experiment_list)))
colors1 = ['#74BBE9']
colors2 = ['#EFB9BD']
plt.rcParams['mathtext.fontset'] = 'stix'
# plt.rcParams['font.size'] = 22
plt.subplots(figsize = (12, 11))  # 使用sharey参数共享纵坐标轴
width = 0.4
plt.bar(np.arange(len(yb_prob_experiment_index_list)) - width / 2, yb_prob_experiment_list, width,
        label = 'Experiment',
        color = colors1)
plt.xlim(-1, 10)
plt.bar([i + width / 2 for i in np.arange(len(yb_prob_index_list))], yb_prob_list, width, label = 'SoftMax',
        color = colors2, edgecolor = 'black', linewidth = 1)
# ax.set_yscale("log")
# 设置纵坐标刻度和标签
x_ticks = np.arange(10)  # 设置纵坐标刻度值
# y_tick_labels = ['y{}'.format(i) for i in range(10)]  # 设置纵坐标标签
x_tick_labels = [r'$\mathrm{y}_\mathrm{' + str(i) + '}^\mathrm{b}$' for i in range(10)]  # 设置纵坐标标签
plt.xticks(x_ticks, x_tick_labels, fontsize = 30)
plt.xlabel('Output Layer Results ', fontname = "Times New Roman", fontsize = 30, color = 'black')
plt.ylabel('Probability', fontname = "Times New Roman", fontsize = 30, color = 'black')
plt.yticks(fontname = "Times New Roman", fontsize = 26)
plt.legend(prop = {'family': 'Times New Roman', 'size': 26}, loc = 'upper right')
# plt.text(0.03, 0.965, 'Vr = ' + '{} V'.format(Vr) + '\nVth = {} V'.format(Vth),
#          transform = plt.gca().transAxes,
#          verticalalignment = 'top', horizontalalignment = 'left',
#          fontsize = 12, bbox = dict(facecolor = 'white', edgecolor = 'lightgray', boxstyle = 'round'),
#          fontproperties = FontProperties(family = 'Times New Roman', size = 12))
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray', zorder = 0)
plt.gca().yaxis.set_label_coords(-0.08, 0.5)
plt.gca().yaxis.set_tick_params(rotation = 0)
plt.subplots_adjust(top = 0.9, bottom = 0.15)
plt.gca().set_axisbelow(True)
# plt.title('seed = {}'.format(SEED))
folder_path = "data"
os.makedirs(folder_path, exist_ok = True)
current_time = datetime.datetime.now().strftime("%Y_%m%d_%H%M_%S")
filename = f"softmax_1_{current_time}.svg"
file_path = os.path.join(folder_path, filename)
plt.savefig(file_path, format = 'svg')
plt.show()
