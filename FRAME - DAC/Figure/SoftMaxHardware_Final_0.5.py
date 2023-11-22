import datetime
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis = 1, keepdims = True)


# 10 20 70 100
SEED = 10
# SEED = 100
np.random.seed(SEED)
rows = 300
cols = 1
K = 1.380649e-23
T = 300
delta_f = 1e10
Vr = 0.05
N = 4 * K * T * delta_f

# Vth = 0.065
Vth = 0.15
# Vth = 0.1 # Vth = 0.1
Weights = np.random.normal(0, 1, (rows, 10))
Wmax = np.max(Weights)
Wmin = np.min(Weights)
R_TIA = 100e3
resistanceOn = 240e3
resistanceOff = 240e3 * 100
Gmax = 1 / resistanceOn
Gmin = 1 / resistanceOff
# Wmax = 2.7
# Wmin = -3.3
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
    print("num_samples:", num_samples)
yb_prob_experiment = yb_sum / num_experiment
yb_prob_sum = np.sum(yb_prob_experiment, axis = 1)
yb_prob = softmax(I0 / (Vr * G0))
yb_prob_list = yb_prob.flatten().tolist()
yb_prob_index_list = list(range(len(yb_prob_list)))
yb_prob_experiment_list = yb_prob_experiment.flatten().tolist()
yb_prob_experiment_index_list = list(range(len(yb_prob_experiment_list)))
colors = ['#77B84A', '#9EC147', '#CAD239', '#F2EA2B', '#F1B627', '#DE602B', '#479CD1', '#00A4B9', '#82629F', '#DA6969']
# colors = ['#2CA02C', '#9EC147', '#CAD239', '#F1B627', '#DA6969', '#82629F', '#00A4B9', '#479CD1', '#EC8B2D',
# '#D32727']
fig, ax = plt.subplots()  # 使用sharey参数共享纵坐标轴
width = 0.4
bars1 = ax.bar(np.arange(len(yb_prob_experiment_index_list)) - width / 2, yb_prob_experiment_list, width,
               label = 'Experimental',
               color = colors)
ax.set_xlim(-1, 10)
ax.set_yscale("log")
ax.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
ax.set_axisbelow(True)
bars2 = ax.bar([i + width / 2 for i in np.arange(len(yb_prob_index_list))], yb_prob_list, width, label = 'SoftMax',
               color = colors, edgecolor = 'black', linewidth = 1)
# color = colors2, hatch = '///')
# 将填充颜色设置为较浅的颜色
for bar in bars2:
    rgba = bar.get_facecolor()  # 获取当前的填充颜色
    rgba_light = (*rgba[:3], 0.7)  # 将 alpha 设置为 0.3，保持其他颜色不变
    bar.set_facecolor(rgba_light)  # 设置柱形的填充颜色
ax.set_yscale("log")
ax.set_xticks(np.arange(10))
plt.xlabel('Index', fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.ylabel('Probability', fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.xticks(fontname = "Times New Roman", fontsize = 14)
plt.yticks(fontname = "Times New Roman", fontsize = 14)
plt.legend(prop = {'family': 'Times New Roman'}, loc = 'upper right')
# plt.text(0.05, 0.95, 'Vth = {} V'.format(Vth) + '\nR_TIA = ' + '{:.0e} Ω'.format(R_TIA),
plt.text(0.03, 0.965, 'Vr = ' + '{} V'.format(Vr) + '\nVth = {} V'.format(Vth),
         transform = plt.gca().transAxes,
         verticalalignment = 'top', horizontalalignment = 'left',
         fontsize = 12, bbox = dict(facecolor = 'white', edgecolor = 'lightgray', boxstyle = 'round'),
         fontproperties = FontProperties(family = 'Times New Roman', size = 12))
ax.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
ax.set_axisbelow(True)
plt.title('seed = {}'.format(SEED))
folder_path = "data"
os.makedirs(folder_path, exist_ok = True)
current_time = datetime.datetime.now().strftime("%Y_%m%d_%H%M_%S")
filename = f"softmax_{current_time}.svg"
file_path = os.path.join(folder_path, filename)
plt.savefig(file_path, format = 'svg')
plt.show()
