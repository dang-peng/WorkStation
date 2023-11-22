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
z_list = []
Y_list = []
yb1_list = []
yb2_list = []
yb3_list = []

num_experiment = 1000

for n in range(110):
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
    yb1 = 0
    yb2 = 0
    yb3 = 0
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
            yb1 += 1
        I_devices_noise_2 = G_Array.copy()
        for G_device in np.nditer(I_devices_noise_2, op_flags = ['readwrite']):
            device_I_noise_2 = np.random.normal(0, np.sqrt(N2 * G_device), (1, 1))
            G_device[...] = device_I_noise_2
        I_noise_2 = np.sum(I_devices_noise_2, axis = 0)
        Iref_device_noise_2 = G_Array.copy()
        for G_ref in np.nditer(Iref_device_noise_2, op_flags = ['readwrite']):
            deviceRef_noise_2 = np.random.normal(0, np.sqrt(N2 * G_ref), (1, 1))
            G_ref[...] = deviceRef_noise_2
        Iref_noise_2 = np.sum(Iref_device_noise_2, axis = 0)
        if (I2 + I_noise_2) > (Iref2 + Iref_noise_2):
            yb2 += 1
        I_devices_noise_3 = G_Array.copy()
        for G_device in np.nditer(I_devices_noise_3, op_flags = ['readwrite']):
            device_I_noise_3 = np.random.normal(0, np.sqrt(N3 * G_device), (1, 1))
            G_device[...] = device_I_noise_3
        I_noise_3 = np.sum(I_devices_noise_3, axis = 0)
        Iref_device_noise_3 = G_Array.copy()
        for G_ref in np.nditer(Iref_device_noise_3, op_flags = ['readwrite']):
            deviceRef_noise_3 = np.random.normal(0, np.sqrt(N3 * G_ref), (1, 1))
            G_ref[...] = deviceRef_noise_3
        Iref_noise_3 = np.sum(Iref_device_noise_3, axis = 0)
        if (I3 + I_noise_3) > (Iref3 + Iref_noise_3):
            yb3 += 1
    # print("Times: {}  ".format(n))
    
    yb1_list.append(yb1 / num_experiment)
    yb2_list.append(yb2 / num_experiment)
    yb3_list.append(yb3 / num_experiment)
    z = (I1 - Iref1) * 1e6
    # z = (I1 - Iref1) / (Vr1 * G0)
    y = 1.0 / (1.0 + np.exp(-z / (Vr1 * G0 * 1e6)))
    z_list.append(z[0][0])
    Y_list.append(y[0][0])

Vr1 = "{:.2f}".format(Vr1)
Vr2 = "{:.2f}".format(Vr2)
Vr3 = "{:.2f}".format(Vr3)
# 创建一个指定尺寸的图形对象
plt.rcParams['mathtext.fontset'] = 'stix'
fig = plt.figure(figsize = (8, 6))
Solid_Type = False
if not Solid_Type:
    l1 = plt.scatter(z_list, yb1_list, s = 50, label = r'$\mathrm{V}_\mathrm{r}$' + ' = ' + str(Vr1) + ' V',
                     marker = 'o', linewidths = 2)
    l2 = plt.scatter(z_list, yb2_list, s = 50, label = r'$\mathrm{V}_\mathrm{r}$' + ' = ' + str(Vr2) + ' V',
                     marker = 'o', linewidths = 2)
    l3 = plt.scatter(z_list, yb3_list, s = 50, label = r'$\mathrm{V}_\mathrm{r}$' + ' = ' + str(Vr3) + ' V',
                     marker = 'o', linewidths = 2)
else:
    l1 = plt.scatter(z_list, yb1_list, color = 'g', s = 50, label = 'Vr = ' + str(Vr1) + ' V')
    l2 = plt.scatter(z_list, yb2_list, color = 'b', s = 50, label = 'Vr = ' + str(Vr2) + ' V')
    l3 = plt.scatter(z_list, yb3_list, color = 'y', s = 50, label = 'Vr = ' + str(Vr3) + ' V')

Ylist = sorted(Y_list)
zlist = sorted(z_list)
l4 = plt.plot(zlist, Ylist, 'r', linewidth = 4, label = 'Logistic')
plt.legend(prop = {'family': 'Times New Roman', 'size': 20}, loc = 'upper left')
plt.xlim(-1.5, 1.5)
# plt.title("sigmoid_1")
plt.xlabel('Mean of EPSC,  ' + r"$\overline{I_j}$" + ' - ' + r"$\overline{I_{ref}}$" + '    [\u03BCA]',
           fontname = "Times New Roman", fontsize = 22, color = 'black')
plt.ylabel('Probability', fontname = "Times New Roman", fontsize = 22, color = 'black')
plt.xticks(fontname = "Times New Roman", fontsize = 22)
plt.yticks(fontname = "Times New Roman", fontsize = 22)
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray', zorder = 0)
# 调整图像的布局
plt.subplots_adjust(top = 0.92, bottom = 0.145)
# plt.subplots_adjust(left=0.2, right=0.9)
folder_path = "data"
os.makedirs(folder_path, exist_ok = True)
current_time = datetime.datetime.now().strftime("%Y_%m%d_%H%M")
filename = f"sigmoid_Vr_{current_time}.svg"
file_path = os.path.join(folder_path, filename)
plt.savefig(file_path, format = 'svg')
# plt.savefig('output.svg', format = 'svg')
plt.show()
