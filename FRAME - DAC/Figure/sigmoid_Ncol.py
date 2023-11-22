import datetime
import os

import numpy as np
from matplotlib import pyplot as plt

# np.random.seed(0)
rows1 = 50
rows2 = 50
rows3 = 50
cols = 1

K = 1.380649e-23
T = 300
delta_f = 1e10
N1 = 4 * K * T * delta_f
N2 = 4 * K * T * delta_f
N3 = 4 * K * T * delta_f
# Vr = 0.035
Vr = 0.027
resistanceOn = 240e3
resistanceOff = 240e3 * 100
Gmax = 1 / resistanceOn
Gmin = 1 / resistanceOff
Wmax = 1
Wmin = -1
Gref = (Wmax * Gmin - Wmin * Gmax) / (Wmax - Wmin)
G0 = (Gmax - Gmin) / (Wmax - Wmin)
a1 = Vr * G0 / np.sqrt(4 * N1 * rows1 * Gref)
Vr1 = a1 * np.sqrt(4 * N1 * rows1 * Gref) / G0
print("a1 = ", a1)
print("Vr1 = ", Vr1)
a2 = Vr * G0 / np.sqrt(4 * N2 * rows2 * Gref)
Vr2 = a2 * np.sqrt(4 * N2 * rows2 * Gref) / G0
print("a2 = ", a2)
print("Vr2 = ", Vr2)
a3 = Vr * G0 / np.sqrt(4 * N3 * rows3 * Gref)
Vr3 = a3 * np.sqrt(4 * N3 * rows3 * Gref) / G0
print("a3 = ", a3)
print("Vr3 = ", Vr3)
z_list = []
Y_list = []
yb1_list = []
yb2_list = []
yb3_list = []
Ncol1 = 0
Ncol2 = 0
Ncol3 = 0
num_experiment = 1000
for n in range(100):
    X1 = np.random.randint(2, size = (rows1, cols))
    V1 = np.array(X1) * Vr1
    V2 = np.tile(V1, reps = (4, 1))
    V3 = np.tile(V1, reps = (20, 1))
    Ncol1 = V1.shape[0]
    Ncol2 = V2.shape[0]
    Ncol3 = V3.shape[0]
    Weights1 = np.random.uniform(-1, 1, size = (rows1, cols))
    G_Array1 = Weights1 * G0 + Gref
    G_Array1 = np.clip(G_Array1, Gmin, Gmax)
    Gref_Col1 = np.ones((rows1, 1)) * Gref
    G_Array2 = np.tile(G_Array1, reps = (4, 1))
    Gref_Col2 = np.tile(Gref_Col1, reps = (4, 1))
    G_Array3 = np.tile(G_Array1, reps = (20, 1))
    Gref_Col3 = np.tile(Gref_Col1, reps = (20, 1))
    I1 = np.dot(V1.T, G_Array1)
    Iref1 = np.dot(V1.T, Gref_Col1)
    I2 = np.dot(V2.T, G_Array2)
    Iref2 = np.dot(V2.T, Gref_Col2)
    I3 = np.dot(V3.T, G_Array3)
    Iref3 = np.dot(V3.T, Gref_Col3)
    yb1 = 0
    yb2 = 0
    yb3 = 0
    for i in range(num_experiment):
        I_devices_noise = G_Array1.copy()
        for G_device in np.nditer(I_devices_noise, op_flags = ['readwrite']):
            device_I_noise = np.random.normal(0, np.sqrt(N1 * G_device), (1, 1))
            G_device[...] = device_I_noise
        I_noise = np.sum(I_devices_noise, axis = 0)
        Iref_device_noise = G_Array1.copy()
        for G_ref in np.nditer(Iref_device_noise, op_flags = ['readwrite']):
            deviceRef_noise = np.random.normal(0, np.sqrt(N1 * G_ref), (1, 1))
            G_ref[...] = deviceRef_noise
        Iref_noise = np.sum(Iref_device_noise, axis = 0)
        if (I1 + I_noise) > (Iref1 + Iref_noise):
            yb1 += 1
        I_devices_noise_2 = G_Array2.copy()
        for G_device in np.nditer(I_devices_noise_2, op_flags = ['readwrite']):
            device_I_noise_2 = np.random.normal(0, np.sqrt(N2 * G_device), (1, 1))
            G_device[...] = device_I_noise_2
        I_noise_2 = np.sum(I_devices_noise_2, axis = 0)
        Iref_device_noise_2 = G_Array2.copy()
        for G_ref in np.nditer(Iref_device_noise_2, op_flags = ['readwrite']):
            deviceRef_noise_2 = np.random.normal(0, np.sqrt(N2 * G_ref), (1, 1))
            G_ref[...] = deviceRef_noise_2
        Iref_noise_2 = np.sum(Iref_device_noise_2, axis = 0)
        if (I2 + I_noise_2) > (Iref2 + Iref_noise_2):
            yb2 += 1
        I_devices_noise_3 = G_Array3.copy()
        for G_device in np.nditer(I_devices_noise_3, op_flags = ['readwrite']):
            device_I_noise_3 = np.random.normal(0, np.sqrt(N3 * G_device), (1, 1))
            G_device[...] = device_I_noise_3
        I_noise_3 = np.sum(I_devices_noise_3, axis = 0)
        Iref_device_noise_3 = G_Array3.copy()
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
    # z = (I1 - Iref1) / (Vr1 * G0)
    # y = 1.0 / (1.0 + np.exp(-z))
    z = (I1 - Iref1) * 1e6
    y = 1.0 / (1.0 + np.exp(-z / (Vr1 * G0 * 1e6)))
    z_list.append(z[0][0])
    Y_list.append(y[0][0])
# 创建一个指定尺寸的图形对象
plt.rcParams['mathtext.fontset'] = 'stix'
fig = plt.figure(figsize = (8, 6))
Solid_Type = False
if not Solid_Type:
    l1 = plt.scatter(z_list, yb2_list, s = 50, label = r'$\mathrm{N}_\mathrm{col}$'+' = ' + str(Ncol2), marker = 'o', linewidths = 2)
    l2 = plt.scatter(z_list, yb3_list, s = 50, label = r'$\mathrm{N}_\mathrm{col}$'+' = ' + str(Ncol3), marker = 'o', linewidths = 2)
    l3 = plt.scatter(z_list, yb1_list, s = 50, label = r'$\mathrm{N}_\mathrm{col}$'+' = ' + str(Ncol1), marker = 'o', linewidths = 2)
else:
    l1 = plt.scatter(z_list, yb1_list, color = 'g', s = 50, label = 'Ncol = ' + str(Ncol1))
    l2 = plt.scatter(z_list, yb2_list, color = 'b', s = 50, label = 'Ncol = ' + str(Ncol2))
    l3 = plt.scatter(z_list, yb3_list, color = 'y', s = 50, label = 'Ncol = ' + str(Ncol3))
Ylist = sorted(Y_list)
zlist = sorted(z_list)
l4 = plt.plot(zlist, Ylist, 'r', linewidth = 4, label = 'Logistic')
plt.legend(prop = {'family': 'Times New Roman', 'size': 20}, loc = 'upper left')
plt.xlim(-0.4, 0.4)
# plt.title("sigmoid_1")
plt.xlabel('Mean of EPSC,  ' + r"$\overline{I_j}$" + ' - ' + r"$\overline{I_{ref}}$" + '    [\u03BCA]',
           fontname = "Times New Roman", fontsize = 22, color = 'black')
plt.ylabel('Probability', fontname = "Times New Roman", fontsize = 22, color = 'black')
plt.xticks(fontname = "Times New Roman", fontsize = 22)
plt.yticks(fontname = "Times New Roman", fontsize = 22)
# plt.title('y vs z', fontsize = 22, color = 'black')
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray', zorder = 0)
# 调整图像的布局
plt.subplots_adjust(top = 0.92, bottom = 0.145)
# plt.subplots_adjust(left=0.2, right=0.9)
folder_path = "data"
os.makedirs(folder_path, exist_ok = True)
current_time = datetime.datetime.now().strftime("%Y_%m%d_%H%M")
filename = f"sigmoid_Ncol_{current_time}.svg"
file_path = os.path.join(folder_path, filename)
plt.savefig(file_path, format = 'svg')
# plt.savefig('output.svg', format = 'svg')
plt.show()
