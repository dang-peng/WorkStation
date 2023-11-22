import datetime
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

from InputData import data

# 10 20 70 100 1
np.random.seed(10)

K = 1.380649e-23
T = 300
delta_f = 1e10
Vr = 0.05
N1 = 4 * K * T * delta_f * 10
N = 4 * K * T * delta_f
N_softmax = N1
Vth = 1.6
R_TIA = 100e3
resistanceOn = 240e3
resistanceOff = 240e3 * 100
Gmax = 1 / resistanceOn
Gmin = 1 / resistanceOff
Wmax = 1
Wmin = -1
Gref = (Wmax * Gmin - Wmin * Gmax) / (Wmax - Wmin)
G0 = (Gmax - Gmin) / (Wmax - Wmin)
a1 = Vr * G0 / np.sqrt(2 * N1 * 100 * Gref)
Vr1 = a1 * np.sqrt(2 * N * 500 * Gref) / G0
print("a1 = ", a1)
print("Vr1 = ", Vr1)
num_experiment = 100
X = data.input
V = np.array(X) * Vr

Weights = data.weights
G_Array = Weights * G0 + Gref
G_Array = np.clip(G_Array, Gmin, Gmax)
I0 = np.dot(V.T, G_Array)


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis = 1, keepdims = True)


def SoftMax(I, G_Array, Gref):
    yb_sum = np.zeros_like(I)
    for i in range(num_experiment):
        yb = []
        y = 0
        num_samples = 0
        while y == 0 and num_samples < 1000:
            yb[:] = np.zeros_like(yb)
            # Array_noise = G_Array.copy()
            # for m in range(Array_noise.shape[0]):
            #     for n in range(Array_noise.shape[1]):
            #         Array_noise[m, n] = np.random.normal(0, np.sqrt(N1 * Array_noise[m, n]), (1, 1))
            Array_noise = np.random.normal(0, np.sqrt(N_softmax * Gref), size = G_Array.shape)
            I_noise = np.sum(Array_noise, axis = 0)
            I_total = I + I_noise
            V = R_TIA * I_total
            yb = np.where(V > Vth, 1, 0)
            y = np.sum(yb, axis = 1)
            num_samples += 1
        yb_sum = yb_sum + yb
    return yb_sum / num_experiment


def run():
    yb_prob = softmax(I0 / (Vr * G0))
    yb_prob_list = yb_prob.flatten().tolist()
    yb_prob_experiment = SoftMax(I0, G_Array, Gref)
    yb_prob_sum = np.sum(yb_prob_experiment, axis = 1)
    yb_prob_index_list = list(range(len(yb_prob_list)))
    yb_prob_experiment_list = yb_prob_experiment.flatten().tolist()
    yb_prob_experiment_index_list = list(range(len(yb_prob_experiment_list)))
    colors1 = ['#F33B27', 'green', '#2070AA', '#EC7728', 'purple']
    colors2 = ['#E77272', '#4FAF4F', '#479CD1', '#ECA156', '#AE8CC3']
    fig, ax = plt.subplots()  # 使用sharey参数共享纵坐标轴
    width = 0.4
    ax.bar(np.arange(len(yb_prob_experiment_index_list)) - width / 2, yb_prob_experiment_list, width,
           label = 'Vth = ' + str(Vth),
           color = colors2)
    ax.set_xlim(-1, 10)
    ax.set_yscale("log")
    ax.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
    ax.bar([i + width / 2 for i in np.arange(len(yb_prob_index_list))], yb_prob_list, width, label = 'SoftMax',
           color = colors1)
    # color = colors2, hatch = '///')
    ax.set_yscale("log")
    ax.set_xticks(np.arange(10))
    plt.xlabel('Index', fontname = "Times New Roman", fontsize = 14, color = 'black')
    plt.ylabel('Probability', fontname = "Times New Roman", fontsize = 14, color = 'black')
    plt.xticks(fontname = "Times New Roman", fontsize = 14)
    plt.yticks(fontname = "Times New Roman", fontsize = 14)
    # plt.text(0.05, 0.95, 'Vth = {} V'.format(Vth) + '\nR_TIA = ' + '{:.0e} Ω'.format(R_TIA),
    plt.text(0.05, 0.95, 'Vr = ' + '{} V'.format(Vr) + '\nVth = {} V'.format(Vth),
             transform = plt.gca().transAxes,
             verticalalignment = 'top', horizontalalignment = 'left',
             fontsize = 12, bbox = dict(facecolor = 'white', edgecolor = 'lightgray', boxstyle = 'round'),
             fontproperties = FontProperties(family = 'Times New Roman', size = 12))
    ax.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
    folder_path = "data"
    os.makedirs(folder_path, exist_ok = True)
    current_time = datetime.datetime.now().strftime("%Y_%m%d_%H%M_%S")
    filename = f"softmax_{current_time}.svg"
    file_path = os.path.join(folder_path, filename)
    plt.savefig(file_path, format = 'svg')
    plt.show()


if __name__ == '__main__':
    run()
