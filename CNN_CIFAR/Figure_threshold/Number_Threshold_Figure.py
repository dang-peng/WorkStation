import datetime
import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

filename = "../TrainCumulativeThreshold/CNN_train_data_03_11_07_43.pickle"
with open(filename, 'rb') as file:
    (updates_num, Loss, Test_Accuracy) = pickle.load(file)
# Loss = [np.mean(Loss[i:i+500]) for i in range(0, len(Loss), 500)]
epochs = list(range(301))
Test_Accuracy = np.array(Test_Accuracy) + 1
threshold = np.arange(0, 101 * 0.01, 0.01)

# 创建图和主轴对象
plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.size'] = 18
fig, ax1 = plt.subplots(figsize = (8, 6))
# 绘制第一条曲线，使用主轴
ax1.plot(threshold, updates_num, color = '#6CB647', label = "Updates Number", linewidth = 3)
ax1.set_xlabel('Threshold', fontname = "Times New Roman", fontsize = 22, color = 'black')
ax1.set_ylabel('Updates Number', fontname = "Times New Roman", fontsize = 22, color = 'black')
ax1.tick_params(axis = 'y', labelcolor = 'black', labelsize = 20)
ax1.tick_params(axis = 'x', labelcolor = 'black', labelsize = 20)
ax1.set_yscale('log')
ax1.set_xlim(0, 1)
# ax1.set_xticks(np.arange(0, 3, 0.5))
ax1.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')

# 创建与主轴共享横坐标的次轴
ax2 = ax1.twinx()
# 绘制第二条曲线（例如，测试准确率），使用次轴
ax2.plot(threshold, Test_Accuracy_5, color = '#2070AA', label = "Test Accuracy", linewidth = 3)
ax2.set_ylabel('Accuracy', fontname = "Times New Roman", fontsize = 22, color = 'black')
ax2.tick_params(axis = 'y', labelcolor = 'black', labelsize = 20)
# ax2.tick_params(axis = 'x', labelcolor = ' black', labelsize = 16)
# 可以选择添加图例
# ax1.legend(loc = 'upper left', prop = {'size': 18, 'family': 'Times New Roman'})
# ax2.legend(loc = 'upper right', prop = {'size': 18, 'family': 'Times New Roman'})
# 创建图例项，这里我们需要手动指定颜色和标签
legend_elements = [Line2D([0], [0], color = '#6CB647', lw = 3, label = 'Updates Number'),
                   Line2D([0], [0], color = '#2070AA', lw = 3, label = 'Test_Accuracy')]
# 使用fig.legend()添加统一的图例
fig.legend(handles = legend_elements, loc = 'upper right', bbox_to_anchor = (0.88, 0.9),
           prop = {'size': 18, 'family': 'Times New Roman'})
# 调整图布局
plt.subplots_adjust(top = 0.9, bottom = 0.13, left = 0.13, right = 0.88)
folder_path = "./data"
os.makedirs(folder_path, exist_ok = True)
current_time = datetime.datetime.now().strftime("%Y_%m%d")
filename = f"Test_Accuracy_num_threshold_{current_time}.svg"
file_path = os.path.join(folder_path, filename)
plt.savefig(file_path, format = 'svg')
# plt.savefig('output.svg', format = 'svg')
plt.show()
