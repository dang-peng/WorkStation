import datetime
import os

import matplotlib.pyplot as plt
import numpy as np

# ReLU(x) = max(0, x)
# 创建 x 值的范围
x = np.linspace(-10, 10, 100)

# 计算 sigmoid 函数的 y 值
y = np.maximum(0, x)

# 显示图表
plt.show()
l4 = plt.plot(x, y, '#1F77B4', linewidth = 3, label = 'ReLU')
plt.legend(prop = {'family': 'Times New Roman'}, loc = 'upper left')
plt.xlim(-8, 8)
# plt.title("sigmoid_1")
plt.xlabel('x', fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.ylabel('y', fontname = "Times New Roman", fontsize = 14, color = 'black')
plt.xticks(fontname = "Times New Roman", fontsize = 14)
plt.yticks(fontname = "Times New Roman", fontsize = 14)
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
# plt.axvline(x=0, color='black', linestyle='--')
folder_path = "data"
os.makedirs(folder_path, exist_ok = True)
current_time = datetime.datetime.now().strftime("%Y_%m%d_%H%M")
filename = f"ReLU_{current_time}.svg"
file_path = os.path.join(folder_path, filename)
plt.savefig(file_path, format = 'svg')
# plt.savefig('output.svg', format = 'svg')
plt.show()
