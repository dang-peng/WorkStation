import datetime
import os

import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子，以确保可复现性
np.random.seed(0)

# 定义噪声信号的参数
duration = 1.0  # 信号持续时间（秒）
sampling_rate = 1000  # 采样率（每秒采样点数）
num_samples = int(duration * sampling_rate)  # 总采样点数

# 生成随机噪声信号
noise = np.random.normal(0, 1, num_samples)

# 计算时间数组
time = np.linspace(0, duration, num_samples)

# 绘制波形图
plt.figure(figsize = (8, 4))
plt.plot(time, noise)
# plt.xlabel('Time (s)')
# plt.ylabel('Current')
# plt.title('Current Noise Waveform')
plt.grid(True)
plt.xticks([])
plt.yticks([])

# 去掉坐标框
plt.box(False)
folder_path = "data"
os.makedirs(folder_path, exist_ok = True)
current_time = datetime.datetime.now().strftime("%Y_%m%d_%H%M")
filename = f"Noise_Filter_{current_time}.svg"
file_path = os.path.join(folder_path, filename)
plt.savefig(file_path, format = 'svg')
plt.show()
