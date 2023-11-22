import datetime
import os

import matplotlib.pyplot as plt
import numpy as np

# 生成输入信号
t = np.linspace(0, 1, 1000)
f = 2  # 频率为5Hz
input_signal = np.sin(2 * np.pi * f * t)

# 生成噪声
noise = np.random.normal(0, 0.15, len(input_signal))

# 加噪声
noisy_signal = input_signal + noise

# 绘制图像
# plt.plot(t, input_signal, label = 'Input')
plt.plot(t, noisy_signal, label = 'Noisy')
plt.axis('off')
# ax.set_title('Smoothed Sin Waveform')
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Amplitude')
folder_path = "data"
os.makedirs(folder_path, exist_ok = True)
current_time = datetime.datetime.now().strftime("%Y_%m%d_%H%M")
filename = f"Noise_Filter_{current_time}.svg"
file_path = os.path.join(folder_path, filename)
plt.savefig(file_path, format = 'svg')
plt.show()
