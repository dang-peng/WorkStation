import pickle

import matplotlib.pyplot as plt

filename = "./weights_biases.pickle"
with open(filename, 'rb') as file:
    (weights, biases) = pickle.load(file)
# 生成模拟数据
# w = np.random.normal(0, 0.1, (784, 500))

# 绘制直方图
plt.hist(weights[2].ravel(), bins = 50, color = 'green', alpha = 0.5)
# plt.hist(biases[2].ravel(), bins = 50, color = 'green', alpha = 0.5)

# plt.text(2, 0.8, r'$\pi=100$', fontsize=14.0)
# 添加图标题和坐标轴标签
plt.title('Weight Distribution')
plt.xlabel('Value')
# plt.xlabel('Value 'r'$\pi=100$')
# plt.xlim(-0.7, 0.7)
# plt.xlim(-2.2, 1.2)
plt.ylabel('Frequency')

# 显示图像
plt.show()
