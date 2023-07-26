import datetime
import os

import matplotlib.pyplot as plt

from GetDataset import GetDataset

train_images, train_labels, test_images, test_labels, batches = GetDataset(100).get_data()
X_Train = train_images
Y_Train = train_labels


fig, ax = plt.subplots(figsize = (6, 6))
img = X_Train[Y_Train == 2][0].reshape(28, 28)  # 通过将布尔索引应用于X_Train, 可以获取与标签i相等的样本的特征向量
# ax[i].imshow(img, cmap = 'Greys', interpolation = 'nearest')
# color image
ax.imshow(img, interpolation = 'nearest')
ax.set_xticks([])
ax.set_yticks([])

folder_path = "./data"
os.makedirs(folder_path, exist_ok = True)
current_time = datetime.datetime.now().strftime("%Y_%m%d_%H%M%S")
filename = f"MNIST_{current_time}.svg"
file_path = os.path.join(folder_path, filename)
plt.savefig(file_path, format = 'svg')
plt.show()
plt.show()

# plt1 = plt.figure()
# for i in range(10):
#     im_Train = X_Train[0][i].reshape(28, 28)  # 训练数据集的第i张图，将其转化为28x28格式
# im=batch_xs[i].reshape(28,28)	#该批次的第i张图
# plt1.imshow(im_Train, cmap = 'Greys', interpolation = 'nearest')
# plt.pause(0.1)  # 暂停时间
# im = X_Train[59999].reshape(28, 28)  # 训练数据集的第i张图，将其转化为28x28格式
# im=batch_xs[i].reshape(28,28)	#该批次的第i张图
#     plt1.imshow(im_Train)
# plt1.show()
# plt = plt.figure()
# for i in range(1):
#     im_Test = X_Train[i][0].reshape(28, 28)  # 训练数据集的第i张图，将其转化为28x28格式
#     # im=batch_xs[i].reshape(28,28)	#该批次的第i张图
#     plt.imshow(im_Test)
#     plt.pause(0.1)  # 暂停时间
# plt.show()
