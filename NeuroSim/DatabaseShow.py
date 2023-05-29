import matplotlib.pyplot as plt

from MNIST import (X_Tr, Y_Tr, X_Te)

X_Train = X_Tr
Y_Train = Y_Tr
X_Test = X_Te
fig, ax = plt.subplots(
    nrows = 2,
    ncols = 5,
    sharex = True,
    sharey = True, )
ax = ax.flatten()
for i in range(10):
    img = X_Train[Y_Train == i][0].reshape(28, 28)  # 通过将布尔索引应用于X_Train, 可以获取与标签i相等的样本的特征向量
    # ax[i].imshow(img, cmap = 'Greys', interpolation = 'nearest')
    # color image
    ax[i].imshow(img, interpolation = 'nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

# plt1 = plt.figure()
# for i in range(10):
#     im_Train = X_Train[i].reshape(28, 28)  # 训练数据集的第i张图，将其转化为28x28格式
#     # im=batch_xs[i].reshape(28,28)	#该批次的第i张图
#     # plt1.imshow(im_Train, cmap = 'Greys', interpolation = 'nearest')
#     plt.pause(0.1)  # 暂停时间
# # im = X_Train[59999].reshape(28, 28)  # 训练数据集的第i张图，将其转化为28x28格式
# # im=batch_xs[i].reshape(28,28)	#该批次的第i张图
# plt1.imshow(im_Train)
# plt1.show()
# plt = plt.figure()
# for i in range(1):
#     im_Test = X_Test[i].reshape(28, 28)  # 训练数据集的第i张图，将其转化为28x28格式
#     # im=batch_xs[i].reshape(28,28)	#该批次的第i张图
#     plt.imshow(im_Test)
#     plt.pause(0.1)  # 暂停时间
# plt.show()
