import datetime
import os
import pickle

from matplotlib import pyplot as plt

filename = "../TrainCumulativeThreshold/CNN_train_data_03_11_07_43.pickle"
with open(filename, 'rb') as file:
    (updates_num, Loss, Test_Accuracy) = pickle.load(file)
# Loss = [np.mean(Loss[i:i+500]) for i in range(0, len(Loss), 500)]

epochs = list(range(101))
# 绘制loss和epoch关系图
plt.figure(figsize = (8, 6))
plt.plot(epochs, Loss, color = '#F33B27',
         label = "baseline", linewidth = 3)
# plt.title('Epochs vs Loss', fontname = "Times New Roman", fontsize = 22, color = 'black', pad = 15)
plt.legend(prop = {'size': 14, 'family': 'Times New Roman'}, loc = 'upper right')
plt.xlabel('Epochs', fontname = "Times New Roman", fontsize = 22, color = 'black')
plt.ylabel('Loss', fontname = "Times New Roman", fontsize = 22, color = 'black')
plt.xticks(fontname = "Times New Roman", fontsize = 18)
plt.yticks(fontname = "Times New Roman", fontsize = 18)
plt.xlim(-1, 100)
# plt.ylim(-30, 30)
plt.subplots_adjust(top = 0.9, bottom = 0.13)
plt.subplots_adjust(left = 0.12, right = 0.92)
# plt.ylim(60, 100)
# plt.title('Test Accuracy')
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
folder_path = "./data"
os.makedirs(folder_path, exist_ok = True)
current_time = datetime.datetime.now().strftime("%Y_%m%d")
filename = f"Loss_epoch_1_{current_time}.svg"
file_path = os.path.join(folder_path, filename)
plt.savefig(file_path, format = 'svg')
# plt.savefig('output.svg', format = 'svg')
plt.show()
# 绘制loss和epoch关系图
plt.figure(figsize = (8, 6))
plt.plot(epochs, Test_Accuracy, color = '#F33B27',
         label = "baseline", linewidth = 3)
# plt.title('Epochs vs Accuracy', fontname = "Times New Roman", fontsize = 22, color = 'black', pad = 15)
plt.legend(prop = {'size': 14, 'family': 'Times New Roman'}, loc = 'lower right')
plt.xlabel('Epochs', fontname = "Times New Roman", fontsize = 22, color = 'black')
plt.ylabel('Accuracy', fontname = "Times New Roman", fontsize = 22, color = 'black')
plt.xticks(fontname = "Times New Roman", fontsize = 18)
plt.xlim(-1, 100)
plt.yticks(fontname = "Times New Roman", fontsize = 18)
plt.subplots_adjust(top = 0.9, bottom = 0.13)
plt.subplots_adjust(left = 0.12, right = 0.92)
# plt.ylim(60, 100)
# plt.title('Test Accuracy')
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
folder_path = "./data"
os.makedirs(folder_path, exist_ok = True)
current_time = datetime.datetime.now().strftime("%Y_%m%d")
filename = f"Test_Accuracy_1_{current_time}.svg"
file_path = os.path.join(folder_path, filename)
plt.savefig(file_path, format = 'svg')
# plt.savefig('output.svg', format = 'svg')
plt.show()
