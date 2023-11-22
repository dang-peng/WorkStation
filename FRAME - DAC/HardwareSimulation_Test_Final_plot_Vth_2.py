import datetime
import os
import pickle

from matplotlib import pyplot as plt

Test_times = 101
Vth1 = 0.1
Vth2 = 0.2
Vth3 = 0.25
filename = "inference_data_Vth.pickle"
with open(filename, 'rb') as file:
    (TestNumber, SoftwareInferenceTest, SoftwareBinaryTest, UsingHardwareTest4, UsingHardwareTest5,
     UsingHardwareTest6) = pickle.load(file)
for i in range(Test_times):
    # print("{}th Testing...".format(i))
    TestNumber.append(i)
plt.rcParams['mathtext.fontset'] = 'stix'
# plt.plot(TestNumber, SoftwareInferenceTest, linewidth = 3, label = 'Software Inference')
plt.plot(TestNumber, SoftwareBinaryTest, linewidth = 3, label = 'Binary Software Inference')
plt.plot(TestNumber, UsingHardwareTest4, linewidth = 3,
         label = 'Hardware Inference ' + r'$\mathrm{V}_\mathrm{th0}$' + ' = {} V'.format(Vth1))
plt.plot(TestNumber, UsingHardwareTest5, linewidth = 3,
         label = 'Hardware Inference ' + r'$\mathrm{V}_\mathrm{th0}$' + ' = {} V'.format(Vth2))
plt.plot(TestNumber, UsingHardwareTest6, linewidth = 3,
         label = 'Hardware Inference ' + r'$\mathrm{V}_\mathrm{th0}$' + ' = {} V'.format(Vth3))
plt.legend(prop = {'size': 16, 'family': 'Times New Roman'}, loc = 'lower right')
plt.xlabel('Test Times', fontname = "Times New Roman", fontsize = 22, color = 'black')
plt.ylabel('Accuracy', fontname = "Times New Roman", fontsize = 22, color = 'black')
plt.xticks(fontname = "Times New Roman", fontsize = 20)
plt.yticks(fontname = "Times New Roman", fontsize = 20)
plt.xlim(0, Test_times - 1)
plt.ylim(30, 100)
plt.subplots_adjust(top = 0.94, bottom = 0.148)
plt.subplots_adjust(left = 0.15, right = 0.93)
# plt.ylim(60, 100)
# plt.title('Test Accuracy')
plt.grid(True, linestyle = '--', dashes = (3, 3), linewidth = 0.5, color = 'gray')
folder_path = "./data"
os.makedirs(folder_path, exist_ok = True)
current_time = datetime.datetime.now().strftime("%Y_%m%d_%H%M")
filename = f"Test_Accuracy_{current_time}.svg"
file_path = os.path.join(folder_path, filename)
plt.savefig(file_path, format = 'svg')
# plt.savefig('output.svg', format = 'svg')
plt.show()
