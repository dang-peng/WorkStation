import datetime
import os

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Helvetica'
# 数据
labels = ['Driver', 'ADC', 'Activation', 'Crossbar']
sizes1 = [4, 50.6, 14.3]
sizes2 = [4.09, 212.4, 42.81, 11.59]
colors = ['#DAE56E', '#AACBE9', '#F8C4C9', '#8BC34A']
colors2 = ['#DAE56E', '#AACBE9', '#82CA9D', '#F8C4C9']
explode = (0.15, 0, 0, 0)


# 自定义 autopct
def my_autopct(pct):
    if round(pct) == round(sizes2[3]):
        return '{:.1f}%'.format(pct)
    else:
        return '{:.0f}%'.format(pct)


# 画第一个饼状图
plt.figure(figsize = (10, 4))
plt.subplot(1, 2, 1)
wedges1, texts1, autotexts1 = plt.pie(sizes1, autopct = '%1.0f%%', startangle = 90, colors = colors, pctdistance = 0.6,
                                      radius = 1)

for autotext in autotexts1:
    autotext.set(size = 14, color = 'black', fontfamily = 'Helvetica')
plt.legend(['Driver', 'Analog', 'Crossbar'], loc = 'upper center', bbox_to_anchor = (-0.2, 0.7),
           frameon = False,
           prop = {'size': 14, 'family': 'Helvetica'})
plt.title('FRAME Energy Distribution', y = 0.96,fontfamily = 'Helvetica', fontsize = 16)

# 画第二个饼状图
plt.subplot(1, 2, 2)
wedges2, texts2, autotexts2 = plt.pie(sizes2, autopct = my_autopct, startangle = 90, colors = colors2,
                                      explode = explode,
                                      pctdistance = 0.6, radius = 1)

for autotext in autotexts2:
    autotext.set(size = 14, color = 'black', fontfamily = 'Helvetica')

plt.title('NeuroSim Energy Distribution', y = 0.96, fontfamily = 'Helvetica', fontsize = 16)

# 显示图例
plt.legend(['Driver', 'ADC', 'Digital', 'Crossbar'], loc = 'upper center', bbox_to_anchor = (-0.2, 0.75),
           frameon = False,
           prop = {'size': 14, 'family': 'Helvetica'})
# 调整子图之间的距离
plt.subplots_adjust(wspace = 0.4)
plt.subplots_adjust(right = 1.0)

# 显示图形
folder_path = "data"
os.makedirs(folder_path, exist_ok = True)
current_time = datetime.datetime.now().strftime("%Y_%m%d_%H%M")
filename = f"pie_DAC_Energy_{current_time}.svg"
file_path = os.path.join(folder_path, filename)
plt.savefig(file_path, format = 'svg')
plt.show()
