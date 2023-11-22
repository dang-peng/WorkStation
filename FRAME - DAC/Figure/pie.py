import datetime
import os

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Helvetica'
# 数据
labels = ['DAC', 'ADC', 'Digital', 'Crossbar']
sizes1 = [15, 57, 19, 9]
sizes2 = [0.1, 81, 17, 2]
colors = ['#D4E155', '#95BEE3', '#F8C4C9', '#96C95C']
# colors = ['#FFCD33', '#7CAFDD', '#7EB559', '#F8C4C9']
explode = (0.2, 0, 0, 0)


# 自定义 autopct
def my_autopct(pct):
    if round(pct) == round(sizes2[0]):
        return '{:.1f}%'.format(pct)
    else:
        return '{:.0f}%'.format(pct)


# 画第一个饼状图
plt.figure(figsize = (10, 4))
plt.subplot(1, 2, 1)
wedges1, texts1, autotexts1 = plt.pie(sizes1, autopct = '%1.0f%%', startangle = 90, colors = colors, pctdistance = 0.6,
                                      radius = 1.1)

for autotext in autotexts1:
    autotext.set(size = 14, color = 'black', fontfamily = 'Helvetica')

plt.title('Energy Distribution', fontfamily = 'Helvetica', fontsize = 16)

# 画第二个饼状图
plt.subplot(1, 2, 2)
wedges2, texts2, autotexts2 = plt.pie(sizes2, autopct = my_autopct, startangle = 90, colors = colors, explode = explode,
                                      pctdistance = 0.6, radius = 1.1)

for autotext in autotexts2:
    autotext.set(size = 14, color = 'black', fontfamily = 'Helvetica')

plt.title('Area Distribution', fontfamily = 'Helvetica', fontsize = 16)

# 显示图例
plt.legend(['DAC', 'ADC', 'Digital', 'Crossbar'], loc = 'upper center', bbox_to_anchor = (-0.3, 0.75), frameon = False,
           prop = {'size': 14, 'family': 'Helvetica'})
# 调整子图之间的距离
plt.subplots_adjust(wspace = 0.72)

# 显示图形
folder_path = "data"
os.makedirs(folder_path, exist_ok = True)
current_time = datetime.datetime.now().strftime("%Y_%m%d_%H%M")
filename = f"pie_{current_time}.svg"
file_path = os.path.join(folder_path, filename)
plt.savefig(file_path, format = 'svg')
plt.show()
