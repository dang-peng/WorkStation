import matplotlib.pyplot as plt

# 数据
sizes = [25, 35, 20, 20]

# 画饼状图
plt.pie(sizes, autopct='', startangle=90)
plt.axis('equal')  # 保持饼状图的纵横比相等

# 显示图形
plt.show()
