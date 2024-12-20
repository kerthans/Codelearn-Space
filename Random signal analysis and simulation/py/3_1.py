import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
font_path = 'py/simhei.ttf'
font_prop = fm.FontProperties(fname=font_path)
plt.style.use('seaborn-v0_8')

# 设置随机种子以便复现
np.random.seed(42)

# 参数设置
a = 2.0           # 常数 a
b = 1.5           # 常数 b
c = 1.0           # 常数 c
sigma_X = 1.0     # 高斯白噪声 X(t) 的方差
omega = 2 * np.pi * 5  # 角频率，5 Hz

# 采样参数
f_max = 5         # 信号的最大频率，确保奈奎斯特准则
f_s = 2 * f_max * 1.2  # 采样频率，略高于奈奎斯特频率
T = 1 / f_s        # 采样间隔
N = 1000           # 采样点数

# 生成时间序列
t = np.arange(N) * T

# 生成随机信号
X = np.random.normal(0, np.sqrt(sigma_X), N)  # 高斯白噪声
W = np.random.uniform(-1, 1, N)              # 均匀分布随机变量

# 生成连续时间随机信号 Y(t)
Y = a + b * X + c * W

# 绘制4条样本函数
plt.figure(figsize=(14, 8))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.plot(t, Y + i*5)  # 为了区分不同样本，将其垂直偏移
    plt.title(f'样本函数 Y[n] 第 {i+1} 条样本', fontproperties=font_prop)
    plt.xlabel('时间 t (秒)', fontproperties=font_prop)
    plt.ylabel('幅值', fontproperties=font_prop)
    plt.grid(True)
plt.tight_layout()
plt.show()