import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram
import matplotlib.font_manager as fm
font_path = 'py/simhei.ttf'
font_prop = fm.FontProperties(fname=font_path)
plt.style.use('seaborn-v0_8')

# 参数设置
f0 = 0.1  # 干扰信号频率，假设为0.1
c = 1.0   # 干扰信号幅度
N_values = [256, 512, 1024, 2048]  # 不同的样本长度
fs = 1.0  # 采样频率，归一化

for N in N_values:
    n = np.arange(N)
    # 生成白噪声X[n]
    X = np.random.normal(0, 1, N)
    # 生成干扰信号
    interference = c * np.sin(2 * np.pi * f0 * n / fs)
    # 干扰后的信号Y[n]
    Y = X + interference
    # 计算功率谱估计
    f, Pxx = periodogram(Y, fs=fs, scaling='density')
    # 绘制功率谱
    plt.figure(figsize=(8, 4))
    plt.semilogy(f, Pxx)
    plt.title(f'功率谱估计 (N={N})', fontproperties=font_prop)
    plt.xlabel('频率 (f)', fontproperties=font_prop)
    plt.ylabel('功率谱密度', fontproperties=font_prop)
    plt.grid(True)
    plt.show()