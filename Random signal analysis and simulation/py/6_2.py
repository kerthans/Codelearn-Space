import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram
import matplotlib.font_manager as fm
font_path = 'py/simhei.ttf'
font_prop = fm.FontProperties(fname=font_path)
plt.style.use('seaborn-v0_8')

# 参数设置
f0 = 0.1  # 干扰信号频率
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
    # 差分滤波Z[n] = Y[n] - Y[n-1]
    Z = np.diff(Y)
    # 计算功率谱估计
    f, Pxx = periodogram(Z, fs=fs, scaling='density')
    # 理论功率谱
    S_Z_theory = (2 - 2 * np.cos(2 * np.pi * f)) * (1 + (c**2 / 2) * (1 - np.cos(2 * np.pi * f0)) * (f == f0))
    # 绘制功率谱
    plt.figure(figsize=(8, 4))
    plt.semilogy(f, Pxx, label='Simulated power spectrum')
    # plt.semilogy(f, Pxx, label='仿真功率谱')
    # 绘制理论功率谱
    plt.semilogy(f, 2 - 2 * np.cos(2 * np.pi * f) + (c**2 / 2) * (2 - 2 * np.cos(2 * np.pi * f0)) * (f == f0), 
                label='Theoretical power spectrum', linestyle='--')
    plt.title(f'Power spectrum estimation after differential filtering (N={N})')
    plt.title(f'差分滤波后功率谱估计 (N={N})', fontproperties=font_prop)
    plt.xlabel('频率 (f)', fontproperties=font_prop)
    plt.ylabel('功率谱密度', fontproperties=font_prop)
    plt.legend()
    plt.grid(True)
    plt.show()