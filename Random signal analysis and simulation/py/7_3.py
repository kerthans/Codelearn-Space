import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from scipy.signal import lfilter, periodogram
import matplotlib.font_manager as fm
font_path = 'py/simhei.ttf'
font_prop = fm.FontProperties(fname=font_path)
plt.style.use('seaborn-v0_8')

# 设置随机种子以便复现
np.random.seed(0)

# 参数设置
N_values = [16, 64]  # 不同的滤波器长度
L = 2048  # 信号长度
f_signal = 0.05  # 信号频率
sigma_w = 0.5  # 噪声标准差
fs = 1.0  # 归一化采样频率

# 生成真实信号 X[n]：正弦随机相位信号
phi = np.random.uniform(0, 2*np.pi, L)
X = np.sin(2 * np.pi * f_signal * np.arange(L) + phi)

# 生成高斯白噪声 W[n]
W = np.random.normal(0, sigma_w, L)

# 生成观测信号 Y[n] = X[n] + W[n]
Y = X + W

for N in N_values:
    # 计算自相关函数 R_YY[m] 和互相关函数 R_XY[m]
    R_YY = np.correlate(Y, Y, mode='full') / L
    mid = len(R_YY) // 2
    R_YY = R_YY[mid:mid + N]

    R_XY = np.correlate(X, Y, mode='full') / L
    R_XY = R_XY[mid:mid + N]

    # 构建自相关矩阵 Toeplitz 矩阵
    R_matrix = toeplitz(R_YY)

    # 求解维纳-霍夫方程
    h = np.linalg.inv(R_matrix).dot(R_XY)

    # 进行滤波
    X_hat = lfilter(h, [1.0], Y)

    # 计算功率谱
    f_Y, P_Y = periodogram(Y, fs=fs, scaling='density')
    f_Xhat, P_Xhat = periodogram(X_hat, fs=fs, scaling='density')

    # 绘制功率谱
    plt.figure(figsize=(14, 6))
    plt.semilogy(f_Y, P_Y, label='Pre-filter power spectrum S_Y(f)')
    plt.semilogy(f_Xhat, P_Xhat, label=f'Filtered power spectrum S_{{\\hat{{X}}}}(f) (N={N})')
    plt.title(f'滤波前后功率谱比较 (滤波器长度 N={N})', fontproperties=font_prop)
    plt.xlabel('频率 f', fontproperties=font_prop)
    plt.ylabel('功率谱密度', fontproperties=font_prop)
    plt.legend()
    plt.grid(True)
    plt.show()