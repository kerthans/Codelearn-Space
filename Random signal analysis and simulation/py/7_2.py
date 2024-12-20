import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from scipy.signal import lfilter
import matplotlib.font_manager as fm
font_path = 'py/simhei.ttf'
font_prop = fm.FontProperties(fname=font_path)
plt.style.use('seaborn-v0_8')

# 设置随机种子以便复现
np.random.seed(0)

# 参数设置
N_values = [16, 32, 64, 128]  # 不同的滤波器长度
L = 1024  # 信号长度
f_signal = 0.05  # 信号频率
sigma_w = 0.5  # 噪声标准差

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

    # 计算误差
    error = X - X_hat

    # 绘制真实信号、观测信号及维纳滤波输出信号
    plt.figure(figsize=(14, 6))
    plt.plot(X, label='True signal X[n]')
    plt.plot(Y, label='Observed signal Y[n]', alpha=0.7)
    plt.plot(X_hat, label=f'Wiener filter output $\hat{{X}}[n]$ (N={N})', linestyle='--')
    plt.title(f'滤波器长度 N={N} 时的信号', fontproperties=font_prop)
    plt.xlabel('样本点 n', fontproperties=font_prop)
    plt.ylabel('幅值', fontproperties=font_prop)
    plt.legend()
    plt.grid(True)
    plt.show()

    # 绘制误差曲线和白噪声序列
    plt.figure(figsize=(14, 6))
    plt.plot(error, label='error e[n] = X[n] - $\hat{X}[n]$')
    plt.plot(W, label='white noise W[n]', alpha=0.7)
    plt.title(f'滤波器长度 N={N} 时的误差曲线和白噪声序列', fontproperties=font_prop)
    plt.xlabel('样本点 n', fontproperties=font_prop)
    plt.ylabel('幅值', fontproperties=font_prop)
    plt.legend()
    plt.grid(True)
    plt.show()