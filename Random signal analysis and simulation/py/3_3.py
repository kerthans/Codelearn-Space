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

# 理论方差
sigma_W = np.sqrt(1/3)  # 标准差
sigma_Y_squared = (b**2) * sigma_X**2 + (c**2) * (1/3)

# 设定 N 的范围
N_values = np.arange(10, 1000, 10)
num_experiments = 200

# 初始化数组存储方差估计
variance_estimated = []

# 初始化数组存储理论方差
variance_theory = sigma_Y_squared / N_values

# 进行蒙特卡洛实验
for N in N_values:
    mu_hats = []
    for _ in range(num_experiments):
        X = np.random.normal(0, np.sqrt(sigma_X), N)
        W = np.random.uniform(-1, 1, N)
        Y = a + b * X + c * W
        mu_hat = np.mean(Y)
        mu_hats.append(mu_hat)
    variance = np.var(mu_hats, ddof=1)
    variance_estimated.append(variance)

# 绘制方差估计与理论方差
plt.figure(figsize=(10, 6))
plt.plot(N_values, variance_estimated, label='Estimate variance $\hat{\\text{Var}}(\\hat{\\mu}_N)$', color='blue')
plt.plot(N_values, variance_theory, label='Theoretical variance $\\frac{\\sigma_Y^2}{N}$', color='red', linestyle='--')
plt.title('均值估计的方差随采样点数 N 的变化关系', fontproperties=font_prop)
plt.xlabel('采样点数 N', fontproperties=font_prop)
plt.ylabel('方差', fontproperties=font_prop)
plt.legend()
plt.grid(True)
plt.show()