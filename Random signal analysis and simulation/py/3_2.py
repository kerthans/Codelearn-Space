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

# 设定 N 的范围
N_values = np.arange(10, 1000, 10)

# 理论均值
mu_theory = a

# 初始化数组存储估计均值
mu_estimated = []

# 生成随机信号一次性生成最大 N
max_N = N_values[-1]
X = np.random.normal(0, np.sqrt(sigma_X), max_N)
W = np.random.uniform(-1, 1, max_N)
Y = a + b * X + c * W

# 计算累计均值
cumulative_sum = np.cumsum(Y)

for N in N_values:
    mu_hat = cumulative_sum[N-1] / N
    mu_estimated.append(mu_hat)

# 绘制均值估计与理论均值
plt.figure(figsize=(10, 6))
plt.plot(N_values, mu_estimated, label='Estimated mean $\hat{\mu}_N$', color='blue')
plt.axhline(y=mu_theory, color='red', linestyle='--', label='Theoretical mean $\mu$')
plt.title('估计均值随采样点数 N 的变化关系', fontproperties=font_prop)
plt.xlabel('采样点数 N', fontproperties=font_prop)
plt.ylabel('均值', fontproperties=font_prop)
plt.legend()
plt.grid(True)
plt.show()
