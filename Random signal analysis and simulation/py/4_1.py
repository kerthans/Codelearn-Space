import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
font_path = 'py/simhei.ttf'
font_prop = fm.FontProperties(fname=font_path)
plt.style.use('seaborn-v0_8')

# 设置参数
sigma = 1.0      # 方差
alpha = 0.5      # 衰减系数
a = np.exp(-alpha)
sigma_w = np.sqrt(sigma**2 * (1 - a**2))
N = 100         # 样本长度

# 生成5条样本函数
num_samples = 5
samples = np.zeros((num_samples, N))

for i in range(num_samples):
    W = np.random.normal(0, sigma_w, N)
    for n in range(1, N):
        samples[i, n] = a * samples[i, n-1] + W[n]

# 绘制样本函数
plt.figure(figsize=(12, 8))
for i in range(num_samples):
    plt.plot(samples[i], label=f'Sample {i+1}')
plt.title('随机信号的5条样本函数', fontproperties=font_prop)
plt.xlabel('样本点 n', fontproperties=font_prop)
plt.ylabel('X[n]', fontproperties=font_prop)
plt.legend()
plt.grid(True)
plt.show()