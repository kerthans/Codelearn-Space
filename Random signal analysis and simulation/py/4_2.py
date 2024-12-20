import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
font_path = 'py/simhei.ttf'
font_prop = fm.FontProperties(fname=font_path)
plt.style.use('seaborn-v0_8')

def generate_ar1(a, sigma_w, N):
    X = np.zeros(N)
    W = np.random.normal(0, sigma_w, N)
    for n in range(1, N):
        X[n] = a * X[n-1] + W[n]
    return X

def estimate_autocorrelation(X, max_lag):
    N = len(X)
    R = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        R[k] = np.sum(X[:N - k] * X[k:]) / N
    return R

# 设置参数
sigma = 1.0
alpha = 0.5
a = np.exp(-alpha)
sigma_w = np.sqrt(sigma**2 * (1 - a**2))
num_experiments = 50
max_lag = 20  # 最大时延

# 设置不同的N
N_values = [50, 100, 200]

plt.figure(figsize=(12, 8))

for N in N_values:
    R_estimates = np.zeros(max_lag + 1)
    for _ in range(num_experiments):
        X = generate_ar1(a, sigma_w, N)
        R = estimate_autocorrelation(X, max_lag)
        R_estimates += R
    R_estimates /= num_experiments
    lags = np.arange(max_lag + 1)
    theoretical_R = sigma**2 * np.exp(-alpha * lags)
    plt.plot(lags, R_estimates, label=f'Estimated R(k), N={N}')
    
# 绘制理论自相关函数
lags = np.arange(max_lag + 1)
theoretical_R = sigma**2 * np.exp(-alpha * lags)
plt.plot(lags, theoretical_R, 'k--', label='Theory R(k)')
plt.title('自相关函数的估计与理论值对比', fontproperties=font_prop)
plt.xlabel('时延 k', fontproperties=font_prop)
plt.ylabel('R(k)', fontproperties=font_prop)
plt.legend()
plt.grid(True)
plt.show()