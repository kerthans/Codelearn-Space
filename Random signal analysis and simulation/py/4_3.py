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

def estimate_power_spectrum(X):
    N = len(X)
    X_fft = np.fft.fft(X)
    S = (np.abs(X_fft)**2) / N
    S = S[:N//2] * 2  # 单边谱
    freqs = np.fft.fftfreq(N, d=1.0)[:N//2]
    return freqs, S

# 设置参数
sigma = 1.0
alpha = 0.5
a = np.exp(-alpha)
sigma_w = np.sqrt(sigma**2 * (1 - a**2))
num_experiments = 50
N_values = [50, 100, 200]

plt.figure(figsize=(12, 8))

for N in N_values:
    S_estimates = np.zeros(N//2)
    for _ in range(num_experiments):
        X = generate_ar1(a, sigma_w, N)
        freqs, S = estimate_power_spectrum(X)
        S_estimates += S
    S_estimates /= num_experiments
    plt.plot(freqs, S_estimates, label=f'Estimated S(f), N={N}')

# 理论功率谱
f = freqs
S_theoretical = (2 * sigma**2 * alpha) / (alpha**2 + (2 * np.pi * f)**2)
plt.plot(f, S_theoretical, 'k--', label='Theory S(f)')
plt.title('功率谱的估计与理论值对比', fontproperties=font_prop)
plt.xlabel('频率 f (Hz)', fontproperties=font_prop)
plt.ylabel('S(f)', fontproperties=font_prop)
plt.legend()
plt.grid(True)
plt.show()