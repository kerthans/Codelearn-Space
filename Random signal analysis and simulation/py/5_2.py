import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
font_path = 'py/simhei.ttf'
font_prop = fm.FontProperties(fname=font_path)
plt.style.use('seaborn-v0_8')

def generate_bandpass_signal(sigma, f1, f2, fs, N):
    """
    生成带通随机信号的时域样本
    """
    df = fs / N
    freqs = np.fft.fftfreq(N, d=1/fs)
    
    # 初始化功率谱
    S_f = np.zeros(N)
    
    # 设置带通功率谱
    band = np.logical_and(np.abs(freqs) >= f1, np.abs(freqs) <= f2)
    S_f[band] = sigma**2
    
    # 生成随机相位
    random_phase = np.random.uniform(0, 2*np.pi, N//2 - 1)
    # 构建频域信号（只填充正频率部分）
    X_f = np.zeros(N, dtype=complex)
    X_f[0] = np.random.normal(0, np.sqrt(S_f[0]))  # DC分量
    X_f[N//2] = np.random.normal(0, np.sqrt(S_f[N//2]))  # Nyquist频率
    
    # 填充正频率部分
    X_f[1:N//2] = np.sqrt(S_f[1:N//2] / 2) * (np.cos(random_phase) + 1j * np.sin(random_phase))
    # 负频率部分由正频率部分的共轭对称性决定
    X_f[N//2+1:] = np.conj(X_f[1:N//2][::-1])
    
    # 逆傅里叶变换得到时域信号
    x = np.fft.ifft(X_f).real
    return x

def estimate_autocorrelation(x, max_lag):
    """
    估计自相关函数
    """
    N = len(x)
    R = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        R[k] = np.sum(x[:N - k] * x[k:]) / N
    return R

# 设置参数
sigma = 1.0
f1 = 5          # 带通下限频率 (Hz)
f2 = 15         # 带通上限频率 (Hz)
fs = 100        # 采样频率 (Hz)
max_lag = 20    # 最大时延
num_experiments = 50
N_values = [50, 100, 200]

plt.figure(figsize=(12, 8))

for N in N_values:
    R_estimates = np.zeros(max_lag + 1)
    for _ in range(num_experiments):
        x = generate_bandpass_signal(sigma, f1, f2, fs, N)
        R = estimate_autocorrelation(x, max_lag)
        R_estimates += R
    R_estimates /= num_experiments
    lags = np.arange(max_lag + 1) / fs  # 转换为时间单位
    plt.plot(lags, R_estimates, label=f'estimation R(k), N={N}')
    
# 计算理论自相关函数
def theoretical_autocorrelation(tau, sigma, f1, f2):
    return (sigma**2 / (np.pi * tau)) * (np.sin(2 * np.pi * f2 * tau) - np.sin(2 * np.pi * f1 * tau))

# 计算理论自相关函数
taus = np.arange(max_lag + 1) / fs
R_theoretical = np.zeros_like(taus)
# 处理 tau=0 的情况
R_theoretical[0] = (sigma**2 / (np.pi * 1e-10)) * (np.sin(2 * np.pi * f2 * 1e-10) - np.sin(2 * np.pi * f1 * 1e-10))
for i in range(1, len(taus)):
    R_theoretical[i] = (sigma**2 / (np.pi * taus[i])) * (np.sin(2 * np.pi * f2 * taus[i]) - np.sin(2 * np.pi * f1 * taus[i]))

plt.plot(taus, R_theoretical, 'k--', label='theory R(k)')
plt.title('自相关函数的估计与理论值对比', fontproperties=font_prop)
plt.xlabel('时延 τ (秒)', fontproperties=font_prop)
plt.ylabel('R(τ)', fontproperties=font_prop)
plt.legend()
plt.grid(True)
plt.show()