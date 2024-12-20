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

def estimate_power_spectrum(x):
    """
    估计功率谱（周期图）
    """
    N = len(x)
    X_fft = np.fft.fft(x)
    S = (np.abs(X_fft)**2) / N
    # 仅取单边谱
    S = S[:N//2] * 2
    freqs = np.fft.fftfreq(N, d=1/fs)[:N//2]
    return freqs, S

# 设置参数
sigma = 1.0
f1 = 5          # 带通下限频率 (Hz)
f2 = 15         # 带通上限频率 (Hz)
fs = 100        # 采样频率 (Hz)
num_experiments = 50
N_values = [50, 100, 200]

plt.figure(figsize=(12, 8))

for N in N_values:
    S_estimates = np.zeros(N//2)
    freqs = np.fft.fftfreq(N, d=1/fs)[:N//2]
    for _ in range(num_experiments):
        x = generate_bandpass_signal(sigma, f1, f2, fs, N)
        _, S = estimate_power_spectrum(x)
        S_estimates += S
    S_estimates /= num_experiments
    plt.plot(freqs, S_estimates, label=f'estimation S(f), N={N}')
    
# 计算理论功率谱
S_theoretical = np.zeros_like(freqs)
S_theoretical[np.logical_and(freqs >= f1, freqs <= f2)] = sigma**2

plt.plot(freqs, S_theoretical, 'k--', label='theory S(f)')
plt.title('功率谱的估计与理论值对比', fontproperties=font_prop)
plt.xlabel('频率 f (Hz)', fontproperties=font_prop)
plt.ylabel('S(f)', fontproperties=font_prop)
plt.legend()
plt.grid(True)
plt.show()