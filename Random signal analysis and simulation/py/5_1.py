import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
font_path = 'py/simhei.ttf'
font_prop = fm.FontProperties(fname=font_path)
plt.style.use('seaborn-v0_8')

# 设置参数
sigma = 1.0       # 功率谱幅度
f1 = 5            # 带通下限频率 (Hz)
f2 = 15           # 带通上限频率 (Hz)
fs = 100          # 采样频率 (Hz)
N = 1024          # 样本长度
t = np.arange(N) / fs  # 时间向量

# 频率分辨率
df = fs / N
freqs = np.fft.fftfreq(N, d=1/fs)

# 初始化功率谱
S_f = np.zeros(N)

# 设置带通功率谱
band = np.logical_and(np.abs(freqs) >= f1, np.abs(freqs) <= f2)
S_f[band] = sigma**2

# 生成随机相位
np.random.seed(0)  # 为了结果可重复
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

# 绘制样本函数
plt.figure(figsize=(12, 4))
plt.plot(t, x)
plt.title('随机带通信号的1条样本函数', fontproperties=font_prop)
plt.xlabel('时间 t (秒)', fontproperties=font_prop)
plt.ylabel('x(t)', fontproperties=font_prop)
plt.grid(True)
plt.show()