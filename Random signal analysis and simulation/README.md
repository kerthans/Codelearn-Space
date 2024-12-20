本项目包含《随机信号分析》期末考查的所有代码实现和文档转换工具。

## 目录结构

```
.
├── README.md                     # 本说明文档
├── requirements.txt              # 项目依赖文件
├── py/                          # Python代码目录
│   ├── __init__.py              # Python包初始化文件
│   ├── 3_1.py             # 第3题代码实现
│   ├── 3_2.py
│   ├── 3_3.py
│   ├── 4_1.py             # 第4题代码实现
│   ├── 4_1.py
│   ├── 4_2.py
│   ├── 4_3.py
│   ├── 5_1.py             # 第5题代码实现
│   ├── 5_2.py
│   ├── 5_3.py
│   ├── 6_1.py              # 第6题代码实现
│   ├── 6_2.py
│   ├── 7_1.py              # 第7题代码实现
│   ├── 7_2.py             
│   └── 7_3.py             
├── image/                       # 图像输出目录
│   ├── 3_1.png                 # 第3题图1
│   ├── 3_2.png                 # 第3题图2
│   ├── 3_3.png                 # 第3题图3
│   └── ...                     # 其他图像文件
└── Random signal analysis simulation general document.md      # Markdown格式的考查文档
```

## 环境配置

1. 首先确保已安装Python 3.8+，然后创建并激活虚拟环境：

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境（根据操作系统选择）
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

2. 安装项目依赖：

```bash
pip install -r requirements.txt
```

## 主要依赖包

在requirements.txt中包含以下主要依赖：

```
numpy==1.24.3
matplotlib==3.7.1
scipy==1.10.1
weasyprint==60.1
markdown==3.4.3
python-markdown-math==0.8
beautifulsoup4==4.12.2
```

## 代码说明


### 问题3实现（3_1.py、3_2.py、3_3.py）

包含随机信号生成、均值估计和蒙特卡洛实验的实现。

主要函数：
- `generate_signal(N, A, sigma, omega0)`: 生成随机信号
- `estimate_mean(signal, N)`: 估计信号均值
- `monte_carlo_variance(N, num_experiments)`: 蒙特卡洛方差估计

输出图像：
- `3_1.png`: 5条样本函数图
- `3_2.png`: 估计均值随采样点数N的变化关系
- `3_3.png`: 均值估计的方差随采样点数N的变化关系

### 问题4实现（4_1.py、4_2.py、4_3.py）

实现了随机信号的自相关函数和功率谱估计。

主要函数：
- `generate_signal(N, b, sigma)`: 生成随机信号
- `estimate_autocorr(signal, max_lag)`: 估计自相关函数
- `estimate_power_spectrum(signal)`: 估计功率谱

输出图像：
- `4_1.png`: 5条样本函数图
- `4_2.png`: 自相关函数的估计与理论值对比
- `4_3.png`: 功率谱的估计与理论值对比

### 问题5实现（5_1.py、5_2.py、5_3.py）

包含带通随机信号的生成和分析。

主要函数：
- `generate_bandpass_signal(sigma, b, c, omega0, N)`: 生成带通随机信号
- `estimate_autocorr(signal, max_lag)`: 估计自相关函数
- `estimate_power_spectrum(signal)`: 估计功率谱

输出图像：
- `5_1.png`: 带通随机信号样本函数
- `5_2.png`: 自相关函数估计结果
- `5_3.png`: 功率谱估计结果

### 问题6实现（6_1.py、6_2.py）

实现了差分滤波器及其性能分析。

主要函数：
- `generate_noisy_signal(N)`: 生成含噪声信号
- `differential_filter(signal, T0)`: 差分滤波实现
- `estimate_power_spectrum(signal)`: 功率谱估计

输出图像：
- `6_1_1.png` ~ `6_1_4.png`: 不同N下的滤波前功率谱
- `6_2_1.png` ~ `6_2_4.png`: 不同N下的滤波后功率谱

### 问题7实现（7_1.py、7_2.py、7_3.py）

包含维纳滤波器的设计和性能分析。

主要函数：
- `generate_signal_noise(N)`: 生成信号和噪声
- `design_wiener_filter(signal, noise, N)`: 设计维纳滤波器
- `apply_wiener_filter(signal, h)`: 应用维纳滤波器

输出图像：
- `7_1_1.png` ~ `7_1_2.png`: 维纳滤波结果和误差分析
- `7_2_1.png` ~ `7_2_8.png`: 不同滤波器长度的滤波效果
- `7_3_1.png` ~ `7_3_2.png`: 滤波前后功率谱对比

## 运行说明

1. 每个Python文件都可以独立运行，例如：

```bash
# 运行第3题代码
python py/3_1.py

# 运行第4题代码
python py/4_1.py

# 以此类推...
```


## 注意事项

1. 所有代码都设置了固定的随机种子（np.random.seed(42)），以确保结果可重现


2. 部分图像生成可能需要较长时间，特别是进行蒙特卡洛实验时

3. 所有数字参数（如signal.length）都可以在代码中调整
