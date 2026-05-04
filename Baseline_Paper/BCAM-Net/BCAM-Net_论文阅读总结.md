# BCAM-Net 论文阅读总结

> 论文：**BCAM-Net: A Bidirectional Cross-Attention Multimodal Network for IoT Spectrum Sensing under Generalized Gaussian Noise**  
> 任务：IoT / 认知无线电场景下的协作频谱感知二分类，即判断主用户 PU 是否存在。  
> 核心格式：**该论文使用多 SU 的时频图模态与协作 IQ 幅度—相位时间序列模态，采用双向交叉注意力 BCA 进行跨模态融合，在 GGD 脉冲噪声仿真数据集和真实 LoRa SDR 数据集上实验，取得了低 SNR 下更高的检测概率与较低模型复杂度。**

---

## 1. 论文基本信息

| 项目 | 内容 |
|---|---|
| 论文名称 | BCAM-Net: A Bidirectional Cross-Attention Multimodal Network for IoT Spectrum Sensing under Generalized Gaussian Noise |
| 研究任务 | 低信噪比、非高斯脉冲噪声下的 IoT 协作频谱感知 |
| 问题形式 | 二分类：$H_0$ 表示 PU 不存在，$H_1$ 表示 PU 存在 |
| 主要挑战 | 低 SNR、GGD 重尾脉冲噪声、多 SU 观测差异、单模态特征容易被噪声伪峰或短时异常扰动 |
| 核心方法 | 双分支多模态网络 + Bidirectional Cross-Attention，简称 BCAM-Net |

---

## 2. 该论文用了什么模态的数据

论文使用了两类互补模态。

### 2.1 时频图模态：Multi-SU CWT Time-Frequency Modality

对每个次用户 SU 接收到的复数 IQ 序列 $x_m(n)$ 独立做连续小波变换 CWT，得到复数 CWT 系数矩阵。由于 CWT 结果是复数，论文将其实部和虚部沿通道维度堆叠，形成每个 SU 的两通道时频图：

```text
每个 SU：2 × 128 × 256
2：CWT 实部 / CWT 虚部
128：尺度/频率维度
256：时间维度
```

多个 SU 堆叠后，形成时频分支输入：

```text
M × 2 × H × W
```

在实验中：

```text
M = 4, H = 128, W = 256
```

因此，这一模态可以理解为：

> 多个 SU 的 CWT 时频图模态，用于捕获局部时频纹理和全局能量分布。

### 2.2 时间序列模态：Cooperative IQ Magnitude-Phase Temporal Modality

时间分支不直接使用 CWT 图，而是先对多个 SU 的复数 IQ 序列做复数域平均：

```text
x_bar(n) = 1/M * sum_m x_m(n)
```

然后把每个复数样本转换为两个实数特征：

```text
RMS 归一化幅度 + 相位
```

所以时间分支输入是：

```text
N × 2
```

实验中：

```text
N = 256
```

这一模态可以理解为：

> 协作 IQ 幅度—相位时间序列模态，用于描述信号包络和相位随时间的演化规律，并抑制单个 SU 的偶发脉冲异常。

---

## 3. 该论文采用了什么融合方法

论文采用的是 **双向交叉注意力融合方法**，即 Bidirectional Cross-Attention，简称 BCA。

### 3.1 双分支特征提取

#### 时频分支

时频分支使用：

```text
CWT 时频图 → 3 个 Residual Block + CBAM → GAP → SU-wise time-frequency tokens
```

输出为：

```text
F_cnn ∈ R^{M × C}
```

其中每一行表示一个 SU 的全局时频 token。实验中：

```text
C = 32
```

#### 时间分支

时间分支使用：

```text
协作 IQ 幅度—相位序列 → 两层 Bi-GRU → temporal tokens
```

输出为：

```text
H ∈ R^{N × 64}
```

因为 Bi-GRU 每个方向 hidden size 为 32，双向拼接后为 64 维。

### 3.2 BCA 双向交叉注意力

BCA 包含两条交互路径。

#### Path 1：时频特征查询时间特征

```text
F_cnn 作为 Q
H 作为 K / V
```

含义是：

> 每个 SU 的时频 token 从时间序列中选择有用的时间证据，生成时间信息增强后的时频表示。

输出：

```text
F_tilde_cnn ∈ R^{M × 32}
```

#### Path 2：时间特征查询时频特征

```text
H 作为 Q
F_cnn 作为 K / V
```

含义是：

> 每个时间点从多个 SU 的时频 token 中选择可靠的时频证据，生成时频信息增强后的时间表示。

输出：

```text
H_tilde ∈ R^{N × 64}
```

### 3.3 最终融合

论文最终拼接三个向量：

```text
z = [f_cnn, f_bar_cnn, h_bar]
```

其中：

| 向量 | 含义 | 维度 |
|---|---|---|
| $f_{cnn}$ | BCA 前的原始全局时频特征 | 32 |
| $\bar{f}_{cnn}$ | BCA 后的增强时频特征 | 32 |
| $\bar{h}$ | BCA 后的时间摘要特征 | 64 |

所以最终融合向量为：

```text
32 + 32 + 64 = 128 维
```

然后通过：

```text
LayerNorm → Linear(128→256) → GELU → Dropout(0.2) → Linear(256→2) → Softmax
```

输出：

```text
p(H0), p(H1)
```

---

## 4. 该论文在哪些数据集上进行了实验

### 4.1 GGD 脉冲噪声仿真数据集

仿真设置如下：

| 项目 | 设置 |
|---|---|
| 系统 | 1 个 PU + 4 个 SU |
| PU 调制方式 | QPSK |
| 脉冲成形 | RRC pulse shaping |
| 信道 | 平坦瑞利衰落信道 |
| 噪声 | GGD 非高斯脉冲噪声 |
| 主要形状参数 | $\beta=0.5$ |
| 采样频率 | 200 kHz |
| 每帧长度 | 256 个复数基带采样点 |
| SNR 范围 | -20 dB 到 0 dB，步长 2 dB |
| 每个 SNR 样本数 | 1000 个 $H_0$，1000 个 $H_1$ |
| 数据集划分 | 训练集:验证集:测试集 = 7:2:1 |

### 4.2 GGD 形状参数鲁棒性数据集

为了验证模型对不同脉冲噪声强度的鲁棒性，论文还设置：

```text
β ∈ {0.3, 0.6, 1.0, 1.6, 2.0}
```

每个固定 $\beta$ 单独生成数据集，并分别训练、验证和测试 BCAM-Net。

### 4.3 真实 LoRa SDR 数据集

论文还使用公开真实 SDR 数据集验证实际可行性。

| 项目 | 设置 |
|---|---|
| 数据类型 | 真实 over-the-air LoRa IQ 记录 + 背景噪声 |
| 采集设备 | RTL-SDR |
| 中心频率 | 868.1 MHz |
| 带宽 | 125 kHz |
| LoRa 参数 | SF7 和 SF12 |
| 任务 | $H_0$：noise only；$H_1$：LoRa plus noise |
| 帧长 | 256 samples |
| 每个假设样本数 | 每个 SU 提取 1500 frames |
| 划分 | 训练集:验证集:测试集 = 7:2:1 |
| 协作设置 | 1 SU 和近似 2 SU 场景 |

---

## 5. 得到了怎么样的效果

### 5.1 消融实验效果

在固定虚警概率：

```text
Pf = 0.1
```

下，BCAM-Net 在低 SNR 区间优于单分支模型和简单拼接模型。

| 模型 | -14 dB | -16 dB |
|---|---:|---:|
| GRU only | 0.6373 | 0.5054 |
| RCAM-Net | 0.8333 | 0.6667 |
| DFFN | 0.8529 | 0.7312 |
| BCAM-Net | 0.9020 | 0.7957 |

说明：

> 双分支结构 + BCA 双向交叉注意力，比单独时间分支、单独时频分支和简单拼接融合更有效。

### 5.2 与其他模型对比

在 $P_f=0.1$ 下，论文与 CNN-Transformer、WT-ResNet、TFCFN、CNN、ED-SVM、DE-SVM 进行比较。

| 方法 | -14 dB | -16 dB |
|---|---:|---:|
| ED-SVM | 0.1275 | 0.0860 |
| DE-SVM | 0.0784 | 0.0645 |
| CNN-Transformer | 0.8529 | 0.7527 |
| WT-ResNet | 0.8431 | 0.7527 |
| TFCFN | 0.6765 | 0.5346 |
| CNN | 0.7451 | 0.6744 |
| BCAM-Net | 0.9020 | 0.7957 |

论文指出，在 SNR = -14 dB、$P_f=0.1$ 时，BCAM-Net 的 $P_d=0.9020$，相比 CNN-Transformer、WT-ResNet、TFCFN、CNN 分别提升 5.75%、6.98%、33.3%、21.1%。

### 5.3 不同 GGD 形状参数下的鲁棒性

当测试不同 $\beta$ 时，结果显示：

- 在极低 SNR 区间，$\beta$ 对性能影响明显；
- 当 SNR 提高到 -10 dB 及以上时，所有测试 $\beta$ 下的 $P_d$ 均超过 0.90；
- 说明 BCAM-Net 对不同脉冲噪声强度具有较强鲁棒性。

### 5.4 真实 LoRa 数据集效果

真实 LoRa 数据集上，在 $P_f=0.1$ 下：

| 方法 | 1 SU | 2 SUs |
|---|---:|---:|
| BCAM-Net | 0.527 | 0.807 |
| CNN-Transformer | 0.507 | 0.800 |
| WT-ResNet | 0.507 | 0.813 |
| TFCFN | 0.513 | 0.807 |
| CNN | 0.480 | 0.787 |

结论：

> 所有方法在 2 SU 协作下性能明显高于 1 SU，说明协作感知有效。BCAM-Net 在真实 LoRa 数据上具有竞争力，但 2 SU 下 WT-ResNet 的 0.813 略高于 BCAM-Net 的 0.807，因此论文更谨慎地将其表述为实际可行性验证和竞争性表现。

### 5.5 模型复杂度

| 模型 | 参数量 | 推理延迟 |
|---|---:|---:|
| BCAM-Net | 0.12M | 0.0313 ms/frame |
| CNN-Transformer | 0.92M | 0.7601 ms/frame |
| WT-ResNet | 0.14M | 0.0115 ms/frame |
| CNN | 0.11M | 0.4412 ms/frame |

BCAM-Net 参数量约 0.12M，FP32 权重存储约 0.48 MB，FP16 约 0.24 MB。说明它在性能和效率之间取得较好平衡，有边缘部署潜力。

---

## 6. 论文结论

这篇论文可以概括为：

> 该论文面向低 SNR 和 GGD 非高斯脉冲噪声下的 IoT 协作频谱感知任务，使用多 SU 的 CWT 时频图模态和协作 IQ 幅度—相位时间序列模态，提出 BCAM-Net 双分支网络，并通过双向交叉注意力实现时间特征与时频特征的相互校准。在仿真 GGD 噪声数据集和真实 LoRa SDR 数据集上，BCAM-Net 在低 SNR 下取得较高检测概率，并保持较低参数量和推理延迟，说明该方法兼具鲁棒性和轻量化优势。

---

