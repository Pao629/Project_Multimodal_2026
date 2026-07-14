# 消融实验代码说明

这些 notebook 均基于 `ours-diri.ipynb` 的同一数据划分、训练设置、阈值校准与评价指标生成，适合直接作为 IEEE 会议论文消融实验表格的代码来源。

## 完整模型

- 文件：`ours-diri.ipynb`
- 组成：六通道时序分支 + 八通道图像分支 + 门控双向跨模态细化 + Dirichlet证据决策头。
- 本次已修正其中与代码不一致的注释和打印内容，例如 Softmax/Dirichlet、4通道/6通道、图像尺寸说明、乱码注释等。

## A1: 去掉门控双向跨模态细化

- 文件：`ablation_A1_no_gated_cross_modal.ipynb`
- 少了什么：去掉 image→time 和 time→image 的跨模态上下文投影，以及两个 learnable scalar gates 控制的残差细化。
- 保留什么：时序分支、图像分支、原始特征拼接、Dirichlet证据头。
- 对比意义：证明论文提出的 gated bidirectional cross-modal refinement 是否比直接融合更有效、更稳定。

## A2: 仅时序分支

- 文件：`ablation_A2_temporal_only.ipynb`
- 少了什么：八通道图像分支、跨模态细化、多模态融合。
- 保留什么：六通道时序特征、ChannelAttention1D、多尺度1D卷积、TCN、Dirichlet证据头。
- 对比意义：验证图像模态和多模态互补信息对性能提升的贡献。

## A3: 仅图像分支

- 文件：`ablation_A3_image_only.ipynb`
- 少了什么：六通道时序TCN分支、跨模态细化、多模态融合。
- 保留什么：八通道信号图像、ChannelAttention2D、ResNet18多阶段聚合、Dirichlet证据头。
- 对比意义：验证时序波形、相位差和瞬时频率动态是否提供了图像表示之外的有效信息。

## A4: 去掉 Dirichlet 证据学习

- 文件：`ablation_A4_softmax_no_dirichlet.ipynb`
- 少了什么：Dirichlet evidence、alpha浓度参数、u=K/S不确定性、非目标证据KL正则。
- 保留什么：双分支、门控跨模态细化、验证集阈值校准、focal与pairwise ranking思想。
- 对比意义：验证证据建模是否降低分布偏移下的过度自信，并提升可靠检测。

