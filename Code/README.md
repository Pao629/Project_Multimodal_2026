simple_concat_baseline.ipynb（简单拼接）：
将 IQ 信号转换成 STFT、GASF、RP 等图像特征后，直接按通道拼接输入 CNN，完成 H0/H1 频谱感知二分类任务。

three_modal_attention_fusion.ipynb（多分支融合）：
分别用不同分支提取时域、时频图和关联图特征，再将多模态特征融合后进行 H0/H1 二分类，相比简单拼接版结构更清晰、表达能力更强。