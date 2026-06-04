# 模型性能对比结果汇总

> 说明：以下表格均根据原始表格内容整理，未对空白项进行推测或补全。空白项用 `—` 表示，代表原表中未填写。

---

## 表 1：完整指标总表

该表保留原始表格中的全部字段，用于总体查看不同模型在 Validation、Test Set 1 和 Test Set 2 上的分类性能与不确定性表现。

| Method | Dataset | ACC | F1-score | Precision | Recall(Pd) | Pfa | AUC | Mean Uncertainty | U (Correct Pred) | U (Incorrect Pred) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BiLSTM | Validation | 0.9124 | 0.9071 | 0.9662 | 0.8548 | 0.0299 | 0.9552 | 0.2529 | 0.2267 | 0.5259 |
| BiLSTM | Test Set 1 | 0.8474 | 0.8258 | 0.9621 | 0.7234 | 0.0285 | 0.8624 | 0.2440 | 0.2178 | 0.3893 |
| BiLSTM | Test Set 2 | 0.8769 | 0.8638 | 0.9669 | 0.7806 | 0.0267 | 0.9039 | 0.2479 | 0.2224 | 0.4298 |
| CNN | Validation | 0.9210 | 0.9170 | 0.9663 | 0.8725 | 0.0304 | 0.9643 | — | — | — |
| CNN | Test Set 1 | 0.8474 | 0.8263 | 0.9588 | 0.7260 | 0.0312 | 0.8669 | — | — | — |
| CNN | Test Set 2 | 0.8781 | 0.8658 | 0.9628 | 0.7866 | 0.0304 | 0.9082 | — | — | — |
| BCAM-Net | Validation | 0.9124 | 0.9071 | 0.9662 | 0.8548 | 0.0299 | 0.9552 | 0.2529 | 0.2267 | 0.5259 |
| BCAM-Net | Test Set 1 | 0.8474 | 0.8258 | 0.9621 | 0.7234 | 0.0285 | 0.8624 | 0.2440 | 0.2178 | 0.3893 |
| BCAM-Net | Test Set 2 | 0.8769 | 0.8638 | 0.9669 | 0.7806 | 0.0267 | 0.9039 | 0.2479 | 0.2224 | 0.4298 |
| BiLSTM + ResNet18 + BCA | Validation | — | — | — | — | — | — | — | — | — |
| BiLSTM + ResNet18 + BCA | Test Set 1 | — | — | — | — | — | — | — | — | — |
| BiLSTM + ResNet18 + BCA | Test Set 2 | — | — | — | — | — | — | — | — | — |

---

## 表 2：分类性能指标对比

该表只保留分类性能相关指标，包括 ACC、F1-score、Precision、Recall(Pd)、Pfa 和 AUC，适合用于论文或报告中展示模型检测性能。

| Method | Dataset | ACC | F1-score | Precision | Recall(Pd) | Pfa | AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| BiLSTM | Validation | 0.9124 | 0.9071 | 0.9662 | 0.8548 | 0.0299 | 0.9552 |
| BiLSTM | Test Set 1 | 0.8474 | 0.8258 | 0.9621 | 0.7234 | 0.0285 | 0.8624 |
| BiLSTM | Test Set 2 | 0.8769 | 0.8638 | 0.9669 | 0.7806 | 0.0267 | 0.9039 |
| CNN | Validation | 0.9210 | 0.9170 | 0.9663 | 0.8725 | 0.0304 | 0.9643 |
| CNN | Test Set 1 | 0.8474 | 0.8263 | 0.9588 | 0.7260 | 0.0312 | 0.8669 |
| CNN | Test Set 2 | 0.8781 | 0.8658 | 0.9628 | 0.7866 | 0.0304 | 0.9082 |
| BCAM-Net | Validation | 0.9124 | 0.9071 | 0.9662 | 0.8548 | 0.0299 | 0.9552 |
| BCAM-Net | Test Set 1 | 0.8474 | 0.8258 | 0.9621 | 0.7234 | 0.0285 | 0.8624 |
| BCAM-Net | Test Set 2 | 0.8769 | 0.8638 | 0.9669 | 0.7806 | 0.0267 | 0.9039 |
| BiLSTM + ResNet18 + BCA | Validation | — | — | — | — | — | — |
| BiLSTM + ResNet18 + BCA | Test Set 1 | — | — | — | — | — | — |
| BiLSTM + ResNet18 + BCA | Test Set 2 | — | — | — | — | — | — |

---

## 表 3：不确定性指标对比

该表只保留 Mean Uncertainty、U (Correct Pred) 和 U (Incorrect Pred)，用于分析模型预测正确和预测错误时的不确定性差异。

| Method | Dataset | Mean Uncertainty | U (Correct Pred) | U (Incorrect Pred) |
|---|---|---:|---:|---:|
| BiLSTM | Validation | 0.2529 | 0.2267 | 0.5259 |
| BiLSTM | Test Set 1 | 0.2440 | 0.2178 | 0.3893 |
| BiLSTM | Test Set 2 | 0.2479 | 0.2224 | 0.4298 |
| CNN | Validation | — | — | — |
| CNN | Test Set 1 | — | — | — |
| CNN | Test Set 2 | — | — | — |
| BCAM-Net | Validation | 0.2529 | 0.2267 | 0.5259 |
| BCAM-Net | Test Set 1 | 0.2440 | 0.2178 | 0.3893 |
| BCAM-Net | Test Set 2 | 0.2479 | 0.2224 | 0.4298 |
| BiLSTM + ResNet18 + BCA | Validation | — | — | — |
| BiLSTM + ResNet18 + BCA | Test Set 1 | — | — | — |
| BiLSTM + ResNet18 + BCA | Test Set 2 | — | — | — |

---

## 表 4：Validation Set 对比

该表单独汇总各模型在验证集上的表现，适合观察模型在已知调制和训练信噪比范围内的检测能力。

| Method | ACC | F1-score | Precision | Recall(Pd) | Pfa | AUC | Mean Uncertainty | U (Correct Pred) | U (Incorrect Pred) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BiLSTM | 0.9124 | 0.9071 | 0.9662 | 0.8548 | 0.0299 | 0.9552 | 0.2529 | 0.2267 | 0.5259 |
| CNN | 0.9210 | 0.9170 | 0.9663 | 0.8725 | 0.0304 | 0.9643 | — | — | — |
| BCAM-Net | 0.9124 | 0.9071 | 0.9662 | 0.8548 | 0.0299 | 0.9552 | 0.2529 | 0.2267 | 0.5259 |
| BiLSTM + ResNet18 + BCA | — | — | — | — | — | — | — | — | — |

---

## 表 5：Test Set 1 对比

该表单独汇总各模型在 Test Set 1 上的表现，适合比较模型在 SNR 泛化任务中的鲁棒性。

| Method | ACC | F1-score | Precision | Recall(Pd) | Pfa | AUC | Mean Uncertainty | U (Correct Pred) | U (Incorrect Pred) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BiLSTM | 0.8474 | 0.8258 | 0.9621 | 0.7234 | 0.0285 | 0.8624 | 0.2440 | 0.2178 | 0.3893 |
| CNN | 0.8474 | 0.8263 | 0.9588 | 0.7260 | 0.0312 | 0.8669 | — | — | — |
| BCAM-Net | 0.8474 | 0.8258 | 0.9621 | 0.7234 | 0.0285 | 0.8624 | 0.2440 | 0.2178 | 0.3893 |
| BiLSTM + ResNet18 + BCA | — | — | — | — | — | — | — | — | — |

---

## 表 6：Test Set 2 对比

该表单独汇总各模型在 Test Set 2 上的表现，适合比较模型在未知调制或盲检测任务中的泛化能力。

| Method | ACC | F1-score | Precision | Recall(Pd) | Pfa | AUC | Mean Uncertainty | U (Correct Pred) | U (Incorrect Pred) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BiLSTM | 0.8769 | 0.8638 | 0.9669 | 0.7806 | 0.0267 | 0.9039 | 0.2479 | 0.2224 | 0.4298 |
| CNN | 0.8781 | 0.8658 | 0.9628 | 0.7866 | 0.0304 | 0.9082 | — | — | — |
| BCAM-Net | 0.8769 | 0.8638 | 0.9669 | 0.7806 | 0.0267 | 0.9039 | 0.2479 | 0.2224 | 0.4298 |
| BiLSTM + ResNet18 + BCA | — | — | — | — | — | — | — | — | — |

---

## 表 7：方法级别的已填写结果概览

该表按方法归纳已填写的数据情况，便于观察哪些模型已经完成实验，哪些模型仍需补充结果。

| Method | 已填写数据集 | 未填写数据集 | 备注 |
|---|---|---|---|
| BiLSTM | Validation / Test Set 1 / Test Set 2 | 无 | 分类指标和不确定性指标均已填写 |
| CNN | Validation / Test Set 1 / Test Set 2 | 不确定性指标未填写 | 已填写分类性能指标 |
| BCAM-Net | Validation / Test Set 1 / Test Set 2 | 无 | 分类指标和不确定性指标均已填写 |
| BiLSTM + ResNet18 + BCA | 无 | Validation / Test Set 1 / Test Set 2 | 所有指标待补充 |

