# COMP9517 项目结果总结（深度学习模块）

## 1. 模块范围
本模块实现并评估两种深度学习方法：
- U-Net
- DeepLabV3 (MobileNetV3 backbone)

数据使用官方 `train / validation / test` 划分，训练和调参不使用 test，避免数据泄漏。

## 2. 主结果对比

| model | precision | recall | f1 | iou | train_time_sec | test_infer_time_sec | infer_time_per_image_ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| U-Net | 0.831785 | 0.833447 | 0.820546 | 0.714028 | 2386.560213 | 3.542335 | 147.597293 |
| DeepLabV3 | 0.653851 | 0.777154 | 0.705393 | 0.566896 | 340.446103 | 4.193665 | 174.736043 |

结论：U-Net 在当前设置下明显优于 DeepLabV3（IoU/F1 更高）。

## 3. 误差分桶

| model | bucket | count | precision | recall | f1 | iou |
|---|---|---:|---:|---:|---:|---:|
| U-Net | 整体 | 24 | 0.831785 | 0.833447 | 0.820546 | 0.714028 |
| U-Net | 光照差 | 10 | 0.880549 | 0.902812 | 0.882325 | 0.798242 |
| U-Net | 遮挡重(代理) | 0 | NaN | NaN | NaN | NaN |
| U-Net | 背景复杂 | 2 | 0.783712 | 0.819402 | 0.800915 | 0.668386 |
| DeepLabV3 | 整体 | 24 | 0.653851 | 0.777154 | 0.705393 | 0.566896 |
| DeepLabV3 | 光照差 | 10 | 0.762391 | 0.893133 | 0.820709 | 0.702513 |
| DeepLabV3 | 遮挡重(代理) | 0 | NaN | NaN | NaN | NaN |
| DeepLabV3 | 背景复杂 | 2 | 0.600223 | 0.768508 | 0.673811 | 0.508450 |

说明：当前阈值定义下“遮挡重(代理)”样本数为 0，因此该桶为 NaN。

## 4. 鲁棒性实验（失真后重评估）

### 4.1 绝对指标

| model | distortion | precision | recall | f1 | iou |
|---|---|---:|---:|---:|---:|
| U-Net | none | 0.831785 | 0.833447 | 0.820546 | 0.714028 |
| DeepLabV3 | none | 0.653851 | 0.777154 | 0.705393 | 0.566896 |
| U-Net | gaussian_noise | 0.930135 | 0.435530 | 0.573374 | 0.423102 |
| DeepLabV3 | gaussian_noise | 0.756309 | 0.403404 | 0.491979 | 0.352894 |
| U-Net | blur | 0.820127 | 0.838996 | 0.816158 | 0.708023 |
| DeepLabV3 | blur | 0.650823 | 0.777136 | 0.703175 | 0.564138 |
| U-Net | low_light_contrast | 0.842506 | 0.668620 | 0.728735 | 0.593436 |
| DeepLabV3 | low_light_contrast | 0.632491 | 0.737815 | 0.673937 | 0.525862 |
| U-Net | partial_occlusion | 0.751379 | 0.811472 | 0.775146 | 0.657428 |
| DeepLabV3 | partial_occlusion | 0.571690 | 0.779236 | 0.651244 | 0.518359 |

### 4.2 IoU 掉点（相对 clean）

| model | distortion | iou | iou_drop_vs_clean |
|---|---|---:|---:|
| U-Net | none | 0.714028 | 0.000000 |
| U-Net | gaussian_noise | 0.423102 | 0.290926 |
| U-Net | blur | 0.708023 | 0.006005 |
| U-Net | low_light_contrast | 0.593436 | 0.120592 |
| U-Net | partial_occlusion | 0.657428 | 0.056600 |
| DeepLabV3 | none | 0.566896 | 0.000000 |
| DeepLabV3 | gaussian_noise | 0.352894 | 0.214003 |
| DeepLabV3 | blur | 0.564138 | 0.002758 |
| DeepLabV3 | low_light_contrast | 0.525862 | 0.041034 |
| DeepLabV3 | partial_occlusion | 0.518359 | 0.048537 |

结论：两种模型均对高斯噪声最敏感，对 blur 最稳健。

## 5. 成功/失败案例
两种方法都已完成：
- 成功案例 Top-3（原图/GT/Pred）
- 失败案例 Bottom-3（原图/GT/Pred）

观察：
- U-Net 对细长叶片和边界保持更稳定。
- 两者在前景很少、背景复杂样本上更易漏检。

## 6. DeepLabV3 改进与消融

### 6.1 阈值消融（不重训，CSV）
来源：`results/deeplabv3_threshold_ablation.csv`

| threshold | val_iou | val_f1 | test_iou | test_f1 | test_precision | test_recall |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.547408 | 0.690408 | 0.566896 | 0.705393 | 0.653851 | 0.777154 |
| 0.4 | 0.544945 | 0.689483 | 0.560821 | 0.701151 | 0.621122 | 0.819113 |
| 0.6 | 0.537723 | 0.680782 | 0.562887 | 0.701554 | 0.687030 | 0.726676 |
| 0.3 | 0.534032 | 0.680906 | 0.546600 | 0.690106 | 0.587811 | 0.854209 |
| 0.7 | 0.512324 | 0.656829 | 0.544943 | 0.685862 | 0.721571 | 0.662765 |

结论：当前最优阈值为 0.5。

### 6.2 Weighted Loss 改进对比（CSV）
来源：`results/deeplabv3_weighted_loss_compare.csv`

| model | setting | pos_weight | best_threshold | precision | recall | f1 | iou | train_time_sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DeepLabV3_baseline | th=0.5 | 1.000000 | 0.5 | 0.653851 | 0.777154 | 0.705393 | 0.566896 | 340.446103 |
| DeepLabV3_baseline | th=best | 1.000000 | 0.5 | 0.653851 | 0.777154 | 0.705393 | 0.566896 | 340.446103 |
| DeepLabV3_weighted | th=0.5 | 2.971902 | 0.6 | 0.620279 | 0.825041 | 0.704105 | 0.563385 | 289.396821 |
| DeepLabV3_weighted | th=best | 2.971902 | 0.6 | 0.654975 | 0.773264 | 0.705732 | 0.566636 | 289.396821 |

结论：weighted loss 在当前实验下主要带来 precision-recall 权衡，IoU 提升不显著（与 baseline 基本持平）。

### 6.3 训练策略消融（CSV）
来源：`results/deeplabv3_strategy_ablation.csv`

| exp_name | train_augment | sampler_mode | lr | epochs | precision | recall | f1 | iou |
|---|---|---|---:|---:|---:|---:|---:|---:|
| aug_off_lr3e4_rand | False | random | 3e-4 | 8 | 0.627110 | 0.773399 | 0.687306 | 0.544402 |
| aug_on_lr1e3_rand | True | random | 1e-3 | 8 | 0.688253 | 0.694552 | 0.686659 | 0.541450 |
| base_aug_on_lr3e4_rand | True | random | 3e-4 | 8 | 0.648454 | 0.723751 | 0.677092 | 0.532535 |
| aug_on_lr3e4_weightedfg | True | weighted_fg | 3e-4 | 8 | 0.613221 | 0.757414 | 0.673347 | 0.527500 |
| aug_on_lr1e4_rand | True | random | 1e-4 | 8 | 0.526867 | 0.670316 | 0.582076 | 0.436306 |

结论：短预算下 3e-4~1e-3 更合适，1e-4 明显欠收敛；weighted_fg 采样未带来收益。

## 7. 完成项对照
- [x] 两方法主结果（P/R/F1/IoU）
- [x] 两方法训练/测试时间
- [x] 成功/失败案例
- [x] 误差分桶
- [x] 鲁棒性实验（多失真）
- [x] DeepLabV3 改进并验证（阈值 + weighted loss）
- [x] DeepLabV3 训练策略消融
- [ ] 多 seed 均值±方差（未做）
- [ ] 标注质量实验（未做）

## 8. 总结（可直接放报告）
在 EWS 小麦分割任务中，U-Net 在当前实验设置下取得最佳综合性能（IoU 0.714），明显优于 DeepLabV3（IoU 0.567）。鲁棒性实验显示两种方法均对高斯噪声最敏感，对模糊最稳健。针对 DeepLabV3 的改进（weighted loss、阈值与训练策略消融）验证了其对配置较敏感，但在当前预算下尚未获得稳定且显著的 IoU 提升。整体而言，本模块已经完成了方法对比、场景误差分析、鲁棒性评估与改进验证的完整证据链。
