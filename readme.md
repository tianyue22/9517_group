# 大纲

## 1. Introduction

第一张写半页以内，说清楚就行

### 1.1 背景

农业场景中图像分割的重要性。强调：精确的作物分割是农业自动化的重要基础

### 1.2 问题定义

- 任务定义，主要是数据集和目标是什么
- 输入输出形式（RGB 图像 -> mask）
- 评价指标+其公式定义
- 提一嘴train，valid和test已经给出，不做处理

## 2. Literature Review

每个点（除了最后一个）都需要有论文支撑，把论文放reference里

- 定义农业场景分割任务 + 难点
- 传统方法是如何做的
- 机器学习方法是如何做的（xgboost，rf）
- 深度学习是如何做的（unet）
- 基于上述，我们选择了哪些方法，大致如下：

传统方法作为**特征工程基线**，采用 RGB+ExG+LBP 表征并结合形态学与小连通域过滤后处理，重点分析了去噪带来的 Precision-Recall 权衡及在 noise/blur/low-light 下的稳定性。机器学习方法作为**监督学习中间层基线**，在相同特征与数据划分下对比 Random Forest 与 XGBoost，并进一步比较类别不平衡策略与阈值设置对 F1/IoU 的影响。深度学习方法作为**端到端表示学习方案**，系统对比了 U-Net 与 DeepLabV3，并开展阈值、损失函数与训练策略消融；结果表明 U-Net 在主指标和鲁棒性上整体优于 DeepLabV3，而后者可通过阈值调节精确率与召回率但提升有限。（**这段我ai的，差不多是这些意思**）

## 3. methods

### 3.1 传统方法

特征工程：RGB、ExG、LBP

后处理：morphology (open + close)、小连通域过滤

### 3.2 机器学习方法

模型：Random Forest， XGBoost

类别不平衡策略：类权重、重采样

### 3.3 深度学习方法

模型：U-Net，DeepLabV3

损失函数：baseline，Weighted BCE + Dice

### 3.4 其他

**这里提一嘴就行，突出做了什么东西**

此外，我们还进行了补充实验以验证方法稳定性与可调性：在 noise/blur/low-light（深度学习部分另含 occlusion）扰动下开展鲁棒性测试，并通过概率二值化阈值扫描分析 Precision-Recall 权衡；机器学习与深度学习部分还分别评估了类别不平衡处理与训练策略/损失函数对最终 F1/IoU 的影响。

我们还进行了**特征消融**（RGB/ExG/LBP 组合）、**后处理消融**（morphology 与小连通域过滤及面积阈值扫描）、**模型对比**（RF vs XGBoost，U-Net vs DeepLabV3）以及**损失函数与训练策略消融**（深度学习部分）；同时统计了 Train/Test/Inference Time，并结合失败案例分析不同方法在复杂场景下的误差模式。

