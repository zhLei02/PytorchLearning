---
title: 手写字体识别
date: 2024-11-19
---

## 快速开始

    均需在MNIST目录下使用终端运行

1. 环境配置
    `pip install -r requirements.txt`

2. 运行
    `python .\main.py --exp_name try`

3. 测试
    `python .\main.py --exp_name try --test`

4. TensorBoard监控
    `tensorboard --logdir=<log>` **注**：\<log> 为日志路径  
    > TensorBoard教程参考：[TensorBoard最全使用教程](https://blog.csdn.net/qq_41656402/article/details/131123121)

## 目录结构

+ config/ 存储模型配置等参数文件
+ data/ 数据存储位置
+ dataset/ 读取数据集
+ model/ 模型定义
+ proj_log/ 运行日志
+ trainer/ 训练器
+ utils/ 工具函数
+ main.py 主函数

> 代码为参考他人代码完成，仅供自我学习使用。
