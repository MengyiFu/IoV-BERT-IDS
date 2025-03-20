import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

labels = [
    'BENIGN',
    'Hulk',
    'PortScan',
    'DDoS',
    'GoldenEye',
    'FTP',
    'SSH',
    'Slowloris',
    'Slowhttptest',
    'Bot',
    'Brute Force',
    'XSS'
    'Infiltration',
    'Sql Injection',
    'Heartbleed'
]

result = pd.read_csv('../result/CICIDS/VAE_CNN/class15_pred.csv')
y_pred = result['y_pred']
y_true = result['y_true']
# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred, normalize='true')

# 绘制热力图
plt.figure(figsize=(24, 30), dpi=500)
# 绘制混淆矩阵热力图
sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f', square=True, xticklabels=True, yticklabels=True)
plt.title('VAE_CNN')
# # 设置中文字体
# plt.rcParams['font.sans-serif'] = 'simhei'
# 设置刻度标签
tick_marks = np.arange(len(labels)) + 0.5
plt.xticks(tick_marks, labels, fontsize=12, rotation=45)
plt.yticks(tick_marks, labels, fontsize=12, rotation=360)
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')

plt.show()