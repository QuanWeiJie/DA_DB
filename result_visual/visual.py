# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from openTSNE import TSNE
with open('20x20mean13-mp_new2.txt','r') as f:
    lines = []
    for ff in f.readlines()[:200]:
        line = []
        a = ff.strip().split(' ')
        for aa in a:
            line.append(float(aa))
        lines.append(line)
data_x = np.array(lines)
# embeding = TSNE().fit(data_x)
plt.rcParams['font.family'] = 'SimSun'
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(data_x[:100,0],data_x[:100,1],c='#FD512A')
ax.scatter(data_x[100:,0],data_x[100:,1],c='#0057DA')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.legend(['ICDAR2013','铭牌'],loc='upper center',fontsize=12)
# plt.title("数据分布对比")
plt.savefig("da_db++icdar13_mp.png")
