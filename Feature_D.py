import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import scienceplots
from Preprocess_AAIndexdata import extract_node_feature
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
# print(plt.style.available)
plt.style.use('nature')
from scipy.stats import pearsonr
#
feature_number = pd.read_excel(r'E:\paper_datasets\Amyloid_Database\AAIndex_data.xlsx',header = [0])
feature = extract_node_feature(feature_number)
feat = []

for i in range(len(feature)):
    feat.append(feature[i][1:])
    x_np = np.array(feat)
X = x_np.transpose()

fcl = []
for i in range(531):
    for j in range(531):
        fc = pearsonr(x_np[i,:],x_np[j,:])
        if (abs(fc[0]) > 0.8 and fc[1] < 0.005 and i != j):
            fcl.append(fc)
            # print(i,j)
# print(len(fcl))
fcl.append()



#TSNE降维
tsne = TSNE(n_components=3)
X_tsne = tsne.fit_transform(x_np)
x = X_tsne[:,0]
y = X_tsne[:,1]
z = X_tsne[:,2]
cm = plt.cm.get_cmap('tab20')
color = np.random.rand(531)
sns.set(style = 'darkgrid')
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111,projection = '3d')
ax1 = plt.gca()
ax1.spines['top'].set_visible(True)
ax1.spines['right'].set_visible(True)
ax1.spines['bottom'].set_visible(True)
ax1.spines['left'].set_visible(True)
ax1.spines['top'].set_linewidth('2')
ax1.spines['right'].set_linewidth('2')
ax1.spines['left'].set_linewidth('2')
ax1.spines['bottom'].set_linewidth('2')
ax.set_xlabel('TSNE1',fontdict = {'family' : 'Times New Roman','size' : 18})
ax.set_ylabel('TSNE2',fontdict = {'family' : 'Times New Roman','size' : 18})
ax.set_zlabel('TSNE3',fontdict = {'family' : 'Times New Roman','size' : 18})

sc = ax.scatter(x,y,z,c = color,cmap = cm,s = 70,alpha = 1)
cax = fig.add_axes([ax.get_position().x1 + 0.15,ax.get_position().y0,0.04,ax.get_position().height])
cb= plt.colorbar(sc,cax =cax,fraction = 0.1)
cb.set_label('AAindex feature',fontdict = {'family' : 'Times New Roman','size' : 18})
# plt.tight_layout()
plt.xticks(fontproperties = 'Times New Roman',fontsize = 17)
plt.yticks(fontproperties = 'Times New Roman',fontsize = 17)
# plt.title('Feature Correlation',x = ax.get_position().x1 - 1,y = ax.get_position().height + 0.1,fontdict = {'family' : 'Times New Roman','size' : 20,'weight' : 'normal'})
ax.tick_params(labelsize = 18)
# # plt.savefig(r'E:\paper_datasets\作图\feature_decomposition8.png',dpi = 2000,bbox_inches = 'tight')
plt.show()
