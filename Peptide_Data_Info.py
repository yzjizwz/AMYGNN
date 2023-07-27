import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import matplotlib.cm as cm
from sklearn import manifold,datasets
import umap
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
import matplotlib as mpl
from matplotlib.pyplot import MultipleLocator
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as ticker
import scienceplots
import seaborn as sns
mpl.rcParams["font.sans-serif"]=["Times New Roman"]
mpl.rcParams["axes.unicode_minus"]=False
plt.rcParams['font.size'] = 13
plt.style.use('nature')


peptide_data = pd.read_excel(r"E:\paper_datasets\Amyloid_Database\CPAD2.0_Data\aggregating_peptides_dropnan.xlsx",header = [0])
x_info = peptide_data['Length'].value_counts()
peptide_length_list = list(set(peptide_data['Length']))

label,sub_peptide_length_list,sub_label,s,sub_s = [],[],[],[],[]
for m in range(0,len(peptide_length_list)):
    label.append(x_info[peptide_length_list[m]])
# print(len(label),len(peptide_length_list))
# print(label)

for i in range(0,len(peptide_length_list)):
    if x_info[peptide_length_list[i]] <= 25:
        s.append(x_info[peptide_length_list[i]] * 10)
    else:
        s.append(x_info[peptide_length_list[i]] / 2)
    if peptide_length_list[i] != 6:
        sub_peptide_length_list.append(peptide_length_list[i])
        if x_info[peptide_length_list[i]] <= 5:
            sub_s.append(x_info[peptide_length_list[i]] * 100)
            sub_label.append(x_info[peptide_length_list[i]])
        elif x_info[peptide_length_list[i]] <= 25 and x_info[peptide_length_list[i]] > 5:
            sub_s.append(x_info[peptide_length_list[i]] * 20)
            sub_label.append(x_info[peptide_length_list[i]])
        else:
            sub_s.append(x_info[peptide_length_list[i]] * 5)
            sub_label.append(x_info[peptide_length_list[i]])

# print(len(s),s)
# print(len(peptide_length_list),peptide_length_list)
# print(len(label),label)
# print(len(sub_s),sub_s)
# print(len(sub_peptide_length_list),sub_peptide_length_list)
# print(len(sub_label),sub_label)

color = np.random.rand(len(peptide_length_list))
color = cm.tab20(np.random.rand((len(peptide_length_list))))
# sub_color = np.random.rand(len(sub_peptide_length_list))
sub_color = cm.Set1(np.random.rand(len(sub_peptide_length_list)))
font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size' : 35,
        }
fig,ax = plt.subplots(1,1,figsize = (8,8))
x_major_locator=MultipleLocator(10)
y_major_locator=MultipleLocator(100)
ax1 =plt.gca()
ax1.spines['top'].set_visible(True)
ax1.spines['right'].set_visible(True)
ax1.spines['bottom'].set_visible(True)
ax1.spines['left'].set_visible(True)
ax1.spines['top'].set_linewidth('2')
ax1.spines['right'].set_linewidth('2')
ax1.spines['left'].set_linewidth('2')
ax1.spines['bottom'].set_linewidth('2')
asc = ax.scatter(peptide_length_list,label,s = s,c = peptide_length_list,cmap = 'tab20',alpha = 0.7)
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.set_xlim(0,80)
ax.set_ylim(-20,1200)
ax.tick_params(axis = 'both',labelsize = 20)
axins = inset_axes(ax, width="70%", height="60%",loc='lower left',
                   bbox_to_anchor=(0.2, 0.3,1,1),
                   bbox_transform=ax.transAxes)
ax1 =plt.gca()
ax1.spines['top'].set_visible(True)
ax1.spines['right'].set_visible(True)
ax1.spines['bottom'].set_visible(True)
ax1.spines['left'].set_visible(True)
ax1.spines['top'].set_linewidth('2')
ax1.spines['right'].set_linewidth('2')
ax1.spines['left'].set_linewidth('2')
ax1.spines['bottom'].set_linewidth('2')
axins.scatter(sub_peptide_length_list,sub_label,s = sub_s,c = sub_peptide_length_list,cmap = 'tab20',alpha = 0.6)
axins.set_xlim(0,80)
axins.set_ylim(-10,150)
axins.tick_params(axis = 'both',labelsize = 25)
sx = [0,80,80,0,0]
sy = [-10,-10,150,150,-10]
ax.plot(sx,sy,'r',linewidth = 3)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec='r', lw=3)
ax.set_xlabel('Peptide Length',fontdict = {'family' : 'Times New Roman','weight' : 'normal','size' : 35},labelpad = 10)
ax.set_ylabel('Peptide  Count  Of  Differnet  Length',fontdict = {'family' : 'Times New Roman','weight' : 'normal','size' : 35},labelpad = 10)
ax.set_title('Peptide Dataset Information',fontdict = font,y = 1.03)
ax.tick_params(axis = 'x',labelsize = 30)
ax.tick_params(axis = 'y',labelsize = 30,rotation = 45)
cax = fig.add_axes([ax.get_position().x1 + 0.06,ax.get_position().y0,0.05,ax.get_position().height])
cb = plt.colorbar(asc,cax = cax,fraction = 0.05,aspect= 50)
tick_locator = ticker.MaxNLocator(nbins = 9)  # colorbar上的刻度值个数
cb.locator = tick_locator
cb.set_ticks([np.min(peptide_length_list),10,20,30,40,50,60,70,np.max(peptide_length_list)])
cb.update_ticks()
cb.ax.tick_params(labelsize=30)
plt.savefig(r"E:\paper_datasets\作图\peptide_dataset_information.png",dpi = 2000,bbox_inches = 'tight')
# plt.show()
# plt.figure(figsize = (6,6))
# plt.scatter(peptide_length_list,label,s = s,c = color,alpha = 0.7)
# plt.xlabel('Peptide Length',font)
# plt.ylabel('Peptide  Count  Of  Different  Length',font)
# plt.title('Peptide Dataset Information',font)
# plt.savefig(r"E:\paper_datasets\作图\peptide_dataset_information.png",dpi = 1000,bbox_inches = 'tight')
# plt.show()

# sub_ax = fig.add_subplot(1,2,2)
# plt.scatter(sub_peptide_length_list,sub_label,s = sub_s,c = sub_color,alpha=0.7)
# plt.xlabel('Sub  Peptide  Length',font)
# plt.ylabel('Sub  Peptide  Count  Of  Different  Length',font)
# plt.title('Sub Peptide Dataset Information',font)
# plt.savefig(r"E:\paper_datasets\作图\sub_peptide_dataset_information.png",dpi = 1000,bbox_inches = 'tight')
# plt.show()

# plt.scatter(peptide_length_list,label,s = s,c = color)
# plt.xlabel('Peptide Length Informations')
# plt.ylabel('Peptide Length Count Informations')
# plt.title('Peptide Dataset Informations')
# plt.show()

# H = plt.hist2d(peptide_length_list,label,bins = 41,cmap = cm.tab20c)
# plt.xlabel('Peptide Length Informations')
# plt.ylabel('Peptide Length Count Informations')
# plt.colorbar(H[3])
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# # H = ax.hexbin(x_list,labels,gridsize = 20,extent = [2,80,0,1110],cmap = cm.YlGnBu)
# H = ax.hexbin(peptide_length_list,label,gridsize = 20,extent = [2,80,0,1110],cmap = cm.tab20b)
# ax.set_title('Peptide Length Informations')
# ax.set_xlabel('Peptide_length')
# ax.set_ylabel('Length_Count')
# fig.colorbar(H,ax = ax)
# plt.show()

# class_dict = {'Amyloid' : 1,'Non-amyloid' : 0}
# class_info,peptide_length = [],[]
# for i in range(len(peptide_data['Entry'])):
#     class_info.append(class_dict[peptide_data['Classification'][i].capitalize()])
#     peptide_length.append(int(peptide_data['Length'][i]))
# peptide_info = np.array([peptide_length,class_info]).transpose()
# y = manifold.TSNE(n_components = 2,init ='pca',random_state = 0).fit_transform(peptide_info)
# color = np.random.rand(1575)
# plt.scatter(y[:,0],y[:,1],c = color,cmap = plt.cm.YlGnBu)
# plt.title('Peptide Dataset Information TSNE')
# plt.xlabel('Peptide Length')
# plt.ylabel('Peptide Class Information')
# plt.savefig(r"E:\paper_datasets\作图\Peptide_Dataset_TSNE.png",dpi = 1000)
# plt.show()
##绘制多肽数据库的饼图表征数据的来源信息
# group_info = peptide_data.groupby('Source (Waltz-DB,CPAD,AmyLoad,Waltz)').size()
# group = np.array(group_info)
# label,la,indices = [],[],[]
# labels = list(group_info.keys())
# for i in range(len(labels)):
#     for j in range(len(labels[i])):
#         la.append(labels[i][j].strip(' '))
#     label.append([l for l in la])
#     la.clear()
# indices = [i for i in range(len(labels))]
# for j in range(len(labels)):
#     labels[j] = str(j) + ':' + labels[j]
# print(labels)
#
# colors = cm.tab20c(np.random.random(12))
# inner_colors = cm.Set3(np.random.random(12))
# plt.figure(figsize = (8,8))
# plt.pie(group,radius=1,labels = indices,colors = colors,
#         explode= (0,0.1,0,0,0.06,0.25,0,0.1,0,0,0.1),
#         shadow = False,autopct = "%0.2f%%",textprops={'fontsize' : 30,'color' : 'black','family' : 'Times New Roman'},
#         wedgeprops={'width' : 0.4,'edgecolor' : 'lightgray'},
#         startangle = 20,labeldistance = 0.82,
#         pctdistance = 1.23,
#         rotatelabels = False
#         )
# plt.pie(group,radius = 0.7,colors = inner_colors,
#         # explode = (0,0,0,0,0,0,0,0,0,0,0),
#         textprops={'fontsize':10,'color':'w'},
#         wedgeprops={'width' : 0.4,'edgecolor':'w'}
#         )
# plt.axis('equal')
# plt.legend(facecolor = 'white',bbox_to_anchor=(2,0.25),loc = 'lower right',labels = labels,
#            shadow = False,fancybox = True,markerscale = 1.3,prop = {'family' : 'Times New Roman','size':18},frameon = False)
# plt.title('Peptide Source Information',loc = 'center',fontsize = 28,fontweight = 'medium',fontdict = {'family':'Times New Roman'},y= 1.08)
# plt.savefig(r"E:\paper_datasets\作图\Peptide_Dataset_Info.png",dpi = 1000, bbox_inches="tight")

# reducer = umap.UMAP(random_state = 42)
# embedding = reducer.fit_transform(np.array(peptide_data['Length']).reshape(-1,1))
# x = embedding[:,0]
# y = embedding[:,1]
#
# cm = plt.cm.get_cmap('Set2')
# color = np.random.rand(1574)
# sns.set(style = 'darkgrid')
# fig = plt.figure(figsize = (10,8))
# ax = fig.add_subplot(111)
# ax.set_xlabel('UMAP1')
# ax.set_ylabel('UMAP2')
#
# sc = ax.scatter(x,y,c = color,cmap = cm,s = 15)
# cb= plt.colorbar(sc)
# cb.set_label('Peptide Length Informations')
# plt.tight_layout()
# plt.savefig(r"E:\paper_datasets\作图\Peptide_Dataset_LengthInformations.png",dpi = 1000, bbox_inches="tight")
plt.show()

