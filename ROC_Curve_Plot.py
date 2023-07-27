import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import *
import matplotlib as mpl
import scienceplots
mpl.rcParams["font.sans-serif"]=["Times New Roman"]
mpl.rcParams["axes.unicode_minus"]=False
plt.style.use('nature')


font = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 25,
        }
#绘制各层GNN对模型的影响
fpr1 = pd.read_csv(r'E:\paper_datasets\AMYGNN_One_layer_Fpr_num_0.7513661202185792.csv',header = None,delimiter = '\t',sep ='\n')
fpr2 = pd.read_csv(r'E:\paper_datasets\AMYGNN_Two_layer_Fpr_num_0.8224043715846995.csv',header = None,delimiter = '\t',sep ='\n')
fpr3 = pd.read_csv(r'E:\paper_datasets\ROC\AMYGNN_Three_layer_Fpr_num_0.9208result_model_235_0.9443573667711599_0.9074074074074074.pt.csv',header = None,delimiter = '\t',sep ='\n')
tpr1 = pd.read_csv(r'E:\paper_datasets\AMYGNN_One_layer_Tpr_num_0.7513661202185792.csv',header = None,delimiter = '\t',sep ='\n')
tpr2 = pd.read_csv(r'E:\paper_datasets\AMYGNN_Two_layer_Tpr_num_0.8224043715846995.csv',header = None,delimiter = '\t',sep ='\n')
tpr3 = pd.read_csv(r'E:\paper_datasets\ROC\AMYGNN_Three_layer_Tpr_num_0.9208result_model_235_0.9443573667711599_0.9074074074074074.pt.csv',header = None,delimiter = '\t',sep ='\n')
#绘制三层GNN情况下的roc曲线
# fpr1 = pd.read_csv(r'E:\paper_datasets\ROC\AMYGNN_Three_layer_Fpr_num_0.9153result_model_299_0.890282131661442_0.9074074074074074.pt.csv',header = None,delimiter = '\t',sep ='\n')
# fpr2 = pd.read_csv(r'E:\paper_datasets\ROC\AMYGNN_Three_layer_Fpr_num_0.9208result_model_235_0.9443573667711599_0.9074074074074074.pt.csv',header = None,delimiter = '\t',sep ='\n')
# fpr3 = pd.read_csv(r'E:\paper_datasets\ROC\AMYGNN_Three_layer_Fpr_num_0.9399result_model_237_0.9294670846394985_0.9074074074074074.pt.csv',header = None,delimiter = '\t',sep ='\n')
# tpr1 = pd.read_csv(r'E:\paper_datasets\ROC\AMYGNN_Three_layer_Tpr_num_0.9153result_model_299_0.890282131661442_0.9074074074074074.pt.csv',header = None,delimiter = '\t',sep ='\n')
# tpr2 = pd.read_csv(r'E:\paper_datasets\ROC\AMYGNN_Three_layer_Tpr_num_0.9208result_model_235_0.9443573667711599_0.9074074074074074.pt.csv',header = None,delimiter = '\t',sep ='\n')
# tpr3 = pd.read_csv(r'E:\paper_datasets\ROC\AMYGNN_Three_layer_Tpr_num_0.9399result_model_237_0.9294670846394985_0.9074074074074074.pt.csv',header = None,delimiter = '\t',sep ='\n')
fpr_1, tpr_1, fpr_2, tpr_2, fpr_3, tpr_3 = [], [], [], [], [], []

for i in range(len(fpr1[0])):
    fpr_1.append(fpr1[0][i].split(','))
    for j in range(len(fpr_1[i])):
        fpr_1[i][j] = float(fpr_1[i][j])
for i in range(len(tpr1[0])):
    tpr_1.append(tpr1[0][i].split(','))
    for j in range(len(tpr_1[i])):
        tpr_1[i][j] = float(tpr_1[i][j])

for i in range(len(fpr2[0])):
    fpr_2.append(fpr2[0][i].split(','))
    for j in range(len(fpr_2[i])):
        fpr_2[i][j] = float(fpr_2[i][j])
for i in range(len(tpr2[0])):
    tpr_2.append(tpr2[0][i].split(','))
    for j in range(len(tpr_2[i])):
        tpr_2[i][j] = float(tpr_2[i][j])

for i in range(len(fpr3[0])):
    fpr_3.append(fpr3[0][i].split(','))
    for j in range(len(fpr_3[i])):
        fpr_3[i][j] = float(fpr_3[i][j])
for i in range(len(tpr3[0])):
    tpr_3.append(tpr3[0][i].split(','))
    for j in range(len(tpr_3[i])):
        tpr_3[i][j] = float(tpr_3[i][j])

save_fig_path = 'E:\paper_datasets\作图\\new_'
path = save_fig_path + '_' + 'error_bar' + 'roc.png'
path1 = save_fig_path + '_' + '_one_layer' + '.png'
path2 = save_fig_path + '_' + '_two_layers' + '.png'
path3 = save_fig_path + '_' + '_three_layers' + '.png'
plt.figure(figsize=(7, 7))
ax1 = plt.gca()
ax1.spines['top'].set_visible(True)
ax1.spines['right'].set_visible(True)
ax1.spines['bottom'].set_visible(True)
ax1.spines['left'].set_visible(True)
ax1.spines['top'].set_linewidth('2')
ax1.spines['right'].set_linewidth('2')
ax1.spines['left'].set_linewidth('2')
ax1.spines['bottom'].set_linewidth('2')
# roc_area_1 = auc(fpr_1[0], tpr_1[0])
# roc_area_2 = auc(fpr_2[0], tpr_2[0])
# roc_area_3 = auc(fpr_3[0], tpr_3[0])
roc_area_1 = auc(fpr_1[0], tpr_1[0])
roc_area_2 = auc(fpr_2[0], tpr_2[0])
roc_area_3 = auc(fpr_3[0], tpr_3[0])
sum = 255
# plt.plot(fpr_1[0], tpr_1[0], color = (33 /sum,158/sum,188 /sum), lw=3.5, label='accuracy = 0.9153 (area = %0.4f)' % roc_area_1)
# plt.plot(fpr_2[0], tpr_2[0], color = (254/sum,183/sum,5/sum), lw=3.5, label='accuracy = 0.9208 (area = %0.4f)' % roc_area_2)
# plt.plot(fpr_3[0], tpr_3[0], color = (250/sum,134/sum,0), lw=3.5, label='accuracy = 0.9399 (area = %0.4f)' % roc_area_3)
# plt.plot(fpr_1[0], tpr_1[0], color = (0.99,0.70,0.68), lw=3.5, label='One Layer GNN (area = %0.4f)' % roc_area_1)
# plt.fill_between(fpr_1[0],[i + 0.05 for i in tpr_1[0]],[i - 0.05 for i in tpr_1[0]] ,color = (0.99,0.70,0.68),alpha = 0.4)
# plt.plot(fpr_2[0], tpr_2[0], color = (0.082,0.592,0.647), lw=3.5, label='Two Layers GNN (area = %0.4f)' % roc_area_2)
# plt.fill_between(fpr_2[0],[i + 0.05 for i in tpr_2[0]],[i - 0.05 for i in tpr_2[0]] ,color = (0.082,0.592,0.647),alpha = 0.4)
plt.plot(fpr_3[0], tpr_3[0], color = (0.05,0.38,0.42), lw=3.5, label='Three Layers GNN (area = %0.4f)' % roc_area_3)
plt.fill_between(fpr_3[0],[i + 0.05 for i in tpr_3[0]],[i - 0.05 for i in tpr_3[0]] ,color = (0.05,0.38,0.42),alpha = 0.4)
plt.plot([0, 1], [0, 1], color = (0.98,0.44,0.41), linestyle='--', lw=3.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Fasle Positive Rate',fontdict = {'family':'Times New Roman','weight' : 'normal','size' : 30},labelpad = 9)
plt.ylabel('True Positive Rate',fontdict= {'family':'Times New Roman','weight':'normal','size' : 30},labelpad = 9)
plt.xticks(fontproperties = 'Times New Roman',fontsize = 25)
plt.yticks(fontproperties = 'Times New Roman',fontsize = 25)
plt.title('AMYGNN Roc Curve',font,y = 1.02)
plt.legend(bbox_to_anchor = (1.02,-0.02),loc='lower right',prop = {'family':'Times New Roman','size' : 20},frameon = False)
plt.tick_params(labelsize=25)
# plt.savefig(path, dpi=1000, bbox_inches='tight')
# plt.savefig(path1, dpi=1000, bbox_inches='tight')
# plt.savefig(path2, dpi=1000, bbox_inches='tight')
# plt.savefig(path3, dpi=1000, bbox_inches='tight')
plt.show()



