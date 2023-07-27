import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots
mpl.rcParams["font.sans-serif"]=["SimHei"]
mpl.rcParams["axes.unicode_minus"]=False
from  mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
from matplotlib.pyplot import MultipleLocator
# print(plt.style.available)
plt.style.use('nature')

#画对比结果的代码
indicator = ['ACC','Sn','Sp','MCC','F1-Score']
amygcn = [0.9556,0.9545,0.8696,0.9145,0.9552]
iamy_scm = [0.4873,0.6288,0.3846,0.0136,0.5141]
bioseq_svm = [0.7771,0.7652,0.7857,0.5471,0.7572]
amypred =[0.5732,0.9621,0.2912,0.3223,0.3710]
iamydc = [0.7681,0.68,1,0.6074,0.8095]
# colorlist = ['#1E2C6C','#1C3C94','#0B66B0','#00A9BD','#40B58E']
acc = [0.8065,0.4348,0.5053,0.6874,0.7101,0.7259,0.7287,0.7681]
sn = [0.7222,0.5,0.521,0.881,0.98,0.8349,0.8851,0.68]
sp = [0.9231,0.2631,0.4681,0.2449,0,0.4714,0.3556,1]
g_mean = []
for i in range(len(sn)):
        g_mean.append(round(math.sqrt(sn[i] * sp[i]),4))
mcc = [0.6399,-0.2134,-0.01,0.159,-0.0748,0.3212,0.2822,0.6100]
f1 = [0.8125,0.5618,0.6039,0.7589,0.8305,0.8108,0.8213,0.8095]
algorithm = ['AMYGNN','iAMY-SCM','Bioseq_CD','Bioseq_RF','AmyPred-FRL','Bioseq_KNN','Bioseq_SVM','iAMY-DC']
num = np.arange(len(algorithm))
print(g_mean)
# colorlist = [(0.97,0.19,0.14),(0.98,0.55,0.35),(1,0.87,0.57),(0.9,0.94,0.95),(0.56,0.74,0.87),(0.29,0.45,0.69)]
font = {'family' : 'Times New Roman',
'weight' : 'bold',
'size'   : 40,
        }
x_num = np.arange(len(indicator))
plt.figure(figsize = (11,11))
ax =plt.gca()
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['top'].set_linewidth('2.5')
ax.spines['right'].set_linewidth('2.5')
ax.spines['left'].set_linewidth('2.5')
ax.spines['bottom'].set_linewidth('2.5')
corl = ['#80AFBF','#608595','#DFC286','#C07A92','#E2C3C9']
# plt.barh(num,acc,color = '#80AFBF')
# plt.plot(acc,num,'o-',markersize = 20,lw = 10,c = '#DFC286',alpha = 1)
# for x,y in zip(num,acc):
#     plt.text(y - 0.3,x-0.15,y,ha = 'center',va = 'bottom',fontdict = {'family': 'Times New Roman','weight' : 'bold','size' :40},rotation = 0)
# plt.barh(num,g_mean,color = '#608595')
# plt.plot(g_mean,num,'o-',markersize = 20,lw = 10,c = '#DFC286',alpha = 1)
# for x,y in zip(num,g_mean):
#     if y == 0:
#         plt.text(y + 0.1, x - 0.2 , y, ha='center', va='bottom',fontdict={'family': 'Times New Roman', 'weight': 'bold', 'size': 40}, rotation=0)
#     else:
#         plt.text(y - 0.25,x-0.2,y,ha = 'center',va = 'bottom',fontdict = {'family': 'Times New Roman','weight' : 'bold','size' :40},rotation = 0)
# plt.barh(num,sn,color = '#608595')
# for x,y in zip(num,sn):
#     plt.text(y - 0.3,x-0.15,y,ha = 'center',va = 'bottom',fontdict = {'family': 'Times New Roman','weight' : 'bold','size' :40},rotation = 0)
# plt.barh(num, sp, color= '#DFC286')
# for x, y in zip(num, sp):
#     plt.text(y - 0.19, x - 0.15, y, ha='center', va='bottom', fontdict = {'family': 'Times New Roman','weight' : 'bold','size' :40}, rotation=0)
# plt.barh(num, mcc, color= '#C07A92')
# plt.plot(mcc,num,'o-',markersize = 20,lw = 10,c = '#DFC286',alpha = 1)
# for x, y in zip(num, mcc):
#     if y > 0:
#         # plt.text(y - 0.15, x - 0.2, y, ha='center', va='bottom', fontdict=font, rotation=0)
#         if y == 0.832:
#                 plt.text(y - 0.6, x - 0.2, y, ha='center', va='bottom', fontdict = font, rotation=0)
#         else:
#                 plt.text(y - 0.16, x - 0.2, y, ha='center', va='bottom', fontdict = font, rotation=0)
#     else:
#         plt.text(y + 0.2, x - 0.2, y, ha='center', va='bottom', fontdict=font, rotation=0)

plt.barh(num, f1, color= '#E2C3C9')
for x, y in zip(num, f1):
    plt.text(y - 0.29, x - 0.15, y, ha='center', va='bottom', fontdict = font, rotation=0)
plt.plot(f1,num,'o-',markersize = 20,lw = 10,c = '#DFC286',alpha = 1)
# plt.bar(x_num - 0.3,iamy_scm,width = 0.15,color = '#80AFBF',alpha = 1,label = 'iAMY_SCM')
# plt.bar(x_num - 0.15,bioseq_svm,width = 0.15,color = '#608595',alpha = 1,label = 'BioSeq_SVM')
# plt.bar(x_num,amypred,width = 0.15,color = '#DFC286',alpha = 1,label = 'AmyPred-FRL')
# plt.bar(x_num + 0.15,iamydc,width = 0.15,color = '#C07A92',alpha = 1,label = 'iAmy_DC')
# plt.bar(x_num + 0.3,amygcn,width = 0.15,color = '#E2C3C9',alpha = 1,label = 'AMYGNN')
# for x,y in zip(x_num - 0.3,iamy_scm):
#     plt.text(x,y+0.01,y,ha = 'center',va = 'bottom',fontsize = 16,rotation = 90)
# for x,y in zip(x_num - 0.15,bioseq_svm):
#     plt.text(x,y+0.01,y,ha = 'center',va = 'bottom',fontsize = 16,rotation = 90)
# for x,y in zip(x_num,amypred):
#     plt.text(x,y+0.005,y,ha = 'center',va = 'bottom',fontsize = 16,rotation = 90)
# for x,y in zip(x_num + 0.15,iamydc):
#     plt.text(x,y+0.01,y,ha = 'center',va = 'bottom',fontsize = 16,rotation = 90)
# for x,y in zip(x_num + 0.3,amygcn):
#     plt.text(x,y+0.01,y,ha = 'center',va = 'bottom',fontsize = 16,rotation = 90)
# plt.xlabel('Sn',fontdict = font,labelpad = 10)
# plt.ylabel('Algorithm',fontdict = font,labelpad = 10)
plt.title('Comparative result of F1_Score',fontdict = {'family' : 'Times New Roman','weight' : 'bold','size' : 40},y = 1.02)
plt.xticks(fontproperties = 'Times New Roman', size = 40)
plt.yticks(num,labels = algorithm,fontproperties = 'Times New Roman', size = 40)
plt.xlim(0,1.0)
plt.legend(prop = {'family':'Times New Roman','size' : 20},loc = 'upper right',ncol = 2,bbox_to_anchor = (1,1),frameon = False)
plt.tick_params(labelsize = 40)
# plt.savefig(r'E:\paper_datasets\作图\Comparative_Result_F1_new.png',dpi = 1000,bbox_inches = 'tight')
##plt.show()

# # #画本文所提出方法的性能的代码
# fig = plt.figure(figsize = (7,7))
# sum = 255
# colorlist = [(78/sum,171/sum,144/sum),(142/sum,182/sum,156/sum),(237/sum,221/sum,195/sum),(238/sum,191/sum,109/sum),(217/sum,79/sum,51/sum),(131/sum,64/sum,38/sum)]
# fontdict = {'family' : 'Times New Roman',
# 'weight' : 'normal',
# 'size'   : 30,
#         }
# # ax = fig.add_subplots()
# ax =plt.gca()
# ax.spines['top'].set_visible(True)
# ax.spines['right'].set_visible(True)
# ax.spines['bottom'].set_visible(True)
# ax.spines['left'].set_visible(True)
# ax.spines['top'].set_linewidth('1.8')
# ax.spines['right'].set_linewidth('1.8')
# ax.spines['left'].set_linewidth('1.8')
# ax.spines['bottom'].set_linewidth('1.8')
# indicator = ['ACC','G-Mean','Sn','Sp','MCC','F1']
# number = [0.9208,0.9203,0.9358,0.9050,0.8417,0.9235]
# # y_err = np.array([(0.0055,0.0049,0.0081,0.0014,0.0101,0.0037),
# #                   (-0.0137,-0.0147,-0.011,-0.0186,-0.0288,-0.0181)])
# # y_err = np.array([0.0137,0.011,0.0186,0.0147,0.0288,0.0181])
# y_err = [
#     (0.0191,0.0198,0.0107,0.05,0.0386,0.0167),
#     (0.0055,0.0048,0.0481,0.0363,0.0095,0.0089)
# ]
# plt.bar(indicator,number,yerr = y_err,error_kw = {'ecolor' : 'gray', 'capsize' :7 ,'elinewidth':4,'capthick':2},color = colorlist,width = 0.8,alpha = 0.5)
# # plt.plot(indicator,number,color = '#005249',marker = 'o',lw = 3)
# for x,y in zip(indicator,number):
#     if y > 0.9 and y < 0.91:
#         plt.text(x, y + 0.05, y, ha='center', va='bottom', fontdict = {'family' : 'Times New Roman','weight':'bold','size':25},rotation = 90)
#     if y > 0.91 and y <0.925:
#         plt.text(x, y + 0.015, y, ha='center', va='bottom', fontdict = {'family' : 'Times New Roman','weight':'bold','size':25},rotation = 90)
#     elif y >0.93 and y < 0.94:
#         plt.text(x, y + 0.06, y, ha='center', va='bottom', fontdict = {'family' : 'Times New Roman','weight':'bold','size':25},rotation = 90)
#     elif y >0.83 and y < 0.89:
#         plt.text(x, y + 0.02, y, ha='center', va='bottom', fontdict = {'family' : 'Times New Roman','weight':'bold','size':25},rotation = 90)
#
#
# plt.xlabel('Metrics',fontdict = {'family' : 'Times New Roman','size':30},labelpad = 10)
# plt.ylabel('Number',fontdict,labelpad=10)
# plt.title('AMYGNN Result',fontdict = {'family' : 'Times New Roman','weight':'bold','size':25},y = 1.02)
# plt.xticks(fontproperties = 'Times New Roman',fontsize = 20)
# plt.yticks(fontproperties = 'Times New Roman',fontsize = 20)
# plt.ylim(0.3,1.15)
#
# plt.tick_params(labelsize = 22)
# # 三维图
# # fig = plt.figure(figsize = (6,6))
# # ax3d = fig.add_subplot(projection='3d')  #创建3d坐标系
# # np.random.seed(202201)
# # fontdict = {'family' : 'Times New Roman',
# # 'weight' : 'normal',
# # 'size'   : 20,
# #         }
# #
# # y = (1,2,3,4,5,6)
# # x = (1,2,3,4,5,6)
# # z = np.zeros(6) #柱子底部坐标
# # dx=1   #柱子平面宽度
# # dy=1   #柱子平面深度
# # dz = [0.9556,0.91,0.93,0.9199,0.9524,0.9375]    #柱子高度
# # sum = 255
# # ax3d.bar3d(x,y,z,dx,dy,dz,
# #            color =[(78/sum,171/sum,144/sum),(142/sum,182/sum,156/sum),(237/sum,221/sum,195/sum),(238/sum,191/sum,109/sum),(217/sum,79/sum,51/sum),(131/sum,64/sum,38/sum)] )  #绘制3d柱形图
# # # ax3d.set_xlabel('Serial Number',fontdict)
# # # ax3d.set_ylabel('Indicator',fontdict)
# # # ax3d.set_zlabel('Performance',fontdict)
# # ax3d.set_title('AmyGcn Result',fontdict,x = 0.7,y = 1.02)
# # ax3d.tick_params(labelsize = 17)
# # trans_angle = ax3d.transData.transform_angles(np.array([45]),np.array([[5,5]]))[0]
# # for nx,ny,nz in zip(x,y,dz):
# #     if nz == 0.9375 :
# #         ax3d.text(x = nx + 0.7, y = ny + 0.2, z=nz + 0.1, s=nz, fontsize=17, ha='center', va='center_baseline')
# #     else:
# #         ax3d.text(x = nx + 0.7,y = ny - 0.15,z = nz + 0.1,s = nz,fontsize = 17,ha = 'center',va = 'center_baseline')
# #
# # #添加图例
# # y_unique = (0,1,2,3,4,5)   # 可以看作图例类型个数
# # color = [(78/sum,171/sum,144/sum),(142/sum,182/sum,156/sum),(237/sum,221/sum,195/sum),(238/sum,191/sum,109/sum),(217/sum,79/sum,51/sum),(131/sum,64/sum,38/sum)]  # 颜色集
# # label = ('ACC','Sn','Sp','G-Mean','F1-Score','MCC')   # 图例说明集
# # legend_lines = [mpl.lines.Line2D([0], [0], linestyle="none",marker = 'o', c=color[y]) for y in y_unique]
# # legend_labels = [label[y] for y in y_unique]
# # ax3d.legend(legend_lines, legend_labels, numpoints=2,markerscale = 1.5,
# #             loc = 'lower right',bbox_to_anchor=(1.4,0.2),prop = {'family' : 'Times New Roman','size' : 15})
# # # ax3d.view_init(elev = 60, azim= -100)
# # # plt.tick_params(labelsize = 18)
# # plt.yticks(fontproperties = 'Times New Roman', size = 17)
# # plt.xticks(fontproperties = 'Times New Roman', size = 17)
# plt.savefig(r'E:\paper_datasets\作图\new_result.png',dpi = 2000,bbox_inches = 'tight')
# # # # plt.show()

# #三层GCN比较的代码
# one_number = [0.7514,0.7447,0.7753,0.5021,0.7952]
# two_number = [0.8224,0.8222,0.8257,0.6457,0.9049]
# three_number = [0.9208,0.9203,0.9235,0.8417,0.9717]
# indicator = ['ACC','G-Mean','F1-Score','MCC','AUROC']
# x_num = np.arange(len(indicator))
# colorlist1 = ['#DBA11C','#BE7344','#FFEDCB','#005249']
# colorlist2 = ['#FFA752','#FFC1B3','#FFBA7F','#C1554D']
# colorlist3 = ['#FFB283','#B5A999','#00C895','#99B0A8']
# fontdict = {'family' : 'Times New Roman',
# 'weight' : 'bold',
# 'size'   : 30,
#         }
#
# fig = plt.figure(figsize = (9,9))
# # ax = fig.add_subplots()
# x_major_locator=MultipleLocator(0.05)
# ax =plt.gca()
# ax.spines['top'].set_visible(True)
# ax.spines['right'].set_visible(True)
# ax.spines['bottom'].set_visible(True)
# ax.spines['left'].set_visible(True)
# ax.spines['top'].set_linewidth('2')
# ax.spines['right'].set_linewidth('2')
# ax.spines['left'].set_linewidth('2')
# ax.spines['bottom'].set_linewidth('2')
# ax.xaxis.set_major_locator(x_major_locator)
# plt.barh(x_num-0.2,three_number,color = colorlist1[3],height = 0.2,alpha = 0.6,label = 'Three layers')
# plt.barh(x_num,two_number,color = colorlist1[0],height = 0.2,alpha = 0.6,label = 'Two layers')
# plt.barh(x_num + 0.2,one_number,color = colorlist1[1],height = 0.2,alpha = 0.6,label = 'One layer')
# plt.plot([three_number[0],two_number[0],one_number[0]],[x_num[0]-0.2,x_num[0],x_num[0]+0.2],'o-',markersize = 8,lw = 3.5,color = '#C1554D')
# plt.plot([three_number[1],two_number[1],one_number[1]],[x_num[1]-0.2,x_num[1],x_num[1]+0.2],'o-',markersize = 8,lw = 3.5,color = '#C1554D')
# plt.plot([three_number[2],two_number[2],one_number[2]],[x_num[2]-0.2,x_num[2],x_num[2]+0.2],'o-',markersize = 8,lw = 3.5,color = '#C1554D')
# plt.plot([three_number[3],two_number[3],one_number[3]],[x_num[3]-0.2,x_num[3],x_num[3]+0.2],'o-',markersize = 8,lw = 3.5,color = '#C1554D')
# plt.plot([three_number[4],two_number[4],one_number[4]],[x_num[4]-0.2,x_num[4],x_num[4]+0.2],'o-',markersize = 8,lw = 3.5,color = '#C1554D')
# for x,y in zip(x_num-0.2,three_number):
#     plt.text(y+0.044,x - 0.1 ,y,ha = 'center',va = 'bottom',fontsize = 20,rotation = 0)
# for x,y in zip(x_num,two_number):
#     plt.text(y + 0.044,x - 0.1,y,ha = 'center',va = 'bottom',fontsize = 20,rotation = 0)
# for x,y in zip(x_num+0.2,one_number):
#     plt.text(y+0.044,x - 0.1,y,ha = 'center',va = 'bottom',fontsize = 20,rotation = 0)
#
# # plt.bar(x_num-0.2,three_number,color = colorlist1[3],width = 0.2,alpha = 0.6,label = 'Three layers')
# # plt.bar(x_num,two_number,color = colorlist1[0],width = 0.2,alpha = 0.6,label = 'Two layers')
# # plt.bar(x_num + 0.2,one_number,color = colorlist1[2],width = 0.2,alpha = 1,label = 'One layer')
# # plt.plot([x_num[0]-0.2,x_num[0],x_num[0]+0.2],[three_number[0],two_number[0],one_number[0]],'o-',markersize = 8,lw = 3,color = '#C1554D')
# # plt.plot([x_num[1]-0.2,x_num[1],x_num[1]+0.2],[three_number[1],two_number[1],one_number[1]],'o-',markersize = 8,lw = 3,color = '#C1554D')
# # plt.plot([x_num[2]-0.2,x_num[2],x_num[2]+0.2],[three_number[2],two_number[2],one_number[2]],'o-',markersize = 8,lw = 3,color = '#C1554D')
# # plt.plot([x_num[3]-0.2,x_num[3],x_num[3]+0.2],[three_number[3],two_number[3],one_number[3]],'o-',markersize = 8,lw = 3,color = '#C1554D')
# # for x,y in zip(x_num-0.2,three_number):
# #     plt.text(x,y,y,ha = 'center',va = 'bottom',fontsize = 18,rotation = 0)
# # for x,y in zip(x_num,two_number):
# #     plt.text(x,y,y,ha = 'center',va = 'bottom',fontsize = 18,rotation = 0)
# # for x,y in zip(x_num+0.2,one_number):
# #     plt.text(x,y,y,ha = 'center',va = 'bottom',fontsize = 18,rotation = 0)
# # plt.xlabel('Metrics',fontdict)
# # plt.ylabel('Number',fontdict)
# plt.xlabel('Number of each metric',fontdict={'family' :'Times New Roman','weight' : 'normal','size' : 30},labelpad=10)
# plt.legend(loc = 'lower right',prop = {'family' :'Times New Roman','weight' : 'bold','size' : 20},bbox_to_anchor= (1.02,0.92),ncol = 3,frameon =  False,handletextpad = 0.1)
# plt.xlim(0.4,1.18)
# plt.title('Impact of each network layer on results',fontdict,y = 1.02)
# # plt.xticks(x_num,fontproperties = 'Times New Roman',fontsize = 18,labels = indicator)
# plt.xticks(fontproperties = 'Times New Roman',fontsize = 25,rotation = 45)
# plt.yticks(x_num,fontproperties = 'Times New Roman',fontsize = 25,labels = indicator)
# # plt.yticks(fontproperties = 'Times New Roman',fontsize = 18)
# plt.tick_params(labelsize = 25)
# plt.savefig(r'E:\paper_datasets\作图\Composition_GCN.png',dpi = 2000,bbox_inches = 'tight')
# # #
# # # plt.figure(figsize = (7,7))
# # # ax =plt.gca()
# # # ax.spines['top'].set_visible(True)
# # # ax.spines['right'].set_visible(True)
# # # ax.spines['bottom'].set_visible(True)
# # # ax.spines['left'].set_visible(True)
# # # ax.spines['top'].set_linewidth('1.8')
# # # ax.spines['right'].set_linewidth('1.8')
# # # ax.spines['left'].set_linewidth('1.8')
# # # ax.spines['bottom'].set_linewidth('1.8')
# # # plt.bar(indicator,two_number,color = colorlist,width = 0.6,alpha = 0.9)
# # # for x,y in zip(indicator,two_number):
# # #     plt.text(x,y,y,ha = 'center',va = 'bottom',fontsize = 18)
# # # plt.xlabel('Metrics',fontdict)
# # # plt.ylabel('Number',fontdict)
# # # plt.title('TwoLayers_Gcn Result',fontdict,y = 1.02)
# # # plt.xticks(fontproperties = 'Times New Roman',fontsize = 18)
# # # plt.yticks(fontproperties = 'Times New Roman',fontsize = 18)
# # # plt.tick_params(labelsize = 18)
# # # plt.savefig(r'E:\paper_datasets\作图\Twolayers_Result.png',dpi = 2000,bbox_inches = 'tight')
# #
# # # plt.figure(figsize = (7,7))
# # # ax =plt.gca()
# # # ax.spines['top'].set_visible(True)
# # # ax.spines['right'].set_visible(True)
# # # ax.spines['bottom'].set_visible(True)
# # # ax.spines['left'].set_visible(True)
# # # ax.spines['top'].set_linewidth('1.8')
# # # ax.spines['right'].set_linewidth('1.8')
# # # ax.spines['left'].set_linewidth('1.8')
# # # ax.spines['bottom'].set_linewidth('1.8')
# # # plt.bar(indicator,one_number,color = colorlist,width = 0.6,alpha = 0.9)
# # # for x,y in zip(indicator,one_number):
# # #     plt.text(x,y,y,ha = 'center',va = 'bottom',fontsize = 18)
# # # plt.xlabel('Metrics',fontdict)
# # # plt.ylabel('Number',fontdict)
# # # plt.title('OneLayer_Gcn Result',fontdict,y = 1.02)
# # # plt.xticks(fontproperties = 'Times New Roman',fontsize = 18)
# # # plt.yticks(fontproperties = 'Times New Roman',fontsize = 18)
# # # plt.tick_params(labelsize = 18)
# # # plt.savefig(r'E:\paper_datasets\作图\Onelayer_Result.png',dpi = 2000,bbox_inches = 'tight')
plt.show()

