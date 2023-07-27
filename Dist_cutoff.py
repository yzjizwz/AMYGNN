import matplotlib.pyplot as plt
import matplotlib as mpl
import  scienceplots
import numpy as np
from scipy.integrate import simpson
from matplotlib.lines import Line2D
from brokenaxes import brokenaxes
mpl.rcParams["font.sans-serif"]=["Times New Roman"]
mpl.rcParams["axes.unicode_minus"]=False
plt.style.use("nature")

x=range(1,6)
acc_5 = [0.7858,0.7871,0.7898,0.8042,0.8078]
acc_6 = [0.7493,0.7558,0.8033,0.8036,0.8042]
acc_7 = [0.7392,0.7678,0.7824,0.7922,0.8067]
acc_8 = [0.7658,0.7724,0.7764,0.7851,0.7978]
labels = ['5','6','7','8']
sum = 255
# pal = [(16/sum,70/sum,128/sum),(109/sum,173/sum,209/sum),(246/sum,178/sum,147/sum),(183/sum,34/sum,48/sum)]
area_5 = np.trapz(acc_5,x)
area_6 = np.trapz(acc_6,x)
area_7 = np.trapz(acc_7,x)
area_8 = np.trapz(acc_8,x)
font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 35,
        }
plt.figure(figsize = (10,8))
f,(ax1,ax) = plt.subplots(2,1,sharex = False)
# ax3 = plt.gca()
# ax3.spines['top'].set_visible(True)
# ax3.spines['right'].set_visible(True)
# ax3.spines['bottom'].set_visible(True)
# ax3.spines['left'].set_visible(True)
# ax3.spines['top'].set_linewidth('3')
# ax3.spines['right'].set_linewidth('3')
# ax3.spines['left'].set_linewidth('3')
# ax3.spines['bottom'].set_linewidth('3')
ax1.spines['top'].set_visible(True)
ax1.spines['right'].set_visible(True)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(True)
ax1.spines['top'].set_linewidth('1')
ax1.spines['right'].set_linewidth('1')
ax1.spines['left'].set_linewidth('1')
ax1.spines['bottom'].set_linewidth('1')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['top'].set_linewidth('1')
ax.spines['right'].set_linewidth('1')
ax.spines['left'].set_linewidth('1')
ax.spines['bottom'].set_linewidth('1')
#
# # ax3.set_ylim(0.81,1)
# # ax3.xaxis.set_major_locator(plt.NullLocator())
#
# ax.fill_between(x,acc_6,color= (109/sum,173/sum,209/sum),alpha = 0.6)
# ax.fill_between(x,acc_7,color= (246/sum,178/sum,147/sum), alpha=0.6)
ax.fill_between(x,acc_5,color = (16/sum,70/sum,128/sum),alpha = 0.9)
ax.fill_between(x,acc_8,color = (183/sum,34/sum,48/sum),alpha = 0.6)

ax1.xaxis.set_major_locator(plt.NullLocator())

ax1.fill_between(x,acc_5,color = (16/sum,70/sum,128/sum),alpha = 0.9)
ax1.fill_between(x,acc_6,color= (109/sum,173/sum,209/sum),alpha = 0.6)
ax1.fill_between(x,acc_7,color= (246/sum,178/sum,147/sum), alpha=0.6)
ax1.fill_between(x,acc_8,color = (183/sum,34/sum,48/sum),alpha = 0.6)


d = 0.03  # 断层线的大小
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False,lw = 2)
ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal

kwargs.update(transform=ax.transAxes, color='k',lw = 2)  # switch to the bottom axes
ax.plot((-d, +d), (1 - d, 1 + d),**kwargs)  # bottom-left diagonal
ax1.set_ylim(0.735,0.87)
ax.set_ylim(0,0.735)
ax.tick_params('x',labelsize = 10)
ax.tick_params('y',labelsize = 10)
ax1.tick_params('x',labelsize = 10)
ax1.tick_params('y',labelsize = 10)
plt.legend(loc='lower right',bbox_to_anchor = (0.95,0),prop = {'family':'Times New Roman','weight':'bold','size' : 8},facecolor = (243/sum,243/sum,243/sum),
           handles = [Line2D([0],[0],color = (16/sum,70/sum,128/sum),lw = 7,label = '5Å(area:3.1779)'),
                      Line2D([0],[0],color = (109/sum,173/sum,209/sum),lw = 7,label = '6Å(area:3.1395)'),
                      Line2D([0],[0],color = (246/sum,178/sum,147/sum),lw = 7,label = '7Å(area:3.1154)'),
                      Line2D([0],[0],color = (183/sum,34/sum,48/sum),lw = 7,label = '8Å(area:3.1157)')])
plt.savefig(r'E:\paper_datasets\作图\dist_cutoff_new.png',dpi = 1000,bbox_inches = 'tight')

#
# plt.fill_between(x, acc_5, color=(16/sum,70/sum,128/sum), alpha=0.9)
# plt.fill_between(x, acc_6, color= (109/sum,173/sum,209/sum), alpha=0.6)
# plt.fill_between(x, acc_7, color= (246/sum,178/sum,147/sum), alpha=0.6)
# plt.fill_between(x, acc_8, color=(183/sum,34/sum,48/sum), alpha=0.6)
# plt.plot( [], [], color=(16/sum,70/sum,128/sum),label='5Å(area:3.1779)')
# # # plt.text(1.9,0.786,'The area is : %0.4f'%area_5,fontdict = font)
# # # plt.text(1.85,0.748,'The area under the 5Å curve is : %0.4f'%area_5,fontdict = font)
# plt.plot( [], [], color=(109/sum,173/sum,209/sum),label='6Å(area:3.1395)')
# # # plt.text(1.9,0.746,'The area is : %0.4f'%area_6,fontdict = font)
# # # plt.text(1.85,0.744,'The area under the 6Å curve is : %0.4f'%area_6,fontdict = font)
# plt.plot( [], [], color=(246/sum,178/sum,147/sum),label='7Å(area:3.1154' )
# # # plt.text(1.9,0.736,'The area is : %0.4f'%area_7,fontdict = font)
# # # # plt.text(1.85,0.74,'The area under the 7Å curve is : %0.4f'%area_7,fontdict = font)
# plt.plot( [], [], color=(183/sum,34/sum,48/sum),label='8Å(area:3.1157)')
# # plt.text(1.9,0.766,'The area is : %0.4f'%area_8,fontdict = font)
# # # plt.text(1.85,0.736,'The area under the 8Å curve is : %0.4f'%area_8,fontdict = font)
# plt.legend(loc='lower right',bbox_to_anchor = (0.95,0),prop = {'family':'Times New Roman','weight':'bold','size' : 30},facecolor = (243/sum,243/sum,243/sum),handles = [Line2D([0],[0],color = (16/sum,70/sum,128/sum),lw = 7,label = '5Å(area:3.1779)'),
#                                                                                        Line2D([0],[0],color = (109/sum,173/sum,209/sum),lw = 7,label = '6Å(area:3.1395)'),
#                                                                                        Line2D([0],[0],color = (246/sum,178/sum,147/sum),lw = 7,label = '7Å(area:3.1154)'),
#                                                                                        Line2D([0],[0],color = (183/sum,34/sum,48/sum),lw = 7,label = '8Å(area:3.1157)')])
# # plt.legend(loc='lower right',bbox_to_anchor = (0.97,0),prop = {'family':'Times New Roman','size' : 40,'weight':'bold'},facecolor = (243/sum,243/sum,243/sum),handles = [Line2D([0],[0],color = (16/sum,70/sum,128/sum),lw = 20,label = '5Å(area:3.1779)')])
# # plt.legend(loc='lower right',bbox_to_anchor = (0.95,0),prop = {'family':'Times New Roman','size' : 40,'weight':'bold'},facecolor = (243/sum,243/sum,243/sum),handles = [Line2D([0],[0],color = (109/sum,173/sum,209/sum),lw = 20,label = '6Å(area:3.1395)')])
# # plt.legend(loc='lower right',bbox_to_anchor = (0.95,0),prop = {'family':'Times New Roman','weight':'bold','size' : 40},facecolor = (243/sum,243/sum,243/sum),handles = [Line2D([0],[0],color = (246/sum,178/sum,147/sum),lw = 20,label = '7Å(area:3.1154)')])
# # plt.legend(loc='lower right',bbox_to_anchor = (0.95,0),prop = {'family':'Times New Roman','weight' : 'bold','size' : 45},facecolor = (243/sum,243/sum,243/sum),handles = [Line2D([0],[0],color = (183/sum,34/sum,48/sum),lw = 20,label = '8Å(area:3.1157)')])
# plt.tick_params(labelsize = 40)
# plt.yticks(fontproperties= {'family':'Times New Roman','size' : 35,'weight' : 'bold'})
# plt.xticks(fontproperties= {'family':'Times New Roman','size' : 35,'weight' : 'bold'})
#
# plt.ylim(0,1)
# # plt.ylim(0.765,0.81)
# #
# # # plt.title("The relationship between the distance threshold and accuracy", loc="center",fontdict = font,y = 1.02)
# # # plt.xlabel("Epoch",fontdict = font)
# # # plt.ylabel("Accuracy",fontdict = font)
# # plt.savefig(r'E:\paper_datasets\作图\dist_cutoff_8.png',dpi = 1000,bbox_inches = 'tight')
plt.show()