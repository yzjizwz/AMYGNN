#用于处理AAIndex1文本文件
import re
from pandas.core.frame import DataFrame
import pandas as pd
pd.set_option('display.width',200)
pd.set_option('display.max_colwidth',100)


f = open(r"E:\paper_datasets\AAindex\aaindex1.txt")
data = f.readlines()
f.close()
print(data[0])
print(data[1])
print(len(data))
str1 = data[9]
str1 = str1.split()
re.match(r'\s*[1-9]\d*\.\d*|0\.\d*[1-9]\d*\s*[1-9]\d*\.\d*|0\.\d*[1-9]\d*',data[9]).group()
ID_list = []
describe_list = []
A_list = []
R_list= []
N_list = []
D_list = []
C_list =[]
Q_list = []
E_list = []
G_list = []
H_list = []
I_list = []
L_list = []
K_list = []
M_list = []
F_list = []
P_list = []
S_list = []
T_list = []
W_list = []
Y_list = []
V_list = []
for entry in data:
    ID_partten = re.match('H{1}\s{1}\w+\d+',entry)
    describe_partten = re.match(r"D+\s{1}[\w*-?\s*;\*\d*%*]*",entry)
    if ID_partten:
        ID_list.append(ID_partten.group())
    else:
        pass
    if describe_partten:
        describe_list.append(describe_partten.group())
    else:
        pass
aa_name1_list = ['A','R','N','D','C','Q','E','G','H','I']
aa_name2_list = ['L','K','M','F','P','S','T','W','Y','V']

# for char in aa_name1_list:
#     aa_str1 = char + '_list'
#     aa_list1 = list(aa_str1.split())

# for _ in aa_name2_list:
#     aa_str2 = _ + '_list'
#     aa_list2 = aa_str2.split()

for i in range(0,len(data)):
    index_partten = re.match(r"I{1}\s*[\w*\/\w*\s*]*",data[i])
    if index_partten:
        aa_index1_list = data[i + 1].split()
        aa_index2_list = data[i + 2].split()
        A_list.append(aa_index1_list[0])
        R_list.append(aa_index1_list[1])
        N_list.append(aa_index1_list[2])
        D_list.append(aa_index1_list[3])
        C_list.append(aa_index1_list[4])
        Q_list.append(aa_index1_list[5])
        E_list.append(aa_index1_list[6])
        G_list.append(aa_index1_list[7])
        H_list.append(aa_index1_list[8])
        I_list.append(aa_index1_list[9])
        L_list.append(aa_index2_list[0])
        K_list.append(aa_index2_list[1])
        M_list.append(aa_index2_list[2])
        F_list.append(aa_index2_list[3])
        P_list.append(aa_index2_list[4])
        S_list.append(aa_index2_list[5])
        T_list.append(aa_index2_list[6])
        W_list.append(aa_index2_list[7])
        Y_list.append(aa_index2_list[8])
        V_list.append(aa_index2_list[9])
    else:
        pass
df_dict = {'ID_name':ID_list,'Describe':describe_list,
           'A':A_list,'R':R_list,'N':N_list,'D':D_list,'C':C_list,'Q':Q_list,'E':E_list,'G':G_list,'H':H_list,'I':I_list,
           'L':L_list,'K':K_list,'M':M_list,'F':F_list,'P':P_list,'S':S_list,'T':T_list,'W':W_list,'Y':Y_list,'V':V_list}
df = DataFrame(df_dict)
df.head()
df.to_excel(r'E:\paper_datasets\AAindex\aaindex1.xlsx',
           sheet_name = 'aaindex1 after processing')