# import networkx as nx
# import matplotlib.pyplot as plt
# import re
# import numpy as np
# import pandas as pd
#
# all_features = pd.read_excel(r'E:\paper_datasets\AAindex\aaindex1.xlsx',header = [0],usecols = [i for i in range(1,23)])
# print(all_features.head(10))
# feature_name = all_features['ID_name']
# feature_dict_list = []
# for i in range(len(feature_name)):
# #     feature_id = all_features.iloc[i][1][2:]  #这一行提取到了寡肽在aaindex数据库中的名字
#     aa_number_dict = dict(all_features.iloc[i][2:])
#     for key,value in aa_number_dict.items():
#         feature_dict_list.append((key[0] + '_' + str(i),all_features.iloc[i][1][2:] + '_' + str(value)))
#         #加下划线和i是为了区分不同性质的氨基酸，python字典会对键自动去重，不这样处理会造成数据量的丢失
# feature_dict = dict(feature_dict_list) #得到的feature_dict可以用于作为节点的特征，但是需要降维处理
# # for key,value in feature_dict.items():
# #     print(key,type(key),key[0])
# # print(feature_dict['A_9'].split('_')[0])
# # print(type(feature_dict['A_9'].split('_')[0]))
#
# # pattern = re.compile('(-?[1-9]\d*\.\d+$|^-?0\.\d+$|^-?[1-9]\d*$|^0$)')
# # flag = re.search(pattern,feature_dict['R_9'])
# # print(flag.group())
# key_list = []
# for key in feature_dict.keys():
#     key_list.append(key)
# print(len(key_list))
# print(key_list[9])
#
# nodes_list = []
# edges_list = []
#
# with open('D:/python代码练习/peptide.fasta', 'r') as file:
#     lines_list = file.readlines()
#     for i in range(0, len(lines_list)):
#         lines_list[i] = lines_list[i].split()[0]
#         Flag = re.match("^>+\w", lines_list[i])
#         if Flag:
#             continue
#         else:
#             # 为图添加边，添加边的时候会涉及到节点，所以不需要再另外添加节点
#             for k in range(0, len(lines_list[i]) - 1):
#                 peptide_graph_i = nx.Graph()
#                 edges_list.append((lines_list[i][k] + '_' + str(i) + '_' + str(k),
#                                    lines_list[i][k + 1] + '_' + str(i) + '_' + str(k + 1)))
#                 peptide_graph_i.add_edges_from(edges_list)
#
#             # 为节点添加属性，后期可以用字典，分别利用key和value作为赋值的来源
#
#             for j in range(0, len(lines_list[i])):
#                 for m in range(0,len(all_features['ID_name'])):
#                     index_str = lines_list[i][j] + '_' + str(m)
#                     feature = feature_dict[index_str].split('_')[0]
#                     feature_number = feature_dict[index_str].split('_')[1]
#                     peptide_graph_i.nodes[lines_list[i][j] + '_' + str(i) + '_' + str(j)][feature] = feature_number
#
#             nx.draw(peptide_graph_i)
#             plt.show()
#             # 为每张图计算邻接矩阵
#             A = nx.adjacency_matrix(peptide_graph_i)
#             #             print(A.todense())
#             print(peptide_graph_i.nodes())
#             #             print(peptide_graph_i.edges)
#             #             print(peptide_graph_i.nodes.data())
#             peptide_graph_i.clear()
#             nodes_list.clear()
#             edges_list.clear()
import pandas as pd
import numpy as np
import os,sys,re
import networkx as nx
import matplotlib.pyplot as plt


feature_data = pd.read_csv(r'E:\paper_datasets\AAC.csv',header = [0])
feature_data.rename(columns = {'Unnamed: 0' : 'peptide_name','AAC_A' : 'A', 'AAC_C' : 'C', 'AAC_D' : 'D', 'AAC_E' : 'E', 'AAC_F' : 'F', 'AAC_G':'G',
                    'AAC_H' : 'H', 'AAC_I' : 'I', 'AAC_K' : 'K', 'AAC_L' : 'L', 'AAC_M' : 'M', 'AAC_N' : 'N', 'AAC_P' : 'P', 'AAC_Q' : 'Q',
                    'AAC_R' : 'R', 'AAC_S' : 'S', 'AAC_T' : 'T', 'AAC_V' : 'V', 'AAC_W' : 'W', 'AAC_Y' : 'Y'},inplace = True)
def read_peptide_sequence(file):
    if os.path.exists(file) == None:
        print('Error : file %s does not exist.'%file)
        sys.exit(1)
    with open(file) as f:
        records = f.read()
        if re.match('>',records) == None:
            print('Error : the input file %s seems not in FASTA format!'%file)
            sys.exit(1)
        records = records.split('>')[1:]
        fasta_sequences = []
        for fasta in records:
            array = fasta.split('\n')
            header,sequence = array[0].split()[0],re.sub('[^ACDEFGHIKLMNPQRSTVWY-]','-',''.join(array[1:]).upper())
            header_array = header.split('|')
            name = header_array[0]
            fasta_sequences.append([name,sequence])
        return fasta_sequences

sequences = read_peptide_sequence(r'D:\python代码练习\peptide.fasta')

nodes_list = []
edges_list = []
for i in range(0,len(sequences)):
    for j in range(0,len(sequences[i][1]) - 1):
        edges_list.append((sequences[i][0] + '_' + sequences[i][1][j] + str(j),sequences[i][0] + '_' + sequences[i][1][j+1] + str(j+1)))
    # print(edges_list,len(edges_list))
    peptide_graph = nx.Graph()
    peptide_graph.add_edges_from(edges_list)
    edges_list.clear()
    # print('---------------------------------------------------------------------------------')
    nx.draw(peptide_graph)
    print(peptide_graph.nodes(),len(peptide_graph))
    peptide_graph.clear()
    plt.show()