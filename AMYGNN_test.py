import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.loader import DataLoader
import networkx as nx
import matplotlib.pyplot as plt
from Amygnn_model import AMYGNN,test_data_loader
import os

# from test import test_loader

from sklearn.metrics import *
import csv


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = torch.load(r'Onelayer_model.pt').to(device)
# model = torch.load(r'Twolayers_model.pt').to(device)
model = torch.load(r'Threelayers_model.pt').to(device)
y_score = []
fprl,tprl = [],[]

def test(test_loader):
    model.eval()
    loss = 0
    for test_data in test_loader:
        output = model(test_data.x.to(device),test_data.edge_index.to(device),test_data.edge_attr.to(device),test_data.batch.to(device))
        classifier = output.argmax(dim = 1)
        loss_test = F.nll_loss(output,test_data.y.to(device))
        acc_test = accuracy_score(classifier.to('cpu'),test_data.y)
        cm = confusion_matrix(test_data.y, classifier.to('cpu'))
        for i in range(len(classifier)):
            prob = output[i][1].cpu().detach().numpy()
            y_score.append(prob)
        f1 = f1_score(test_data.y,classifier.to('cpu'))
        fpr, tpr, thresholds = roc_curve(test_data.y, y_score, pos_label=1)
        fprl.append([f for f in fpr])
        tprl.append([f for f in tpr])
        auroc = auc(fpr,tpr)
        y_score.clear()
        mcc = matthews_corrcoef(test_data.y,classifier.to('cpu'))
        loss += loss_test
    losstest = loss / len(test_loader)
    print("Test set results:",
              "loss= {:.4f}".format(losstest),
              "accuracy= {:.4f}".format(acc_test),
              "f1-score= {:.4f}".format(f1),
              "auroc= {:.4f}".format(auroc),
              "mcc= {:.4f}".format(mcc),
              cm,
              )
    return fprl,tprl,auroc,acc_test

fpr,tpr,auroc,acc = test(test_data_loader)


# fpr_path = r'E:\paper_datasets\ROC\AMYGNN_Three_layer_Fpr_num_' + str(acc) + '.csv'
# tpr_path = r'E:\paper_datasets\ROC\AMYGNN_Three_layer_Tpr_num_' + str(acc) + '.csv'
# with open(fpr_path,'w',newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     for row in fpr:
#         writer.writerow(row)
#     csvfile.close()
# with open(tpr_path,'w',newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     for row in tpr:
#         writer.writerow(row)
#     csvfile.close()
