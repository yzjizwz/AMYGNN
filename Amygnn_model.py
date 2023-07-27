import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv,GCNConv
from torch_geometric.nn import global_sort_pool
from torch_geometric.loader import DataLoader
from torch.nn import Linear
from torch_geometric.data import Dataset
from torch_geometric.nn.pool import SAGPooling
from torch.utils.data import  random_split
from generate_peptide_graph import peptide_to_graph
from sklearn import svm
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, auc,roc_auc_score
from visdom import  Visdom
# from test import test_loader
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

read_peptide_file = r"E:\paper_datasets\Amyloid_Database\CPAD2.0_Data\aggregating_peptides.xlsx"
# afterpre_peptide_file = r"E:\paper_datasets\Amyloid_Database\CPAD2.0_Data\aggregating_peptides_dropnan.xlsx"
afterpre_peptide_file = r'E:\paper_datasets\after_SMOTE.csv'
pre_aaindex_file = r"E:\paper_datasets\AAindex\aaindex1.txt"
afterpre_aaindex_file = r'E:\paper_datasets\Amyloid_Database\AAIndex_data.xlsx'
peptide_pdb_file = r"E:\paper_datasets\Amyloid_Database\PDB_Data"
AADist_file = r"E:\paper_datasets\Amyloid_Database\Feature\new_aggregating_peptide_AADIST_feature.txt"

dataset = peptide_to_graph(pre_aaindex_file,afterpre_aaindex_file,afterpre_peptide_file,AADist_file)
# print(len(dataset))
train_size = int(0.7 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
# print(train_size,val_size,test_size)
train_data,val_data,test_data = random_split(dataset,[train_size,val_size,test_size])
train_loader = DataLoader(train_data,batch_size = 128,shuffle=True)
val_loader = DataLoader(val_data,batch_size = 64,shuffle = True)
test_data_loader = DataLoader(test_data,batch_size = 366,shuffle=True)
# print(train_data[0],val_data[0],test_data[0])
# print(len(train_loader))
# print(len(train_data))

#定义AMYGNN模型
class AMYGNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(AMYGNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(dataset[0].num_node_features, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(64,eps = 1e-06,momentum = 0.01)
        self.sag1 = SAGPooling(64, ratio = 1e-05)

        self.conv2 = GraphConv(hidden_channels,64)
        self.bn2 = torch.nn.BatchNorm1d(64,eps = 1e-06,momentum = 0.01)
        self.sag2 = SAGPooling(64, ratio = 1e-05)

        self.conv3 = GraphConv(64,32)
        self.bn3 = torch.nn.BatchNorm1d(32,eps = 1e-06,momentum = 0.01)
        self.sag3 = SAGPooling(32, ratio = 1e-05)


        self.lin = Linear(64,32)
        self.lin1 = Linear(32,16)
        self.lin2 = Linear(16,2)

    def forward(self, x, edge_index,edge_weight, batch):
        x = self.conv1(x, edge_index,edge_weight)
        x = x.relu()
        x = self.bn1(x)
        # y = self.sag1(x,edge_index,edge_weight,batch = batch)
        # x = y[0]
        # edge_index = y[1]
        # edge_weight = y[2]
        # batch = y[3]


        x = self.conv2(x, edge_index,edge_weight)
        x = x.relu()
        x = self.bn2(x)
        # y = self.sag2(x,edge_index,edge_weight,batch = batch)
        # x = y[0]
        # edge_index = y[1]
        # edge_weight = y[2]
        # batch = y[3]


        x = self.conv3(x, edge_index,edge_weight)
        x = x.relu()
        x = self.bn3(x)
        y = self.sag3(x,edge_index,edge_weight,batch = batch)
        x = y[0]

        edge_index = y[1]
        edge_weight = y[2]
        batch = y[3]


        x = global_sort_pool(x,batch,2)
        x = F.dropout(x, p = 0.4, training=self.training)
        x = self.lin(x)
        x = x.relu()
        x = self.lin1(x)
        x = self.lin2(x)
        return F.log_softmax(x,dim= 1)

#定义模型和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AMYGNN(hidden_channels = 64).to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(),lr = 0.00001)
#
# viz = Visdom()
# viz.line([0.],[0.], win="train loss", opts=dict(title='train_loss'))
lossl,loss_l = [],[]
y_score,fprl,tprl = [],[],[]
#训练网络
def train():
    model.train()
    for epoch in range(300):
        loss_train,train_correct = 0,0
        optimizer.zero_grad()
        for data in train_loader:
            out = model(data.x.to(device),data.edge_index.to(device),data.edge_attr.to(device),data.batch.to(device))
            probs = out.argmax(dim = 1)
            loss = F.nll_loss(out,data.y.to(device))
            # loss = torch.nn.CrossEntropyLoss()(out.to('cpu'), data.y)
            train_correct += int((probs == data.y.to(device)).sum())
            # train_acc = accuracy_score(probs.to('cpu'),data.y)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        loss_train = loss_train / len(train_loader)
        train_acc = train_correct / len(train_data)
        lossl.append(loss_train)
        # viz.line([loss_train],[epoch],win="train loss",update = 'append')


        # 在验证集上进行评估
        model.eval()
        correct,loss_all = 0,0
        with torch.no_grad():
            # optimizer.zero_grad()
            for val_data in val_loader:
                pred = model(val_data.x.to(device),val_data.edge_index.to(device),val_data.edge_attr.to(device),val_data.batch.to(device))
                probs = pred.argmax(dim = 1)
                val_loss = F.nll_loss(pred, val_data.y.to(device))
                # val_loss = torch.nn.CrossEntropyLoss()(pred.to('cpu'),val_data.y)
                correct += int((probs == val_data.y.to(device)).sum())
                val_acc = accuracy_score(val_data.y,probs.cpu().detach().numpy())
                # val_loss.backward()
                # optimizer.step()
                loss_all += val_loss.item()
            loss_all = loss_all / len(val_loader)
            loss_l.append(loss_all)
        # print(loss_l)
        # val_acc = correct / len

        # model.eval()
        # correct = 0
        # for data in test_loader:
        #     out = model(data.x.to(device),data.edge_index.to(device),data.edge_attr.to(device),data.batch.to(device))
        #     probs = out.argmax(dim = 1)
        #     # print(data.y,probs)
        #     correct += int((probs.to('cpu') == data.y).sum())
        #     acc = accuracy_score(probs.to('cpu'),data.y)
        #     f1 = f1_score(data.y,probs.to('cpu'))
        #     mcc = matthews_corrcoef(data.y,probs.to('cpu'))

        #在单独的数据集上进行验证
        # model.eval()
        # correct = 0
        # for data in test_data_loader:
        #     out = model(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device), data.batch.to(device))
        #     probs = out.argmax(dim=1)
        #     # print(data.y, probs)
        #     correct += int((probs.to('cpu') == data.y).sum())
        #
        #     acc = accuracy_score(data.y,probs.to('cpu'))
        #     f1 = f1_score(data.y, probs.to('cpu'))
        #     mcc = matthews_corrcoef(data.y, probs.to('cpu'))
            # print("Test study results:",
            #       "accuracy= {:.4f}".format(acc),
            #       "f1-score= {:.4f}".format(f1),
            #       "mcc= {:.4f}".format(mcc),
            #       )
        print('Epoch: {:03d}, Train Loss: {:.4f},Val Loss: {:.4f},Train Accuracy : {:.4f},Val Accuracy: {:.4f}'.format(epoch, loss_train, loss_all, train_acc, val_acc))
        # if epoch % 5 == 0:
        # torch.save(model,'D:\python代码练习\\new_model\\'  + str(epoch) + '_' + str(round(train_acc,4)) + '_' + str(round(val_acc,4)) + '.pt')
        # if acc >= 0.83 and mcc >= 0.66 and val_acc > 0.9:
        #     torch.save(model, 'D:\python代码练习\Model\\paper' + '_' + str(epoch) + '_' + str(round(train_acc,5)) + '_' + str(round(val_acc,5)) + '_' + str(round(acc,5)) + '_' + str(round(mcc,5)) + '.pt')
        # if acc_test >= 0.9 and mcc_test >= 0.9:
        #     torch.save(model, 'D:\python代码练习\Model\\paper' + '_' + str(epoch) + '_' + str(round(acc_test,5)) + '_' + str(round(mcc_test,5)) + '.pt')
        # print('Epoch: {:03d}, Train Loss: {:.4f},Val Loss: {:.4f},Train Accuracy : {:.4f},Val Accuracy: {:.4f},Test Acuuracy:{:.4f},Test MCC:{:.4f},Test F1score:{:.4f}'.format(epoch,loss_train,loss_all,train_acc, val_acc,acc,mcc,f1))
    plt.plot(np.arange(300), lossl)
    plt.plot(np.arange(300),loss_l,'r')
    plt.ylim(0,1)
    plt.show()


# train()
torch.cuda.empty_cache()
