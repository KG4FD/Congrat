from random import sample
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import pickle

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, GATv2Conv, Linear
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, recall_score, roc_curve
from torch.nn import ReLU, Sigmoid, Softmax
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import numpy as np

from torch_geometric.nn import MessagePassing, GINConv, GATConv
from torch_geometric.utils import add_self_loops, degree, softmax, to_dense_adj, dense_to_sparse
from torch_scatter import scatter_add
import math
import numpy as np

class ContrasLoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrasLoss, self).__init__()
        self.temp = temperature


    def forward(self, aug_one, aug_two):
        """ aug_one: is a augmentation with knowledge graph1
            aug_two: is another augmentation with knowledge graph2.
        """
        # dimension is [2N, dim]
        out = torch.cat([aug_one, aug_two], 0)
        size = out.shape[0]
        # inner product of each news piece, we can get a sim matrix [2N, 2N]
        news_sim = torch.exp(torch.mm(out, out.t().contiguous()) / self.temp)
        # remove the similarity with self
        mask = (torch.ones_like(news_sim) - torch.eye(size, device=news_sim.device)).bool()
        # dim = [2N, 2N - 1]
        news_sim = news_sim.masked_select(mask).view(size, -1)

        # Here, we can compute the loss
        pos_smaple = torch.exp(torch.sum(aug_one * aug_two, dim=-1) / self.temp)
        pos_sim = torch.cat([pos_smaple, pos_smaple])

        contras_loss = (- torch.log(pos_sim / news_sim.sum(dim=-1))).mean()

        return contras_loss
    
class SupContrasLoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(SupContrasLoss, self).__init__()
        self.temperature = temperature

    def forward(self, aug1, aug2, label):

        # device = torch.device("cuda") if aug1.is_cuda else torch.device("cpu")

        label = label.repeat(2)
        out = torch.cat([aug1,aug2], 0)
        dot_product_tempered = torch.mm(out, out.T) / self.temperature

        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )
        mask_similar_class = (label.unsqueeze(1).repeat(1, label.shape[0]) == label).cuda()
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).cuda()
        mask_combined = mask_similar_class * mask_anchor_out

        cardinality = torch.sum(mask_combined, dim=1)
        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))

        supervised_contrastive_loss = torch.sum(log_prob * mask_combined, dim=1) / cardinality

        loss = torch.mean(supervised_contrastive_loss)

        return loss


# class ContrasLoss1(torch.nn.Module):
#     def __init__(self, temperature=0.5):
#         super(ContrasLoss1, self).__init__()
#         self.temperature = temperature

#     def forward(self, aug1, aug2, label, mask=None):
#         n = label.shape[0]  # batch
#         out = torch.cat([aug1,aug2], 1)
#         # label = label.repeat(2)

#         sim_matrix = F.cosine_similarity(out.unsqueeze(1), out.unsqueeze(0), dim=2)
#         sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
#         # print(sim_matrix)

#         # 这步得到它的label矩阵，相同label的位置为1
#         mask = torch.ones_like(sim_matrix) * (label.expand(n, n).eq(label.expand(n, n).t()))
#         # print(mask)

#         # 这步得到它的不同类的矩阵，不同类的位置为1
#         mask_no_sim = torch.ones_like(mask) - mask
#         mask_no_sim = mask_no_sim.cuda()

#         # print(mask_no_sim)

#         # 这步产生一个对角线全为0的，其他位置为1的矩阵
#         mask_dj_0 = torch.ones(n , n) - torch.eye(n, n)
#         mask_dj_0 = mask_dj_0.cuda()

#         # 这步给相似度矩阵求exp,并且除以温度参数T
#         sim_matrix = torch.exp(sim_matrix/self.temperature)

#         # 这步将相似度矩阵的对角线上的值全置0，因为对比损失不需要自己与自己的相似度
#         sim_matrix = sim_matrix * mask_dj_0

#         # 这步产生了相同类别的相似度矩阵，标签相同的位置保存它们的相似度，其他位置都是0，对角线上也为0
#         sim = mask * sim_matrix

#         # 用原先的对角线为0的相似度矩阵减去相同类别的相似度矩阵就是不同类别的相似度矩阵
#         no_sim = sim_matrix - sim

#         #把不同类别的相似度矩阵按行求和，得到的是对比损失的分母(还差一个与分子相同的那个相似度，后面会加上)
#         no_sim_sum = torch.sum(no_sim , dim=1)

#         '''
#         将上面的矩阵扩展一下,再转置,加到sim(也就是相同标签的矩阵上),然后再把sim矩阵与sim_num矩阵做除法。
#         至于为什么这么做，就是因为对比损失的分母存在一个同类别的相似度，就是分子的数据。做了除法之后，就能得到
#         每个标签相同的相似度与它不同标签的相似度的值,它们在一个矩阵(loss矩阵)中。
#         '''
#         no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
#         sim_sum  = sim + no_sim_sum_expend
#         loss = torch.div(sim , sim_sum)

#         '''
#         由于loss矩阵中,存在0数值,那么在求-log的时候会出错。这时候,我们就将loss矩阵里面为0的地方
#         全部加上1,然后再去求loss矩阵的值,那么-log1 = 0 ，就是我们想要的。
#         '''
#         x =  torch.eye(n, n)
#         x = x.cuda()
#         loss = mask_no_sim + loss + x

#         #接下来就是算一个批次中的loss了
#         loss = -torch.log(loss)  #求-log
#         # loss = torch.sum(torch.sum(loss, dim=1)) / (len(torch.nonzero(loss)))
#         loss = torch.sum(torch.sum(loss, dim=1) )/(n*2)   

#         return loss     

    
class HGCL(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroConv({
                # news <-> entity relation
                ('news', 'has', 'entities'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('entities', 'in', 'news'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),

                # news <-> topic relation
                ('news', 'on', 'topic'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('topic', 'in', 'news'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                
                # entity <-> entity relation
                ('entities', 'similar', 'entities'): GATv2Conv(-1, hidden_channels, add_self_loops=False),
            }, aggr='sum')
            self.convs.append(conv)
            if i!= num_layers-1:
                # hidden_channels = 256
                hidden_channels = int(hidden_channels/2)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        # out = self.sigmoid(self.lin1(x_dict["news"]))
        return self.lin(x_dict['news'])
        # return self.lin(out)


# # self attention implement
# class Self_Attention(nn.Module):
#     def __init__(self, input_dim, dim_k, dim_v):
#         super(Self_Attention,self).__init__()
#         self.q = nn.Linear(input_dim,dim_k)
#         self.k = nn.Linear(input_dim,dim_k)
#         self.v = nn.Linear(input_dim,dim_v)
#         self._norm_fact = 1 / sqrt(dim_k)
        
    
#     def forward(self, x):
#         # Q: batch_size * seq_len * dim_k
#         Q = self.q(x) 
#         # K: batch_size * seq_len * dim_k
#         K = self.k(x) 
#         # V: batch_size * seq_len * dim_v
#         V = self.v(x) 
         
#         atten = nn.Softmax(dim=-1)(torch.bmm(Q,K.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len
        
#         output = torch.bmm(atten,V) # Q * K.T() * V # batch_size * seq_len * dim_v
        
        # return
def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

def semi_loss(z1: torch.Tensor, z2: torch.Tensor):
    # f = lambda x: torch.exp(x / torch.tensor(TAU, device = x.device))
    # print(sim(z1,z1))
    # refl_sim = f(sim(z1, z1))
    # between_sim = f((z1, z2))
    refl_sim = torch.exp(
        sim(z1, z1) / 0.5
    )
    between_sim = torch.exp(
        sim(z1, z2) / 0.5
    )

    return -torch.log(
        between_sim.diag()
        / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))


class MHGCL(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroConv({
                # news <-> entity relation
                ('news', 'has', 'entities'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('entities', 'in', 'news'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),

                # news <-> topic relation
                ('news', 'on', 'topic'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('topic', 'in', 'news'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                
                # entity <-> entity relation
                ('entities', 'similar', 'entities'): GATv2Conv(-1, hidden_channels, add_self_loops=False),

                # kg entities <-> news relation
                ('kg_entities', 'in', 'news'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('news', 'has', 'kg_entities'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),

                # entity <-> kg entity relation
                ('kg_entities', 'to', 'entity'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False)
            }, aggr='sum')
            self.convs.append(conv)

        self.convs1 = torch.nn.ModuleList()
        for i in range(num_layers):
            conv1 = HeteroConv({
                # news <-> entity relation
                ('news', 'has', 'entities'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('entities', 'in', 'news'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),

                # news <-> topic relation
                ('news', 'on', 'topic'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('topic', 'in', 'news'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                
                # entity <-> entity relation
                ('entities', 'similar', 'entities'): GATv2Conv(-1, hidden_channels, add_self_loops=False),

                # kg entities <-> news relation
                ('kg1_entities', 'in', 'news'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('news', 'has', 'kg1_entities'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),

                # entity <-> kg entity relation
                ('kg1_entities', 'to', 'entity'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),

                # new add
                 # kg entities <-> news relation
                # ('kg_entities', 'in', 'news'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                # ('news', 'has', 'kg_entities'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),

                # # entity <-> kg entity relation
                # ('kg_entities', 'to', 'entity'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False)
            }, aggr='sum')
            self.convs1.append(conv1)

        self.convs2 = torch.nn.ModuleList()
        for i in range(num_layers):
            conv2 = HeteroConv({
                # news <-> entity relation
                ('news', 'has', 'entities'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('entities', 'in', 'news'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),

                # news <-> topic relation
                ('news', 'on', 'topic'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('topic', 'in', 'news'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                
                # # entity <-> entity relation
                ('entities', 'similar', 'entities'): GATv2Conv(-1, hidden_channels, add_self_loops=False),

                # kg entities <-> news relation
                ('kg_entities', 'in', 'news'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('news', 'has', 'kg_entities'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('kg1_entities', 'in', 'news'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('news', 'has', 'kg1_entities'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),

                # entity <-> kg entity relation
                ('kg_entities', 'to', 'entity'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('kg1_entities', 'to', 'entity'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
            }, aggr='sum')
            self.convs2.append(conv2)

            # if i!= num_layers-1:
            #     # hidden_channels = 256
            #     hidden_channels = int(hidden_channels/2)
        # projection layer
        # self.lin = nn.Sequential(nn.Linear(hidden_channels, 128, bias=False),
        #                          nn.BatchNorm1d(128),
        #                          nn.ReLU(inplace=True),
        #                          nn.Linear(128, 64, bias=True))
        # self.lin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels, bias=False),
        #                          nn.BatchNorm1d(hidden_channels),
        #                          nn.ReLU(inplace=True),
        #                          nn.Linear(hidden_channels, 64))
        self.fc1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, hidden_channels)

        # self.q = nn.Linear(hidden_channels*3, 256)
        # self.k = nn.Linear(hidden_channels*3, 256)
        # self.v = nn.Linear(hidden_channels*3, 256)
        # self._norm_fact = 1 / math.sqrt(256)

        self.lin1 = Linear(64, out_channels)
        self.lin2 = Linear(hidden_channels*3, out_channels)

    def l2_norm(self,input,axis = 1):
        norm = torch.norm(input,2,axis,True)
        output = torch.div(input,norm)
        return output

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def forward(self, x_dict, edge_index_dict):

        for conv in self.convs:
            x_aug1 = conv(x_dict, edge_index_dict)
            x_aug1 = {key: x.relu() for key, x in x_aug1.items()}
        # out = self.sigmoid(self.lin1(x_dict["news"]))
        # # aug_one = self.lin(x_aug1['news']) 
    
        for conv in self.convs1:
            x_aug2 = conv(x_dict, edge_index_dict)
            x_aug2 = {key: x.relu() for key, x in x_aug2.items()}
        # # out = self.sigmoid(self.lin1(x_dict["news"]))
        # # aug_two = self.lin(x_aug2['news'])
        # news_embeddings = torch.cat([x_aug1['news'], x_aug2['news']], dim=1)
        # pos_feature = torch.sum(aug_one, dim=1)
        # pos_feature = pos_feature.unsqueeze(1)
        # neg_feature = torch.sum(aug_two, dim=1)
        # neg_feature = neg_feature.unsqueeze(1)

        # features = torch.cat([pos_feature, neg_feature], dim=1)
        # features = torch.sigmoid(features)
        # features = self.l2_norm(features, axis=-1)


        # embeddings for detection
        # print("aug_one:", aug_one.shape)
        # print("aug_two:", aug_two.shape)
        # print(news_embeddings.shape)
        # new add
        # aug_one1, aug_two1 = torch.flatten(aug_one, start_dim=1), torch.flatten(aug_two, start_dim=1)
        # news_embeddings = torch.cat([x_aug1['news'], x_aug2['news']], 1)


        for conv in self.convs2:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        # out = self.sigmoid(self.lin1(x_dict["news"]))
        # news_embeddings = torch.cat([news_embeddings, x_dict['news']], dim=1)
        aug_one = self.projection(x_aug1['news'])
        aug_two = self.projection(x_aug2['news'])
        aug_three = self.projection(x_dict['news'])
        news_embeddings = torch.cat([aug_one,aug_two], dim=1)
        news_embeddings = torch.cat([news_embeddings,aug_three], dim=1)
        # Q = self.q(news_embeddings) 
        
        # K = self.k(news_embeddings) 
       
        # V = self.v(news_embeddings) 
         
        
        # atten = nn.Softmax(dim=-1)(torch.mm(Q, K.T)) * self._norm_fact 

        # # Q * K.T() * V # batch_size * seq_len * dim_v
        # output = torch.mm(atten, V) 
        # print(output.shape)
        # output = self.lin1(output)
        # output1 = self.lin1(x_aug2['news'])

        # return F.normalize(aug_one1, dim=-1) , F.normalize(aug_two1, dim=-1), self.lin1(x_dict['news']) 
        return aug_one, aug_two, aug_three, self.lin2(news_embeddings) 
        # return self.lin1(x_aug1['news']), features
        # return self.lin2(output)
        
        # return self.lin(out)
        


# def unsup_train(model, data, args):
#     model.train()
#     lossCL = ContrasLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
#     for epoch in range(args.epochs):
#         total_loss = 0

#         optimizer.zero_grad()
#         out1, out2, _ = model(data.x_dict, data.edge_index_dict)

#         loss = lossCL(out1, out2)
#         loss.backward()
#         optimizer.step()
#         # print("epoch", epoch, 'loss:', loss.detach().item())

def train(model, data, args):
    # hyparameter: adjust
    # a = 0.2
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # lossCL = SupContrasLoss()
    # lossCL = ContrasLoss()
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        # out1, out2, out = model(data.x_dict, data.edge_index_dict)
        out1, out2, out3, out = model(data.x_dict, data.edge_index_dict)
        mask = data['news'].train_mask

        # New code
        # cal cl loss
        kg1_cl_loss = (semi_loss(out1, out3) + semi_loss(out3, out1)) * 0.5
        kg2_cl_loss = (semi_loss(out2, out3) + semi_loss(out3, out2)) * 0.5
        kg1_cl_loss = kg1_cl_loss.mean()
        kg2_cl_loss = kg2_cl_loss.mean()
        clf_loss = criterion(out[mask], data['news'].y[mask])
        # if epoch >= 200:
        #     loss = clf_loss
        # else:
        #     loss = (kg1_cl_loss + kg2_cl_loss) / 2
        loss = args.alpha * (kg1_cl_loss + kg2_cl_loss) / 2  + clf_loss
        # print("epoch", epoch, 'loss:', loss.detach().item())


        # print(out[mask].shape, data['news'].y[mask])
        # new revise, maynbe the lossCL did not converge when epoch <= 200
        # if epoch >= 200:
        #     loss = lossCL(out1, out2) + args.alpha * criterion(out[mask], data['news'].y[mask])
        #     # loss = lossCL(out1, out2, data['news'].y) + args.alpha * criterion(out[mask], data['news'].y[mask])
        # else:
        #     # loss = lossCL(out1, out2, data['news'].y)
        #     loss = lossCL(out1, out2)

        # loss = criterion(out[mask], data['news'].y[mask])
        # print("epoch", epoch, 'loss:', loss.detach().item())

        loss.backward()
        optimizer.step()

def test(model, data, args):
    # _, _, out = model(data.x_dict, data.edge_index_dict)
    _, _, _, out = model(data.x_dict, data.edge_index_dict)
    pred = out[data['news'].test_mask].argmax(dim=1).cpu()

    y = data['news'].y[[data['news'].test_mask]].cpu()
    # pred_list = out[data['news'].test_mask].tolist()
    # predict = []
    def softmax(p):
        e_x = torch.exp(p)
        partition_x = e_x.sum(1, keepdim=True)
        return e_x / partition_x
    predict = softmax(out[data['news'].test_mask])
    col, row = predict.shape
    # print(col)
    pred_list = []
    for i in range(col):
        pred_list.append(predict[i][1].cpu().tolist())
    pred_list = torch.Tensor(pred_list)

    # print("pred_list is", pred_list)
    # print('label is', y)

  
    acc = accuracy_score(y, pred)
    precision = precision_score(y, pred, )
    # 修改
    f1 = f1_score(y, pred)
    recall = recall_score(y, pred,)
    # f1_1 = f1_score(y, pred, average='weighted')
    # f1_2  = f1_score(y, pred, pos_label=0)

    auc = roc_auc_score(y, pred,)
    print(f"Testing Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f},F1: {f1:.4f}")
    with open("./Para_analysis.txt", "a+", encoding="utf8") as f:
        f.write(f"epoch: {args.epochs}; hidden_channels: {args.hidden_channels} ; Acc:{acc:.4f}; Precision: {precision:.4f}; Recall: {recall:.4f}; F1: {f1:.4f} \n")

    
    # draw ROC curve 
    # fpr, tpr, thresholds = roc_curve(y, pred_list)
    # np.savetxt("./ROC_CURVE/MHGCL_FPR_{}.txt".format(args.dataset), fpr)
    # np.savetxt("./ROC_CURVE/MHGCL_TPR_{}.txt".format(args.dataset), tpr)
    # # np.loadtxt()
    # # print(fpr)
    # roc_auc = sm.auc(fpr, tpr)

    # plt.figure()
    # lw = 2
    # plt.figure(figsize=(10, 10))
    # # 假正率为横坐标， 真正率为纵坐标
    # plt.plot(fpr, tpr, color='red', lw=lw, label="ROC curve (area = %0.2f)" % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])

    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')


    # plt.title('Receiver operating characteristic curve(ROC)_{}'.format(args.dataset))
    # plt.legend(loc="lower right")
    # plt.savefig('./ROC/roc_curve_{}_{}.png'.format(args.dataset, i))
    # plt.show()