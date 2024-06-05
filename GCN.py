import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features)) # [in, out]
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features)) # [out]
        else:
            self.register_parameter('bias', None)
        self.reset_parameters() # limit in a small range

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight) # X W dense matrix
        output = torch.spmm(adj, support) # A(sparse) XW(dense)
        if self.bias is not None:
            return output + self.bias
        else:
            return output # AXW + b

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(GCNLayer, self).__init__()
        self.conv = GraphConvolution(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        x = self.conv(x, adj)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class ResGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(ResGCNLayer, self).__init__()
        self.gcn = GCNLayer(in_channels, out_channels, dropout)
        self.residual = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x, adj):
        out = self.gcn(x, adj)
        res = self.residual(x)
        return F.relu(out + res)


class GCN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(GCN, self).__init__()
        self.args = args
        self.k = args.k

        self.gcn_layers = nn.ModuleList()
        in_channels = args.in_channels
        out_channels = 64
        num_layers = args.n_blocks
        self.dynamic = args.dynamic
        self.dilated = args.dilated

        gcns_channel = 0
        for i in range(num_layers):
            # self.gcn_layers.append(ResGCNLayer(in_channels, out_channels,dropout=args.dropout))
            self.gcn_layers.append(ResGCNLayer(in_channels, out_channels))
            in_channels = out_channels
            gcns_channel += out_channels
            # if i % 2 == 1 and out_channels < 512:  # Increase channels progressively
            #     out_channels *= 2

        self.bn_final = nn.BatchNorm1d(args.emb_dims)
        self.conv_final = nn.Sequential(nn.Conv1d(gcns_channel, args.emb_dims, kernel_size=1, bias=False),
                                        self.bn_final,
                                        nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)
        

    def knn(self, x, k):
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
        return idx

    def get_adj_matrix(self, x, dilation=1):
        batch_size = x.size(0)
        num_points = x.size(2)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        idx = self.knn(x, self.k* dilation)
        idx = idx[:, :, ::dilation] # for dilated case

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        adj = torch.zeros(batch_size * num_points, batch_size * num_points, device=device)
        adj[idx, torch.arange(batch_size * num_points, device=device).repeat_interleave(self.k)] = 1
        return adj

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        # Reshape x to (batch_size * num_points, 3)
        x = x.permute(0, 2, 1).reshape(batch_size * num_points, -1)

        # Generate adjacency matrix using KNN
        adj = self.get_adj_matrix(x.view(batch_size, 3, num_points))

        # Apply GCN layers
        outputs = []
        dilation = 1
        
        for gcn in self.gcn_layers:
            x = gcn(x, adj)
            outputs.append(x.view(batch_size, num_points, -1).permute(0, 2, 1))
            if self.dynamic:
                adj = self.get_adj_matrix(x.view(batch_size, num_points, -1).permute(0, 2, 1), dilation)
            if self.dilated:
                dilation += 1

        x = torch.cat(outputs, dim=1)

        x = self.conv_final(x)  # [8, args.emb_dims, 1024]
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # [8, args.emb_dims]
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)  # [8, args.emb_dims]
        x = torch.cat((x1, x2), 1)  # [8, args.emb_dims * 2]

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  # [8, 512]
        x = self.dp1(x)  # [8, 512]
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)  # [8, 256]
        x = self.dp2(x)  # [8, 256]
        x = self.linear3(x)  # [8, 40]
        return x