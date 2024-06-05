import argparse
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F

class EdgeGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, k=20, dropout=0.0, dynamic=True, dilation=1,stochastic=True, epsilon=0.2, head=True):
        super(EdgeGCNLayer, self).__init__()
        self.k = int(k)
        self.in_channels = in_channels
        self.out_channels = out_channels

        # self.weight = Parameter(torch.FloatTensor(in_channels * 2, out_channels))  # [in*2, out]
        # if bias:
        #     self.bias = Parameter(torch.FloatTensor(out_channels))  # [out]
        # else:
        #     self.register_parameter('bias', None)
        self.conv = nn.Sequential(nn.Conv2d(in_channels*2, out_channels, kernel_size=1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Dropout(dropout)  # 添加Dropout层
                                )
        self.dynamic = dynamic
        self.stochastic = stochastic
        self.head = head
        self.epsilon = epsilon
        self.dilation = int(dilation)
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True
                    
    def knn(self, x, k):
        with torch.no_grad():
            x = x.detach() # detach
            if len(x.shape)==3:
                batch_size, num_dims, num_points = x.shape
                inner = -2 * torch.matmul(x.transpose(2, 1), x)
                xx = torch.sum(x ** 2, dim=1, keepdim=True)
                pairwise_distance = -xx - inner - xx.transpose(2, 1)
                idx = pairwise_distance.topk(k, dim=-1)[1]  # (batch_size, num_points, k)
            elif len(x.shape)==2:
                num_dims, num_points = x.shape
                x = x.contiguous().transpose(1, 0).contiguous()
                x_inner = -2 * torch.matmul(x, x.t())
                x_square = torch.sum(x ** 2, dim=-1, keepdim=True)
                dist = x_square + x_inner + x_square.t()

                _, idx = torch.topk(-dist, k=k, dim=-1)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # Create edge index
                num_points = x.size(0)
                row_idx = torch.arange(num_points, device=device).view(-1, 1).repeat(1, k).view(-1)
                col_idx = idx.view(-1)

                idx = torch.stack([row_idx, col_idx], dim=0) # [2, num_points*k]

        return idx

    def dense_knn_matrix(self, x, k=20):
        with torch.no_grad():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            batch_size, n_points, n_dims = x.shape
            nn_idx = self.knn(x,k)
            center_idx = torch.arange(0, n_points, device=device).expand(batch_size, k, -1).transpose(2, 1)
        return torch.stack((nn_idx, center_idx), dim=0)

    def init_knn(self, x, k):
        x = x.squeeze(-1)
        B, C, N = x.shape
        edge_index = []
        for i in range(B):
            edgeindex = self.knn(x[i], self.k * self.dilation)
            edgeindex = edgeindex.view(2, N, self.k * self.dilation)
            edge_index.append(edgeindex)
        edge_index = torch.stack(edge_index, dim=1)
        return edge_index
    
    def batched_index_select(self, x, idx):
        batch_size, num_dims, num_vertices = x.shape[:3]
        k = idx.shape[-1]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_vertices
        idx = idx + idx_base
        idx = idx.contiguous().view(-1)

        x = x.transpose(2, 1)
        feature = x.contiguous().view(batch_size * num_vertices, -1)[idx, :]
        feature = feature.view(batch_size, num_vertices, k, num_dims).permute(0, 3, 1, 2).contiguous()
        return feature # batch_size, k, num_dims, num_vertices

    def get_graph_feature(self, x, idx=None):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        if idx == None: # dynamic == True
            if self.head:
                edge_index = self.init_knn(x, int(self.k*self.dilation))
            else:
                edge_index = self.init_knn(x, int(self.k*self.dilation))
                
            if self.stochastic:
                if torch.rand(1) < self.epsilon and self.training:
                    num = self.k * self.dilation
                    randnum = torch.randperm(num)[:self.k]
                    idx = edge_index[:, :, :, randnum]
                else:
                    idx = edge_index[:, :, :, ::self.dilation]
            else:
                idx = edge_index[:,:,::self.dilation]  # (batch_size, num_points, k)
        else:
            edge_index = idx

        x_i = self.batched_index_select(x, edge_index[1])
        x_j = self.batched_index_select(x, edge_index[0])
        
        feature = torch.cat([x_i, x_j - x_i], dim=1)

        return feature, idx

    def forward(self, x, edge_idx=None):
        batch_size, num_dims, num_points = x.size()
        x, edge_idx = self.get_graph_feature(x, edge_idx)  # (batch_size, 2*num_dims, num_points, k)
        x = self.conv(x)  # (batch_size, out_channels, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]
        return x, edge_idx

class ResGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, k, dropout=0., dynamic=True, dilation=1,head=False):
        super(ResGCNLayer, self).__init__()
        self.gcn = EdgeGCNLayer(in_channels, out_channels, k=k, dropout=dropout, dynamic=dynamic, dilation=dilation, head=head)
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x, edge_idx):
        out, edge_idx = self.gcn(x, edge_idx)
        res = self.residual(x)
        return F.relu(out + res), edge_idx


class EdgeGCN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(EdgeGCN, self).__init__()
        self.args = args
        self.k = args.k

        self.gcn_layers = nn.ModuleList()
        in_channels = args.in_channels
        out_channels = 64
        output_channels = args.n_classes
        num_layers = args.n_blocks
        self.dynamic = args.dynamic
        self.dilated = args.dilated

        gcns_channel = 0
        dilation = 1
        for i in range(num_layers):
            # self.gcn_layers.append(ResGCNLayer(in_channels, out_channels,dropout=args.dropout))
            if i == 0:
                self.gcn_layers.append(ResGCNLayer(in_channels, out_channels, k=self.k, dilation=dilation,head=True))
            else:
                self.gcn_layers.append(ResGCNLayer(in_channels, out_channels, k=self.k, dilation=dilation,head=False))
           
            if self.dilated and i!=0: # from the third layer
                dilation += 1
    
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
        

    # def knn(self, x, k):
    #     inner = -2 * torch.matmul(x.transpose(2, 1), x)
    #     xx = torch.sum(x**2, dim=1, keepdim=True)
    #     pairwise_distance = -xx - inner - xx.transpose(2, 1)
    #     idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    #     return idx

    # def get_adj_matrix(self, x, dilation=1):
    #     batch_size = x.size(0)
    #     num_points = x.size(2)
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     idx = self.knn(x, self.k* dilation)
    #     idx = idx[:, :, ::dilation] # for dilated case

    #     idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    #     idx = idx + idx_base
    #     idx = idx.view(-1)

    #     adj = torch.zeros(batch_size * num_points, batch_size * num_points, device=device)
    #     adj[idx, torch.arange(batch_size * num_points, device=device).repeat_interleave(self.k)] = 1
    #     return adj

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        # # Reshape x to (batch_size * num_points, 3)
        # x = x.permute(0, 2, 1).reshape(batch_size * num_points, -1)

        # Apply GCN layers
        outputs = []
        edge_idx = None
        for i, gcn in enumerate(self.gcn_layers):
            if self.dynamic:
                x, _ = gcn(x, edge_idx)
            else:
                x, edge_idx = gcn(x, edge_idx)
            outputs.append(x.view(batch_size, num_points, -1).permute(0, 2, 1))


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

# Usage example
if __name__ == "__main__":
    batch_size = 16
    num_points = 1024
    in_channels = 3
   # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--in_channels', type=int, default=3, help='Dimension of input ')
    parser.add_argument('--n_classes', type=int, default=40, help='Dimension of out_channels ')

    # DeepGCNs
    parser.add_argument('--n_blocks', type=int, default=5, help='number of basic blocks in the backbone')
    parser.add_argument('--n_filters', default=64, type=int, help='number of channels of deep features')   

    parser.add_argument('--dynamic', default=False, type=bool, help='dynamic for adj matrix')
    parser.add_argument('--dilated', default=True, type=bool, help='dilated graph convolution')

        
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(batch_size, in_channels, num_points).to(device)
    print(x.shape)
    model = EdgeGCN(args).to(device)
    out = model.forward(x)
    print(out.shape)  # Expected shape: (batch_size, out_channels, num_points)