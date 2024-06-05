import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F

def batched_index_select(x, idx):
    r"""fetches neighbors features from a given neighbor idx

    Args:
        x (Tensor): input feature Tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times 1}`.
        idx (Tensor): edge_idx
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times l}`.
    Returns:
        Tensor: output neighbors features
            :math:`\mathbf{X} \in \mathbb{R}^{B \times C \times N \times k}`.
    """
    batch_size, num_dims, num_vertices = x.shape[:3]
    k = idx.shape[-1]
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)

    x = x.transpose(2, 1)
    feature = x.contiguous().view(batch_size * num_vertices, -1)[idx, :]
    feature = feature.view(batch_size, num_vertices, k, num_dims).permute(0, 3, 1, 2).contiguous()
    return feature


class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels) if norm else nn.Identity(),
            nn.ReLU() if act == 'relu' else nn.LeakyReLU(0.2)
        )

    def forward(self, x, edge_index):
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value.squeeze(-1)


class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(GCNLayer, self).__init__()
        self.gcnconv = EdgeConv2d(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        x = self.gcnconv(x, adj)
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

class DenseDilated(nn.Module):
    """
    Find dilated neighbor from neighbor list

    edge_index: (2, batch_size, num_points, k)
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, :, randnum]
            else:
                edge_index = edge_index[:, :, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, :, ::self.dilation]
        return edge_index

def pairwise_distance(x):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    x_inner = -2*torch.matmul(x, x.transpose(2, 1))
    x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
    return x_square + x_inner + x_square.transpose(2, 1)

def dense_knn_matrix(x, k=16):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points ,k) (batch_size, num_points, k)
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        _, nn_idx = torch.topk(-pairwise_distance(x.detach()), k=k)
        center_idx = torch.arange(0, n_points, device=x.device).expand(batch_size, k, -1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


class DenseDilatedKnnGraph(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0, head=False):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)
        self.knn = dense_knn_matrix
        self.head = head

    def knn_graph(self, x, k):
        """
        Get KNN graph based on the pairwise distance.
        Args:
            x: (num_points, num_dims)
            k: int
        Returns:
            edge_index: (2, num_points * k)
        """
        # Compute pairwise distance
        x_inner = -2 * torch.matmul(x, x.t())
        x_square = torch.sum(x ** 2, dim=-1, keepdim=True)
        dist = x_square + x_inner + x_square.t()

        # Find the k nearest neighbors
        _, idx = torch.topk(-dist, k=k, dim=-1)

        # Create edge index
        num_points = x.size(0)
        row_idx = torch.arange(num_points, device=x.device).view(-1, 1).repeat(1, k).view(-1)
        col_idx = idx.view(-1)

        edge_index = torch.stack([row_idx, col_idx], dim=0)
        return edge_index
    
    def forward(self, x):
        if self.head:
            x = x.squeeze(-1)
            B, C, N = x.shape
            edge_index = []
            for i in range(B):
                edgeindex = self.knn_graph(x[i].contiguous().transpose(1, 0).contiguous(), self.k * self.dilation)
                edgeindex = edgeindex.view(2, N, self.k * self.dilation)
                edge_index.append(edgeindex)
            edge_index = torch.stack(edge_index, dim=1)
        else:
            edge_index = self.knn(x, self.k * self.dilation)

        return self._dilated(edge_index)

class DynConv2d(EdgeConv2d):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, knn='matrix', head=False):
        super(DynConv2d, self).__init__(in_channels, out_channels)
        self.k = kernel_size
        self.d = dilation
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon,head)

    def forward(self, x, edge_index=None):
        if edge_index is None:
            edge_index = self.dilated_knn_graph(x)
        return super(DynConv2d, self).forward(x, edge_index)
    
class ResDynBlock2d(nn.Module):
    """
    Residual Dynamic graph convolution block
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, knn='matrix', res_scale=1, head=False):
        super(ResDynBlock2d, self).__init__()
        head = (res_scale==0)
        self.body = DynConv2d(in_channels, out_channels, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, knn, head)
        self.res_scale = res_scale

    def forward(self, x, edge_index=None):
        if self.res_scale == 0:
            return self.body(x, edge_index)
        return self.body(x, edge_index) + x*self.res_scale

class CenGCN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(CenGCN, self).__init__()
        self.args = args
        self.k = args.k

        self.gcn_layers = nn.ModuleList()
        in_channels = args.in_channels
        out_channels = 64
        num_layers = args.n_blocks
        self.dynamic = args.dynamic
        
        gcns_channel = 0
        for i in range(num_layers):
            # self.gcn_layers.append(ResGCNLayer(in_channels, out_channels,dropout=args.dropout))
            if i ==0:
                self.gcn_layers.append(ResDynBlock2d(in_channels, out_channels, self.k, i + 1,res_scale=0))
            else:
                self.gcn_layers.append(ResDynBlock2d(in_channels, out_channels, self.k, i))
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
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True
                    
    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        # # Reshape x to (batch_size * num_points, 3)
        # x = x.permute(0, 2, 1).reshape(batch_size * num_points, -1)

        # Apply GCN layers
        outputs = []
        for gcn in self.gcn_layers:
            x = gcn(x)
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