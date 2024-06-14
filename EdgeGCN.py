"""
@Author: Weida Wang
@Contact: tjudavidwang@gmail.com
@File: EdgeGCN.py
@Time: 2024/06/15 1:44 AM
"""

import argparse
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F

class EdgeGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, k=20, dropout=0.0, dynamic=True, dilation=1, stochastic=True, epsilon=0.2, head=True):
        """
        Initialize the EdgeGCNLayer.

        Parameters:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        k (int): Number of nearest neighbors.
        dropout (float): Dropout rate.
        dynamic (bool): Whether to use dynamic graph construction.
        dilation (int): Dilation rate for the convolution.
        stochastic (bool): Whether to use stochastic graph construction.
        epsilon (float): Epsilon value for stochastic graph construction.
        head (bool): Indicates if this is the head layer.
        """
        super(EdgeGCNLayer, self).__init__()
        self.k = int(k)  # Number of nearest neighbors
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Convolutional layers for feature extraction
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout)  # Dropout layer
        )

        self.dynamic = dynamic
        self.stochastic = stochastic
        self.head = head
        self.epsilon = epsilon
        self.dilation = int(dilation)
        self.model_init()

    def model_init(self):
        """Initialize model parameters."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def knn(self, x, k):
        """
        Compute k-nearest neighbors for each point in the point cloud.

        Parameters:
        x (torch.Tensor): Input tensor of shape (batch_size, num_dims, num_points).
        k (int): Number of nearest neighbors.

        Returns:
        torch.Tensor: Indices of k-nearest neighbors.
        """
        with torch.no_grad():
            x = x.detach()
            if len(x.shape) == 3:
                batch_size, num_dims, num_points = x.shape
                inner = -2 * torch.matmul(x.transpose(2, 1), x)
                xx = torch.sum(x ** 2, dim=1, keepdim=True)
                pairwise_distance = -xx - inner - xx.transpose(2, 1)
                idx = pairwise_distance.topk(k, dim=-1)[1]  # (batch_size, num_points, k)
            elif len(x.shape) == 2:
                num_dims, num_points = x.shape
                x = x.contiguous().transpose(1, 0).contiguous()
                x_inner = -2 * torch.matmul(x, x.t())
                x_square = torch.sum(x ** 2, dim=-1, keepdim=True)
                dist = x_square + x_inner + x_square.t()
                _, idx = torch.topk(-dist, k=k, dim=-1)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                num_points = x.size(0)
                row_idx = torch.arange(num_points, device=device).view(-1, 1).repeat(1, k).view(-1)
                col_idx = idx.view(-1)
                idx = torch.stack([row_idx, col_idx], dim=0)  # [2, num_points*k]

        return idx

    def dense_knn_matrix(self, x, k=20):
        """
        Compute dense k-nearest neighbors matrix.

        Parameters:
        x (torch.Tensor): Input tensor of shape (batch_size, num_points, num_dims).
        k (int): Number of nearest neighbors.

        Returns:
        torch.Tensor: Dense k-nearest neighbors matrix.
        """
        with torch.no_grad():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            batch_size, n_points, n_dims = x.shape
            nn_idx = self.knn(x, k)
            center_idx = torch.arange(0, n_points, device=device).expand(batch_size, k, -1).transpose(2, 1)
        return torch.stack((nn_idx, center_idx), dim=0)

    def init_knn(self, x, k):
        """
        Initialize k-nearest neighbors for the input.

        Parameters:
        x (torch.Tensor): Input tensor of shape (batch_size, num_channels, num_points).
        k (int): Number of nearest neighbors.

        Returns:
        torch.Tensor: Indices of k-nearest neighbors.
        """
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
        """
        Perform batched index selection.

        Parameters:
        x (torch.Tensor): Input tensor of shape (batch_size, num_dims, num_vertices).
        idx (torch.Tensor): Indices tensor.

        Returns:
        torch.Tensor: Selected features tensor.
        """
        batch_size, num_dims, num_vertices = x.shape[:3]
        k = idx.shape[-1]
        device = x.device
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_vertices
        idx = idx + idx_base
        idx = idx.view(-1)

        x = x.transpose(2, 1).contiguous()  # 确保张量在内存中是连续的，以便后续的view操作不会引发不必要的拷贝
        feature = torch.index_select(x.view(batch_size * num_vertices, num_dims), 0, idx)  # 直接在展平的张量上进行索引选择，避免多次reshape操作
        feature = feature.view(batch_size, num_vertices, k, num_dims).permute(0, 3, 1, 2)  # 调整张量维度
        return feature  # batch_size, num_dims, num_vertices, k



    def get_graph_feature(self, x, idx=None):
        """
        Compute graph features for the input.

        Parameters:
        x (torch.Tensor): Input tensor of shape (batch_size, num_dims, num_points).
        idx (torch.Tensor, optional): Indices tensor.

        Returns:
        tuple: Feature tensor and indices tensor.
        """
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        if idx is None:  # dynamic == True
            if self.head:
                edge_index = self.init_knn(x, int(self.k * self.dilation))
            else:
                edge_index = self.init_knn(x, int(self.k * self.dilation))
                
            if self.stochastic:
                if torch.rand(1) < self.epsilon and self.training:
                    num = self.k * self.dilation
                    randnum = torch.randperm(num)[:self.k]
                    idx = edge_index[:, :, :, randnum]
                else:
                    idx = edge_index[:, :, :, ::self.dilation]
            else:
                idx = edge_index[:, :, ::self.dilation]  # (batch_size, num_points, k)
        else:
            edge_index = idx

        x_i = self.batched_index_select(x, edge_index[1])
        x_j = self.batched_index_select(x, edge_index[0])
        
        feature = torch.cat([x_i, x_j - x_i], dim=1)

        return feature, idx

    def forward(self, x, edge_idx=None):
        """
        Forward pass of the EdgeGCN layer.

        Parameters:
        x (torch.Tensor): Input tensor of shape (batch_size, num_dims, num_points).
        edge_idx (torch.Tensor, optional): Edge indices tensor.

        Returns:
        tuple: Output tensor and edge indices tensor.
        """
        batch_size, num_dims, num_points = x.size()
        x, edge_idx = self.get_graph_feature(x, edge_idx)  # (batch_size, 2*num_dims, num_points, k)
        x = self.conv(x)  # (batch_size, out_channels, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]
        return x, edge_idx

class ResGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, k, dropout=0., dynamic=True, dilation=1, head=False):
        """
        Initialize the ResGCNLayer.

        Parameters:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        k (int): Number of nearest neighbors.
        dropout (float): Dropout rate.
        dynamic (bool): Whether to use dynamic graph construction.
        dilation (int): Dilation rate for the convolution.
        head (bool): Indicates if this is the head layer.
        """
        super(ResGCNLayer, self).__init__()
        self.gcn = EdgeGCNLayer(in_channels, out_channels, k=k, dropout=dropout, dynamic=dynamic, dilation=dilation, head=head)
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x, edge_idx):
        """
        Forward pass of the ResGCN layer.

        Parameters:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, num_points).
        edge_idx (torch.Tensor): Edge indices tensor.

        Returns:
        tuple: Output tensor and edge indices tensor.
        """
        out, edge_idx = self.gcn(x, edge_idx)
        res = self.residual(x)
        return F.relu(out + res), edge_idx

class EdgeGCN(nn.Module):
    def __init__(self, args, output_channels=40):
        """
        Initialize the EdgeGCN network.

        Parameters:
        args (argparse.Namespace): Arguments for the network configuration.
        output_channels (int): Number of output channels (default is 40).
        """
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
            # Append ResGCN layers
            if i == 0:
                self.gcn_layers.append(ResGCNLayer(in_channels, out_channels, k=self.k, dilation=dilation, head=True))
            else:
                self.gcn_layers.append(ResGCNLayer(in_channels, out_channels, k=self.k, dilation=dilation, head=False))
           
            if self.dilated and i != 0:  # Increase dilation rate for layers after the first one
                dilation += 1
    
            in_channels = out_channels
            gcns_channel += out_channels

        self.bn_final = nn.BatchNorm1d(args.emb_dims)
        self.conv_final = nn.Sequential(
            nn.Conv1d(gcns_channel, args.emb_dims, kernel_size=1, bias=False),
            self.bn_final,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        """
        Forward pass of the EdgeGCN network.

        Parameters:
        x (torch.Tensor): Input tensor of shape (batch_size, num_channels, num_points).

        Returns:
        torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        batch_size = x.size(0)
        num_points = x.size(2)

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

        x = self.conv_final(x)  # [batch_size, args.emb_dims, num_points]
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # [batch_size, args.emb_dims]
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)  # [batch_size, args.emb_dims]
        x = torch.cat((x1, x2), 1)  # [batch_size, args.emb_dims * 2]

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  # [batch_size, 512]
        x = self.dp1(x)  # [batch_size, 512]
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)  # [batch_size, 256]
        x = self.dp2(x)  # [batch_size, 256]
        x = self.linear3(x)  # [batch_size, output_channels]
        return x

# Usage example
if __name__ == "__main__":
    batch_size = 16
    num_points = 1024
    in_channels = 3

    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N', help='Pretrained model path')
    parser.add_argument('--in_channels', type=int, default=3, help='Dimension of input ')
    parser.add_argument('--n_classes', type=int, default=40, help='Number of output classes')
    parser.add_argument('--n_blocks', type=int, default=5, help='Number of basic blocks in the backbone')
    parser.add_argument('--n_filters', default=64, type=int, help='Number of channels of deep features')   
    parser.add_argument('--dynamic', default=False, type=bool, help='Dynamic for adjacency matrix')
    parser.add_argument('--dilated', default=True, type=bool, help='Dilated graph convolution')
        
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(batch_size, in_channels, num_points).to(device)
    print(x.shape)
    model = EdgeGCN(args).to(device)
    out = model.forward(x)
    print(out.shape)  # Expected shape: (batch_size, num_classes)
