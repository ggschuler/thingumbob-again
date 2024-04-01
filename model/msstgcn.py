import torch
import torch.nn as nn
from torch import nn
import torch.nn.functional as F
from graph import Graph

class msstgcn(nn.Module):
    def __init__(self, dil, num_layers_R, num_R, num_f_maps, dim, num_classes, connections, pool):
        super(msstgcn, self).__init__()
        self.stream = SpatialTemporalGraph(connections=connections, pool=pool, filters=num_f_maps, in_channels=dim, num_class=num_classes, dil=dil,)
    def forward(self, x, mask):
        out = self.stream(x) * mask[:, 0:1, :]
        outputs = out.unsqueeze(0)
        probabilities = F.softmax(outputs, dim=2)
        return probabilities

class SpatialTemporalGraph(nn.Module):

    def __init__(self, connections, pool, filters, in_channels=2, num_class=2, dil=[1,2,4,8,16,32,64,128,256,512,1024,2048], edge_importance_weighting=False, **kwargs):
        super(SpatialTemporalGraph, self).__init__()
        graph_args = {'connections':connections, 'layout':'minirgdb', 'strategy':'uniform'}
        self.pool = pool
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 5
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        self.conv_1x1 = nn.Conv2d(in_channels, filters, 1)
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[0], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[1], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[2], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[3], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[4], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[5], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[6], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[7], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[8], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[9], residual=True)
        ))

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([nn.Parameter(torch.ones(self.A.size())) for i in self.st_gcn_networks])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.conv_out = nn.Conv1d(filters, num_class, kernel_size=1)

    def forward(self, x):
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(N, C, T, V)
        x = self.conv_1x1(x)
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
        if self.pool == 'avg':
            x = F.avg_pool2d(x, kernel_size=(1, V))
        elif self.pool == 'max':
            x = F.max_pool2d(x, kernel_size=(1, V))
        out = self.conv_out(x.squeeze())
        return out

class st_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, A=None, dilation=1, residual=True):
        super(st_gcn, self).__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        pad = int((dilation*(kernel_size[0]-1))/2)
        self.gcn = ConvTempGraph(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(nn.BatchNorm2d(out_channels),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size[0], 1), stride=(stride, 1), padding=(pad, 0), dilation=(dilation, 1)),
                                 nn.BatchNorm2d(out_channels))

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                                          nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = x + res
        return x

class ConvTempGraph(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, tkernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size, kernel_size=(tkernel_size, 1), padding=(t_padding, 0), stride=(t_stride, 1), dilation=(t_dilation, 1), bias=bias)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        x = self.batch_norm(x)
        return x.contiguous(), A

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False