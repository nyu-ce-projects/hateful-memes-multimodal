import torch
from torch.nn import Sequential,Linear,ReLU,BatchNorm1d
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import MLP, MLPAggregation

class GraphClassifier(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, batch_norm=True,
                 dropout=0.5):
        super(GraphClassifier, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.lins = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(Linear(in_channels, out_channels))
            self.batch_norms.append(BatchNorm1d(out_channels))
            in_channels = out_channels

        self.reset_parameters()

    def reset_parameters(self):
        for lin, batch_norm in zip(self.lins, self.batch_norms):
            lin.reset_parameters()
            batch_norm.reset_parameters()

    def forward(self, x, g_data):
        x = global_mean_pool(x,g_data.batch)
        for i, (lin, bn) in enumerate(zip(self.lins, self.batch_norms)):
            if i == self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = lin(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = bn(x) if self.batch_norm else x
        x = x.view(-1)
        return x

    def __repr__(self):
        return '{}({}, {}, num_layers={}, batch_norm={}, dropout={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_layers, self.batch_norm, self.dropout)


class AdaptiveReadoutMLPClassifier(torch.nn.Module):
    def __init__(self, in_channels,hidden_channels,out_channels,max_num_elements,num_layers) -> None:
        super().__init__()

        self.mlp_aggregation = MLPAggregation(in_channels,out_channels,max_num_elements,num_layers=num_layers,hidden_channels=hidden_channels)


    def forward(self,x,g_data):
        batch = g_data.batch
        x = self.mlp_aggregation(x,batch)
        x = x.view(-1)
        return x



    