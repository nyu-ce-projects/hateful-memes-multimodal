import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.models import InnerProductDecoder, VGAE
from torch_geometric.nn.conv import GCNConv,GATv2Conv
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.gcn_shared = GCNConv(in_channels, hidden_channels)
        self.gcn_mu = GCNConv(hidden_channels, out_channels)
        self.gcn_logvar = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.gcn_shared(x, edge_index))
        mu = self.gcn_mu(x, edge_index)
        logvar = self.gcn_logvar(x, edge_index)
        return mu, logvar
    
class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,nheads,dropout):
        super(GATEncoder, self).__init__()
        self.gcn_shared = GATv2Conv(in_channels, hidden_channels,nheads,dropout=dropout)
        self.gcn_mu = GATv2Conv(hidden_channels, out_channels,nheads,dropout=dropout)
        self.gcn_logvar = GATv2Conv(hidden_channels, out_channels,nheads,dropout=dropout)

    def forward(self, x, edge_index):
        x = F.relu(self.gcn_shared(x, edge_index))
        mu = self.gcn_mu(x, edge_index)
        logvar = self.gcn_logvar(x, edge_index)
        return mu, logvar


class DeepVGAE(VGAE):
    def __init__(self, encoder,decoder=InnerProductDecoder()):
        super(DeepVGAE, self).__init__(encoder,decoder)

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        adj_pred = self.decoder.forward_all(z)
        return adj_pred