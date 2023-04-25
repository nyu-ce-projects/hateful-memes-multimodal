import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.models import InnerProductDecoder, VGAE
from torch_geometric.nn.conv import GCNConv,GATv2Conv
from torch_geometric.nn.norm import GraphNorm,BatchNorm
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
from torch_geometric.nn import JumpingKnowledge, GATConv
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, average_precision_score

class GCNVGAEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNVGAEEncoder, self).__init__()
        self.gcn_shared = GCNConv(in_channels, hidden_channels)
        self.gcn_mu = GCNConv(hidden_channels, out_channels)
        self.gcn_logvar = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.gcn_shared(x, edge_index))
        mu = self.gcn_mu(x, edge_index)
        logvar = self.gcn_logvar(x, edge_index)
        return mu, logvar
    
class GATVGAEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,nheads,dropout):
        super(GATVGAEEncoder, self).__init__()
        self.gcn_shared = GATv2Conv(in_channels, hidden_channels,nheads,dropout=dropout)
        # self.jump = JumpingKnowledge(mode='cat')
        self.gcn_mu = GATv2Conv(hidden_channels*nheads, out_channels,1,dropout=dropout)
        self.gcn_logvar = GATv2Conv(hidden_channels*nheads, out_channels,1,dropout=dropout)

    def forward(self, x, edge_index):
        # x = self.jump(F.relu(self.gcn_shared(x, edge_index)))
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
    
    def loss(self,z,g_data):
        return self.recon_loss(z,g_data.edge_index) + (1 / g_data.num_nodes) * self.kl_loss()
    
    def metrics(self,z,g_data):
        neg_edge_index = negative_sampling(g_data.edge_index, z.size(0))
        
        pos_y = z.new_ones(g_data.edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, g_data.edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()    

        return roc_auc_score(y, pred), average_precision_score(y, pred),f1_score(y, pred, average="micro"),accuracy_score(y, pred)