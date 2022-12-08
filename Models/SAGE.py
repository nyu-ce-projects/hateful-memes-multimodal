import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv,NeighborSampler
import tqdm

# class SAGE(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, num_layers,training):
#         super(SAGE, self).__init__()
#         self.training = training
#         self.num_layers = num_layers
#         self.convs = torch.nn.ModuleList()
        
#         for i in range(num_layers):
#             in_channels = in_channels if i == 0 else hidden_channels
#             self.convs.append(SAGEConv(in_channels, hidden_channels))

#     def forward(self, x, adjs):
        
#         for i, (edge_index, _, size) in enumerate(adjs):
#             x_target = x[:size[1]]  # Target nodes are always placed first.
#             x = self.convs[i]((x, x_target), edge_index)
#             if i != self.num_layers - 1:
#                 x = x.relu()
#                 x = F.dropout(x, p=0.5, training=self.training)
#         return x

#     def full_forward(self, x, edge_index):
#         for i, conv in enumerate(self.convs):
#             x = conv(x, edge_index)
#             if i != self.num_layers - 1:
#                 x = x.relu()
#                 x = F.dropout(x, p=0.5, training=self.training)
#         return x


class GraphSAGE(torch.nn.Module):
    ##GraphSAGE
    def __init__(self, dim_in, dim_h, dim_out,training=True):
        super().__init__()
        self.training = training
        self.sage1 = SAGEConv(dim_in, dim_h*2)
        self.sage2 = SAGEConv(dim_h*2, dim_h)
        self.sage3 = SAGEConv(dim_h, dim_out)

    def forward(self, x, edge_index):
        h = self.sage1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.sage2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.sage3(h, edge_index)
        return h, F.log_softmax(h, dim=1)