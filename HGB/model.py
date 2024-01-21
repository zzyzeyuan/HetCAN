import torch
import torch.nn.functional as F
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GraphConv

import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv, GATv2Conv
from conv import MLPLayer, TALayer, DALayer


class HetCAN(nn.Module):
    def __init__(self, g, edge_dim, num_etypes, num_ntypes, 
                 in_dims, num_hidden, ffn_dim, num_classes, num_layers,
                 dim_layers, num_blocks, heads, activation, feat_drop, 
                 attn_drop, negative_slope, residual, alpha) -> None:
        super(HetCAN, self).__init__()

        self.num_blocks = num_blocks

        self.hetcan_block = HetCANLayer(g, edge_dim, num_etypes, num_ntypes, 
                                        num_hidden, ffn_dim, num_layers, 
                                        dim_layers, heads, activation, 
                                        feat_drop, attn_drop, negative_slope, 
                                        residual,  alpha)

        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        
        self.logits_layer = nn.Linear(num_hidden, num_classes, bias=False)
        self.epsilon = torch.FloatTensor([1e-12]).cuda()


    def forward(self, features_list, e_feat, node_type):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            # h.append(fc(feature))
            h.append(F.leaky_relu(fc(feature)))
        h = torch.cat(h, 0) # [n, dim]

        for i in range(self.num_blocks):

            if i < self.num_blocks - 1:
                h = self.hetcan_block(h, e_feat, node_type)
            else:
                h_final = self.hetcan_block(h, e_feat, node_type)
    

        logits = self.logits_layer(F.relu(h_final))
        logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
        # logits = F.normalize(logits)

        return logits
    


class HetCANLayer(nn.Module):
    def __init__(self,
                 g,
                 edge_dim,
                 num_etypes,
                 num_ntypes,
                 num_hidden,
                 ffn_dim,
                 num_layers,
                 dim_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 alpha):
        super(HetCANLayer, self).__init__()

        self.g = g
        self.type_encoder = nn.ModuleList()
        self.dim_encoder = nn.ModuleList()

        self.num_layers = num_layers
        self.dim_layers = dim_layers
        self.num_ntypes = num_ntypes
        self.activation = activation  # ELU
        self.nodetype_emb_layer = nn.Embedding(num_ntypes, num_hidden)
        self.fc_nodetype = nn.Linear(num_hidden, num_hidden, bias=False)
        self.mlp = MLPLayer(num_hidden, num_hidden, activation=F.relu, dropout=0)

        # input projection (no residual)
        self.type_encoder.append(TALayer(edge_dim, num_etypes, num_hidden, num_hidden,
                                         heads[0], feat_drop, attn_drop, negative_slope,
                                         False, self.activation, alpha=alpha))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.type_encoder.append(TALayer(edge_dim, num_etypes, num_hidden * heads[l-1], num_hidden, heads[l],
                                             feat_drop, attn_drop, negative_slope, residual, self.activation, alpha=alpha))
        # output projection
        self.type_encoder.append(TALayer(edge_dim, num_etypes, num_hidden * heads[-2], num_hidden, heads[-1],
                                         feat_drop, attn_drop, negative_slope, residual, None, alpha=alpha))

        for l in range(self.dim_layers):
            self.dim_encoder.append(DALayer(hidden_dim=1, ffn_dim=ffn_dim, dropout=feat_drop, attn_dropout=attn_drop, num_heads=1))

        self.final_proj = nn.Linear(2 * num_hidden, num_hidden, bias=False)


    def forward(self, x, e_feat, node_type):

        h = x
        ntype_feat = torch.arange(self.num_ntypes).to(h.device)
        ntype_feat = self.nodetype_emb_layer(ntype_feat)
        ntype_feat = self.fc_nodetype(ntype_feat)[node_type] # [n, dim]

        # ntype_feat = torch.ones(h.shape[0], h.shape[1]).to(h.device)
        h = h * ntype_feat # element-wise type-aware  * or + or cat
        res_attn = None
        
        for l in range(self.num_layers):
            h, res_attn = self.type_encoder[l](self.g, h, e_feat, res_attn=res_attn) # [n, nheads, dim]
            h = h.flatten(1) # [n, nheads*dim]
        h_feat, _ = self.type_encoder[-1](self.g, h, e_feat, res_attn=None) 
        h_feat = h_feat.squeeze() #  [n, dim]
        
        # dimension-aware encoder
        h_dim = h_feat * ntype_feat #  [n, dim]
        h_dim = h_dim.unsqueeze(2) # [n, dim, 1] : batch_size=n   len=dim   emb_dim = 1
        for l in range(self.dim_layers):
            h_dim = self.dim_encoder[l](h_dim, attn_bias=None)
        h_dim = h_dim.squeeze()
        h_dim = F.normalize(h_dim)
        h_feat = F.normalize(h_feat)
        h_final = torch.cat((h_feat, h_dim), dim=1)
        h_final = self.final_proj(h_final)


        return h_final





        
