import dgl
import dgl.function as fn
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
from dgl import apply_each
from dgl.dataloading import DataLoader, NeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset
from model.conv import DALayer, TALayer
from dgl.nn.pytorch import HeteroGraphConv, GATConv

import os
import argparse

class HeteroGAT(nn.Module):
    def __init__(self, etypes, in_size, hid_size, out_size, 
                 num_layers, dim_layers, ntypes, n_heads=4, ffn_dim=16, dropout=0.5, attn_dropout=0.5):
        super().__init__()
        self.negative_slope = 0.01
        self.dim_layers = dim_layers
        self.ntypes = ntypes
        self.etypes = etypes
        self.ntype_emb_layer = nn.Embedding(ntypes, in_size)
        self.layers = nn.ModuleList()
        # type-aware encoder
        self.layers.append(
            HeteroGraphConv(
                {
                    etype: GATConv(in_size, hid_size // n_heads, n_heads)
                    for etype in etypes
                }
            )
        )
        for l in range(num_layers-1):
            self.layers.append(
                HeteroGraphConv(
                    {
                        etype: GATConv(hid_size, hid_size // n_heads, n_heads)
                        for etype in etypes
                    }
                )
            )
        
        self.dropout = nn.Dropout(0.5)

        # dim-aware encoder
        self.dim_encoder = nn.ModuleList()
        for l in range(dim_layers):
            self.dim_encoder.append(
                DALayer(hidden_dim=1, ffn_dim=ffn_dim, dropout=dropout, attn_dropout=attn_dropout, num_heads=1)
            )
        self.final_proj = nn.Linear(2 * hid_size, hid_size, bias=False)
        # self.logits_layer = nn.Linear(hid_size, num_classes, bias=False)
        self.linear = nn.Linear(2 * hid_size, out_size)  # Should be HeteroLinear

    def forward(self, blocks, x):
        """
        
        author, field_of_study, institution, paper
        """
        device = x['paper'].device
        h = x
        ntype_emb = torch.arange(self.ntypes).to(device)
        ntype_emb = self.ntype_emb_layer(ntype_emb)
        # print(x.keys())
        for i, nty in enumerate(x.keys()):
            tmp_ntype_emb = ntype_emb[i].repeat(x[str(nty)].shape[0], 1)
            x[str(nty)] = x[str(nty)] * tmp_ntype_emb


        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            # One thing is that h might return tensors with zero rows if the number of dst nodes
            # of one node type is 0.  x.view(x.shape[0], -1) wouldn't work in this case.
            h = apply_each(
                h, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2])
            )
            if l != len(self.layers) - 1:
                h = apply_each(h, F.relu)
                h = apply_each(h, self.dropout)
        
        h1 = h['paper'] # h_from_type_encoder
        
        h_dim = h1.unsqueeze(2) # torch.Size([1000, 256])
        
        for l in range(self.dim_layers):
            h_dim = self.dim_encoder[l](h_dim, attn_bias=None)
        h2 = h_dim.squeeze()
        h1, h2 = F.normalize(h1), F.normalize(h2)

        #output
        h_final = torch.cat((h1, h2), dim=1)
        return self.linear(h_final)

def evaluate(num_classes, model, dataloader, desc):
    preds = []
    labels = []
    with torch.no_grad():
        for input_nodes, output_nodes, blocks in tqdm.tqdm(
            dataloader, desc=desc
        ):
        # for input_nodes, output_nodes, blocks in dataloader:
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]["paper"][:, 0]
            y_hat = model(blocks, x)
            preds.append(y_hat.cpu())
            labels.append(y.cpu())
        preds = torch.cat(preds, 0)
        labels = torch.cat(labels, 0)
        acc = MF.accuracy(
            preds, labels, task="multiclass", num_classes=num_classes
        )
        return acc


def train(train_loader, val_loader, test_loader, num_classes, model, args):
    # loss function and optimizer
    loss_fcn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # training loop
    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            tqdm.tqdm(train_dataloader, desc="Train")
        ):
        # for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]["paper"][:, 0]
            y_hat = model(blocks, x)
            loss = loss_fcn(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        model.eval()
        val_acc = evaluate(num_classes, model, val_dataloader, "Val. ")
        test_acc = evaluate(num_classes, model, test_dataloader, "Test ")
        print(
            f"Epoch {epoch:05d} | Loss {total_loss/(it+1):.4f} | Validation Acc. {val_acc.item():.4f} | Test Acc. {test_acc.item():.4f}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='MAHAN')
    ap.add_argument('--hidden_dim', type=int, default=256, help="default 256")
    ap.add_argument('--num_heads', type=int, default=4)
    ap.add_argument('--epoch', type=int, default=50)
    ap.add_argument('--num_layers', type=int, default=3)
    ap.add_argument('--dim_layers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=5e-3)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--weight_decay', type=float, default=0)
    ap.add_argument('--slope', type=float, default=0.2)
    ap.add_argument('--batch_size', type=int, default=4096)
    ap.add_argument('--residual', type=bool, default=True, help='Default is False')
    ap.add_argument('--device', type=int, default=1)
    ap.add_argument('--ffn-dim', type=int, default=16)
    ap.add_argument('--attn_dropout', type=float, default=0.3)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--sample_num', type=int, default=10, help="5, 10.    sample_num of val/test is 2 * sample_num of training")
    args = ap.parse_args()
    os.makedirs('checkpoint', exist_ok=True)

    print(
        f"Training with DGL built-in HeteroGraphConv using GATConv as its convolution sub-modules"
    )
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")
    # load and preprocess dataset
    print("====> Loading data...")
    dataset = DglNodePropPredDataset("ogbn-mag")
    graph, labels = dataset[0]

    graph.ndata["label"] = labels
    # add reverse edges in "cites" relation, and add reverse edge types for the rest etypes
    graph = dgl.AddReverse()(graph)
    # precompute the author, topic, and institution features
    # graph.update_all(
    #     fn.copy_u("feat", "m"), fn.mean("m", "feat"), etype="rev_writes"
    # )
    # graph.update_all(
    #     fn.copy_u("feat", "m"), fn.mean("m", "feat"), etype="has_topic"
    # )
    # graph.update_all(
    #     fn.copy_u("feat", "m"), fn.mean("m", "feat"), etype="affiliated_with"
    # )
    feats = torch.load('./feat/feats.pt')
    graph.ndata['feat'] = feats

    # find train/val/test indexes
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = (
        split_idx["train"],
        split_idx["valid"],
        split_idx["test"],
    )
    train_idx = apply_each(train_idx, lambda x: x.to(device))
    val_idx = apply_each(val_idx, lambda x: x.to(device))
    test_idx = apply_each(test_idx, lambda x: x.to(device))

    # create RGAT model
    in_size = graph.ndata["feat"]["paper"].shape[1]
    num_classes = dataset.num_classes
    model = HeteroGAT(graph.etypes, in_size, hid_size=args.hidden_dim, out_size=num_classes, num_layers=args.num_layers,
                      dim_layers=args.dim_layers, ntypes=4, n_heads=args.num_heads, ffn_dim=args.ffn_dim,
                      dropout=args.dropout, attn_dropout=args.attn_dropout).to(device)

    train_fanouts = [args.sample_num] * args.num_layers
    val_fanouts = [args.sample_num*2] * args.num_layers
    # dataloader + model training + testing
    train_sampler = NeighborSampler(
        fanouts=train_fanouts,
        prefetch_node_feats={k: ["feat"] for k in graph.ntypes},
        prefetch_labels={"paper": ["label"]},
    )
    val_sampler = NeighborSampler(
        fanouts=val_fanouts,
        prefetch_node_feats={k: ["feat"] for k in graph.ntypes},
        prefetch_labels={"paper": ["label"]},
    )
    train_dataloader = DataLoader(
        graph,
        train_idx,
        train_sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=torch.cuda.is_available(),
    )
    val_dataloader = DataLoader(
        graph,
        val_idx,
        val_sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        use_uva=torch.cuda.is_available(),
    )
    test_dataloader = DataLoader(
        graph,
        test_idx,
        val_sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        use_uva=torch.cuda.is_available(),
    )
    print('====> Hyperparameters: \n', args)
    train(train_dataloader, val_dataloader, test_dataloader, num_classes, model, args)