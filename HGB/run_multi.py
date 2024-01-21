import sys
sys.path.append('../../')
import time
import random
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
from utils.pytorchtools import EarlyStopping
from utils.data import load_data
#from utils.tools import index_generator, evaluate_results_nc, parse_minibatch
from model import HetCAN
import dgl

def set_seed(seed=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) 
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False

def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)

def run_IMDB(args):
    print('========>>>>>> LOADING AND PROCESSING DATA')
    st = time.time()
    if not os.path.exists('checkpoint/'):
        os.makedirs('checkpoint/')
    feats_type = args.feats_type
    features_list, adjM, labels, train_val_test_idx, dl = load_data(args.dataset, args.seed)
    device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')
    features_list = [mat2tensor(features).to(device) for features in features_list]
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []#[features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    
    labels = torch.FloatTensor(labels).to(device)
    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)

    node_cnt = [features.shape[0] for features in features_list]
    node_type = [i for i, z in zip(range(len(node_cnt)), node_cnt) for x in range(z)]

    filename = 'etype/'+args.dataset+'_e_type_seed_'+str(args.seed)+'.pt'
    need_compute = True
    if os.path.exists(filename):
        e_type = torch.load(filename)
        need_compute = False
    if need_compute:
        edge2type = {}
        for k in dl.links['data']:
            for u,v in zip(*dl.links['data'][k].nonzero()):
                edge2type[(u,v)] = k
        for i in range(dl.nodes['total']):
            edge2type[(i,i)] = len(dl.links['count'])

    g = dgl.from_scipy(adjM)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)

    if need_compute:
        print('========>>>>>> COMPUTE EDGE FEAT')
        e_type = []
        for u, v in zip(*g.edges()):
            u = u.cpu().item()
            v = v.cpu().item()
            e_type.append(edge2type[(u,v)])
        
        torch.save(e_type, args.dataset+'_e_type_seed_'+str(args.seed)+'.pt')

    e_type = torch.tensor(e_type, dtype=torch.long).to(device)

    num_etypes = len(dl.links['count'])*2+1
    num_ntypes = len(dl.nodes['count'])
    num_classes = dl.labels_train['num_classes']
    print('========>>>>>> LOADING AND PROCESSING DATA FINISHED')

    micro_f1 = torch.zeros(args.repeat)
    macro_f1 = torch.zeros(args.repeat)

    for i in range(args.repeat):
        set_seed(args.seed + i)
        loss = nn.BCELoss()
        heads = [args.num_heads] * args.num_layers + [1]
        net = HetCAN(g, args.edge_feats, num_etypes, num_ntypes, in_dims, args.hidden_dim, args.ffn_dim, num_classes, args.num_layers, args.dim_layers, args.num_blocks, heads, F.elu, args.dropout, args.attn_dropout, args.slope, True, args.alpha)
        net.to(device)
        # print('========>>>>>> TOTAL PARAMS:', sum(p.numel() for p in net.parameters()))
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        # training loop
        early_stopping = EarlyStopping(patience=args.patience, verbose=False, save_path='checkpoint/checkpoint_{}_{}.pt'.format(args.dataset, args.num_layers))
        for epoch in range(args.epoch):
            t_start = time.time()
            # training
            net.train()

            logits = net(features_list, e_type, node_type)
            logp = torch.sigmoid(logits)
            train_loss = loss(logp[train_idx], labels[train_idx])
            # autograd
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            t_end = time.time()

            # print training info
            # print('Epoch {:05d} | Train_Loss: {:.4f} | Time: {:.4f}'.format(epoch, train_loss.item(), t_end-t_start))

            t_start = time.time()
            # validation
            net.eval()
            with torch.no_grad():
                logits = net(features_list, e_type, node_type)
                val_logits = logits[val_idx]
                logp = torch.sigmoid(logits)
                val_loss = loss(logp[val_idx], labels[val_idx])
                pred=(val_logits.cpu().numpy() > 0).astype(int)
                print(dl.evaluate_valid(pred, dl.labels_train['data'][val_idx]))
            t_end = time.time()
            scheduler.step(val_loss)
            # print validation info
            print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(epoch, val_loss.item(), t_end - t_start))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break
        
        # testing with evaluate_results_nc
        net.load_state_dict(torch.load('checkpoint/checkpoint_{}_{}.pt'.format(args.dataset, args.num_layers)))
        net.eval()
        with torch.no_grad():
            logits = net(features_list, e_type, node_type)
            test_logits = logits[test_idx]
            pred = (test_logits.cpu().numpy() > args.threshold ).astype(int)

            if args.mode == 1:
                dl.gen_file_for_evaluate(test_idx=test_idx, label=pred, file_name=f"{args.dataset}_{i+1}.txt")
            else:
                result=dl.evaluate(pred)
                print(result)
                micro_f1[i] = result['micro-f1']
                macro_f1[i] = result['macro-f1']
            
    print(args)
    print('Micro-f1: %.2f±%.2f' % (micro_f1.mean().item()*100, micro_f1.std().item()*100))
    print('Macro-f1: %.2f±%.2f' % (macro_f1.mean().item()*100, macro_f1.std().item()*100))
    et = time.time()
    print('Total time: %.2fs'%(et-st))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
    ap.add_argument('--feats-type', type=int, default=2,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2;' +
                        '4 - only term features (id vec for others);' + 
                        '5 - only term features (zero vec for others).')
    ap.add_argument('--dataset', type=str, default='IMDB', help='IMDB')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--ffn-dim', type=int, default=16)
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--epoch', type=int, default=500, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=30, help='Patience.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--num-layers', type=int, default=2)
    ap.add_argument('--dim-layers', type=int, default=2)
    ap.add_argument('--num-blocks', type=int, default=1, help="the number of HetCAN blocks")
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--attn-dropout', type=float, default=0.5)
    ap.add_argument('--weight-decay', type=float, default=0)
    ap.add_argument('--slope', type=float, default=0.05)
    ap.add_argument('--edge-feats', type=int, default=64, help='edge embedding dimension')
    ap.add_argument('--threshold', type=float, default=0)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--mode', type=int, default=0)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--device', type=int, default=0)

    args = ap.parse_args()
    run_IMDB(args)