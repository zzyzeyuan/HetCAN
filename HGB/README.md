# HetCAN
We provide the implementation of HetCAN based on the offical PyTorch implementation of HGB (https://github.com/THUDM/HGB).

## File descriptions
- dataset/: the original data of six benchmark dataset.
- run_new.py: multi-class node classification of HetCAN.
- run_multi.py: multi-label node classification of HetCAN (for IMDB).
- model.py: the implementation of HetCAN based on Pytorch.
- conv.py: the implementation of Type-aware Layer and Dimension-aware Layer.
- utils/: contains the tools we used.

## Datasets
You can download data as follows (from HGB)
- https://drive.google.com/drive/folders/10-pf2ADCjq_kpJKFHHLHxr_czNNCJ3aX?usp=sharing 


## Experiments
For DBLP: 
```
python run_new.py --dataset DBLP --feats-type 2 --hidden-dim 128 --num-heads 4 --num-layers 4 --weight-decay 1e-5 --edge-feats 128 --alpha 0 --device 0
```
For ACM:
```
python run_new.py --dataset ACM --feats-type 2 --hidden-dim 64 --num-heads 4 --num-layers 2 --lr 0.0005 --dropout 0.5 --attn-dropout 0.5 --weight-decay 1e-5 --edge-feats 128 --alpha 0.1 --device 0
```
For IMDB:
```
python run_multi.py --dataset IMDB --feats-type 0  --hidden-dim 256 --num-heads 8 --num-layers 3 --lr 0.0002 --dropout 0.3 --attn-dropout 0.5 --weight-decay 1e-05 --slope 0.05 --edge-feats 128 --alpha 0.1 --device 0
```
For Freebase:
```
python run_new.py --dataset Freebase --feats-type 2 --dropout 0 --attn-dropout 0.2 --lr 0.0002 --num-heads 4 --num_layers 3 --dim_layers 2 --weight_decay 0.0001 --alpha 0.2 --device 0 
```

## Environment
* CUDA version 11.4
* python 3.9.16
* torch 1.12.1+cu113
* dgl 1.0.0+cu113
* networkx 2.3
* scikit-learn 1.1.1
* scipy 1.9.3