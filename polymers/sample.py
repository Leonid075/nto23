import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import math, random, sys
import numpy as np
import argparse
from tqdm import tqdm
import pandas as pd

from poly_hgraph import *
import rdkit

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--vocab', required=True)
parser.add_argument('--atom_vocab', default=common_atom_vocab)
parser.add_argument('--model', required=True)

parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--nsample', type=int, default=10)
#python3 sample.py --vocab ./vocab.txt --model ./ckpt/tmp/model.30
parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--hidden_size', type=int, default=250)
parser.add_argument('--embed_size', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--latent_size', type=int, default=24)
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=20)
parser.add_argument('--diterT', type=int, default=1)
parser.add_argument('--diterG', type=int, default=5)
parser.add_argument('--dropout', type=float, default=0.0)

args = parser.parse_args()

vocab = [x.strip("\r\n ").split() for x in open(args.vocab)] 
MolGraph.load_fragments([x[0] for x in vocab if eval(x[-1])])
args.vocab = PairVocab([(x,y) for x,y,_ in vocab])

model = HierVAE(args).cuda()

model.load_state_dict(torch.load(args.model))
model.eval()

torch.manual_seed(args.seed)
random.seed(args.seed)

# with torch.no_grad():
#     for _ in tqdm(range(args.nsample // args.batch_size)):
#         smiles_list = model.sample(args.batch_size)
#         for _,smiles in enumerate(smiles_list):
#             print(smiles)

# with torch.no_grad():
#     smiles_list = model.sample_new('CCCC(=O)OCC(Cc1cncn1C)C(CC)C(=O)OCc1ccccc1', 'O=[N+]([O-])c1ccc(Oc2ccc(Cl)cc2Cl)cc1', 5, args.vocab)
#     for _,smiles in enumerate(smiles_list):
#         print(smiles)
smiles = pd.read_csv('./train.csv')['SMILES'].to_list()

mean = np.array([0]*24, dtype=np.float64)
min = np.array([0]*24, dtype=np.float64)
max = np.array([0]*24, dtype=np.float64)

with torch.no_grad():
    for i in range(len(smiles) // args.batch_size):
        tens = model.get_tens(smiles[args.batch_size*i : args.batch_size*i+args.batch_size], args.vocab).T
        mean += [np.mean(i) for i in tens]
        min += [np.min(i) for i in tens]
        max += [np.max(i) for i in tens]

print(list(mean/(len(smiles) // args.batch_size)))
print(list(min/(len(smiles) // args.batch_size)))
print(list(max/(len(smiles) // args.batch_size)))