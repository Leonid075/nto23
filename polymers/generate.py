from rdkit.Chem import AllChem as Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
import numpy as np
from joblib import load
from poly_hgraph import *
import torch
from random import shuffle
from random import randint

calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
    [x[0] for x in Descriptors._descList])


def smiles2desc(smiles: str) -> np.ndarray:
    return np.asarray(calculator.CalcDescriptors(Chem.MolFromSmiles(smiles)))


logp_model = load('./logp.joblib')


def calc_logp(smiles: str) -> float:
    return float(logp_model.predict([smiles2desc(smiles)])[0])


def get_penalty(y_pred: float, min_target=2.3, max_target=2.7) -> float:
    if y_pred > max_target:
        return abs(y_pred - max_target)
    elif y_pred < min_target:
        return abs(y_pred - min_target)
    else:
        return 0


iter_ = 0
old_smiles = []
old_tens = []
new_smiles = []

vocab = [x.strip("\r\n ").split() for x in open('./vocab.txt')]
MolGraph.load_fragments([x[0] for x in vocab if eval(x[-1])])
vocab = PairVocab([(x, y) for x, y, _ in vocab])
model = HierVAE(vocab).cuda()
model.load_state_dict(torch.load('./model30.pt'))
model.eval()


def gen(iter: int, n=20) -> str:
    global old_tens
    global old_smiles
    global new_smiles
    bests = set()
    for j in range(5):
        if iter <= 2:
            old_ind = 0
        else:
            old_ind = randint(0, len(old_tens)-1)
        new_ind = randint(0, len(old_tens)-1)
        while old_ind == new_ind:
            new_ind = randint(0, len(old_tens)-1)

        try:
            new_gens = list(set(model.sample_new(old_tens[old_ind],
                                                 old_tens[new_ind], n)))
        except:
            continue

        for i in new_gens:
            if i not in old_smiles and i not in new_smiles and get_penalty(calc_logp(i)) == 0:
                bests.add(i)
        if len(bests) > 3:
            return list(bests)[randint(0, len(bests)-1)]
        elif j == 4 and len(bests) > 0:
            return list(bests)[0]
        elif j == 4:
            return sorted(new_gens, key=lambda x: get_penalty(calc_logp(x)))[0]
    return None


def search_step(smiles_pop: [str]) -> [str]:
    global iter_
    global old_tens
    global old_smiles
    global new_smiles
    if iter_ == 0:
        try:
            old_smiles.extend(model.sample(len(smiles_pop)//20 + 3))
        except:
            print('', end='')
        old_smiles.extend(smiles_pop)
        shuffle(old_smiles)
        for i in old_smiles:
            try:
                old_tens.append(model.to_tensor([i], vocab)[0])
            except:
                continue

    if iter_ >= len(old_smiles):
        return smiles_pop

    new = gen(iter_)
    if new is not None:
        new_smiles.append(new)
        smiles_pop[iter_] = new
    iter_ += 1
    return smiles_pop
