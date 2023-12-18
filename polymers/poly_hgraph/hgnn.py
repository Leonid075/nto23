import torch
import torch.nn as nn
import random
import numpy as np
import rdkit.Chem as Chem
import torch.nn.functional as F
from poly_hgraph.mol_graph import MolGraph
from poly_hgraph.encoder import HierMPNEncoder
from poly_hgraph.decoder import HierMPNDecoder
from poly_hgraph.nnutils import *
from poly_hgraph import common_atom_vocab


def make_cuda(tensors):
    tree_tensors, graph_tensors = tensors
    def make_tensor(x): return x if type(
        x) is torch.Tensor else torch.tensor(x)
    tree_tensors = [make_tensor(x).cuda().long()
                    for x in tree_tensors[:-1]] + [tree_tensors[-1]]
    graph_tensors = [make_tensor(x).cuda().long()
                     for x in graph_tensors[:-1]] + [graph_tensors[-1]]
    return tree_tensors, graph_tensors


def to_numpy(tensors):
    def convert(x): return x.numpy() if type(x) is torch.Tensor else x
    a, b, c = tensors
    b = [convert(x) for x in b[0]], [convert(x) for x in b[1]]
    return a, b, c


class HierVAE(nn.Module):

    def __init__(self, vocab):
        super(HierVAE, self).__init__()
        self.encoder = HierMPNEncoder(vocab,
                                      common_atom_vocab, 'LSTM', 250, 250, 20, 20, 0.0)
        self.decoder = HierMPNDecoder(
            vocab, common_atom_vocab, 'LSTM', 250, 250, 24, 1, 5, 0.0)
        self.encoder.tie_embedding(self.decoder.hmpn)
        self.latent_size = 24

        self.R_mean = nn.Linear(250, 24)
        self.R_var = nn.Linear(250, 24)

    def rsample(self, z_vecs, W_mean, W_var, perturb=True):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs(W_var(z_vecs))
        kl_loss = -0.5 * \
            torch.sum(1.0 + z_log_var - z_mean * z_mean -
                      torch.exp(z_log_var)) / batch_size
        epsilon = torch.randn_like(z_mean).cuda()
        z_vecs = z_mean + torch.exp(z_log_var / 2) * \
            epsilon if perturb else z_mean
        return z_vecs, kl_loss

    def sample(self, n=2):
        if n > 20:
            n = 20
        max = [1.772879437717025, 1.855030971443008, 2.073098067291275, 2.131152494875845, 2.1404341610973487, 1.7731584349949518, 1.4827983760105106, 0.7373775978552136, 2.473108266780754, 1.7844772620764906, 1.4685786385723727, 2.06796244664756,
               1.7094518054941135, 2.323227072287657, 2.3739419024071857, 1.664892610782134, 2.4425831799038904, 1.4871361406150467, 1.9127010974234235, 1.2486777719490991, 1.2371797114759981, 1.8304306673621367, 1.775017478691791, 0.5313864848641809]
        min = [-2.0028503707988947, -1.7930755234194662, -1.8584644289795527, -1.8888170588948205, -1.5474797618651916, -2.2626166035512645, -2.030720619257084, -2.3553849513640626, -1.870459238011278, -1.4396958034597562, -2.345865563065829, -1.809549065056688, -
               1.9945887214912918, -1.7873842424763466, -1.2597777846610856, -1.8988269518515868, -1.3633014413720381, -1.897992891992978, -1.722914433670426, -1.6684396601392177, -2.4747395567999098, -1.8116595021230664, -1.8680846980196202, -2.5602389172705]
        tensors = []
        for i in range(n):
            tens = []
            for i in range(24):
                tens.append(random.uniform(min[i]-0.35, max[i]+0.35))
            tensors.append(tens)
        tens = torch.tensor(tensors).cuda()
        out = self.decoder.decode(
            (tens, tens, tens), greedy=True, max_decode_step=150)
        torch.cuda.empty_cache()
        return out

    def to_tensor(self, smiles, vcb):
        tens = to_numpy(MolGraph.tensorize(smiles, vcb, common_atom_vocab))[1]
        tree_tensors, graph_tensors = tens = make_cuda(tens)
        tens, tree_vecs, _, graph_vecs = self.encoder(
            tree_tensors, graph_tensors)
        tens, root_kl = self.rsample(tens, self.R_mean, self.R_var, True)
        tens = tens.cpu().detach().numpy()
        torch.cuda.empty_cache()
        return tens

    def sample_new(self, frm, to, n):
        z = []
        for t in np.linspace(0., 1., n)[1:-1]:
            z.append(frm * (1-t) + to * t)
        z = torch.tensor(np.asarray(z, dtype=np.float32)).cuda()
        o = self.decoder.decode((z, z, z), greedy=True, max_decode_step=150)
        torch.cuda.empty_cache()
        return o

    # def sample_new(self, frm, to, n, vcb):
    #     tens = to_numpy(MolGraph.tensorize([frm, to], vcb, common_atom_vocab))[1]

    #     tree_tensors, graph_tensors = tens = make_cuda(tens)
    #     tens, tree_vecs, _, graph_vecs = self.encoder(tree_tensors, graph_tensors)
    #     tens, root_kl = self.rsample(tens, self.R_mean, self.R_var, True)
    #     tens = tens.cpu().detach().numpy()

    #     z = np.zeros(([n] + list(tens.shape)))
    #     for i, t in enumerate(np.linspace(0., 1., n)[1:-1]):
    #         z[i] = tens[0] * (1-t) + tens[1] * t
    #     z = torch.tensor(z).to(torch.float).cuda()
    #     o = []
    #     for i in range(n):
    #         o.append(self.decoder.decode((z[i], z[i], z[i]), greedy=True, max_decode_step=150)[0])
    #     return o

    def reconstruct(self, batch):
        graphs, tensors, _ = batch
        tree_tensors, graph_tensors = tensors = make_cuda(tensors)
        root_vecs, tree_vecs, _, graph_vecs = self.encoder(
            tree_tensors, graph_tensors)

        root_vecs, root_kl = self.rsample(
            root_vecs, self.R_mean, self.R_var, perturb=False)
        return self.decoder.decode((root_vecs, root_vecs, root_vecs), greedy=True, max_decode_step=150)

    def forward(self, graphs, tensors, orders, beta, perturb_z=True):
        tree_tensors, graph_tensors = tensors = make_cuda(tensors)

        root_vecs, tree_vecs, _, graph_vecs = self.encoder(
            tree_tensors, graph_tensors)

        # graph_vecs = stack_pad_tensor( [graph_vecs[st : st + le] for st,le in graph_tensors[-1]] )
        # size = graph_vecs.new_tensor([le for _,le in graph_tensors[-1]])
        # graph_vecs = graph_vecs.sum(dim=1) / size.unsqueeze(-1)

        # tree_vecs = stack_pad_tensor( [tree_vecs[st : st + le] for st,le in tree_tensors[-1]] )
        # size = tree_vecs.new_tensor([le for _,le in tree_tensors[-1]])
        # tree_vecs = tree_vecs.sum(dim=1) / size.unsqueeze(-1)

        root_vecs, root_kl = self.rsample(
            root_vecs, self.R_mean, self.R_var, perturb_z)
        # tree_vecs, tree_kl = self.rsample(tree_vecs, self.T_mean, self.T_var, perturb_z)
        # graph_vecs, graph_kl = self.rsample(graph_vecs, self.G_mean, self.G_var, perturb_z)
        kl_div = root_kl  # + tree_kl + graph_kl

        # loss, wacc, iacc, tacc, sacc = self.decoder((root_vecs, tree_vecs, graph_vecs), graphs, tensors, orders)
        loss, wacc, iacc, tacc, sacc = self.decoder(
            (root_vecs, root_vecs, root_vecs), graphs, tensors, orders)
        return loss + beta * kl_div, kl_div.item(), wacc, iacc, tacc, sacc
