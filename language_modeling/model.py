import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop
from sparse_seq.embedding import SparseEmbedding
from sparse_seq.rnn import SparseLSTM


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nhidlast, nlayers,
                 dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, 
                 ldropout=0.5, n_experts=10, emblocks=1, emdensity=1., **sparseconfig):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.encoder = SparseEmbedding(ntoken, ninp, blocks=emblocks, density=emdensity, dropoute=dropoute, dropouti=dropouti, has_decoder=True)
        self.emblocks = emblocks
        self.emdensity = emdensity

        #sparse config
        if not 'sparse_mode' in sparseconfig:
            self.sparse_mode = 'direct'
        elif sparseconfig['sparse_mode'] in ['direct', 'sparse_hidden', 'sparse_emb', 'sparse_last', 'sparse_all']:
            self.sparse_mode = sparseconfig['sparse_mode']
        else:
            raise IOError

        mode = self.sparse_mode
        if mode == 'direct': #original implementation, dense LSTMs; ignore sparse rnn settings.
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid,
                                       nhid if l != nlayers - 1 else nhidlast, 1, dropout=0)
                         for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]

        elif mode == 'sparse_hidden':
            sparse_fract = sparseconfig['sparse_fract']
            if nlayers == 1:
                reduction = [(1, 1)]
            else:
                reduction = [(1, sparse_fract)]
                for l in range(1, nlayers-1):
                    reduction.append((sparse_fract, sparse_fract))
                reduction.append((sparse_fract, 1))

            self.rnns = [SparseLSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else nhidlast,
                                    reduce_in=reduction[l][0], reduce_out=reduction[l][1], wdrop=wdrop) for l in
                         range(nlayers)]
        elif mode == 'sparse_emb': #sparse embedding layer, and compensate in outer rnn layers to have same number of params in rnns
            assert nlayers >= 2
            sparse_fract = sparseconfig['sparse_fract']
            reduction = [(sparse_fract, 1)]
            wdrops = [wdrop]
            for l in range(1, nlayers-1):
                reduction.append((1, 1))
                wdrops.append(wdrop) #watch out, use same wdrop everywhere
            reduction.append((1, sparse_fract))
            wdrops.append(wdrop)

            self.rnns = [SparseLSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else nhidlast,
                                    reduce_in=reduction[l][0], reduce_out=reduction[l][1], wdrop=wdrops[l]) for l in
                         range(nlayers)]


        elif mode == 'sparse_all': #sparse embedding layer, and all rnn layers sparse with same density
            assert nlayers >= 2
            sparse_fract = sparseconfig['sparse_fract']

            self.rnns = [SparseLSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else nhidlast,
                                    reduce_in=sparse_fract, reduce_out=sparse_fract, wdrop=wdrop) for l in
                         range(nlayers)]


        #wdrop integrated in SparseLSTM instead of previously added at this level
        self.rnns = torch.nn.ModuleList(self.rnns)

        self.prior = nn.Linear(nhidlast, n_experts, bias=False)

        self.needs_output_trf = nhidlast != n_experts * ninp
        if self.needs_output_trf:
            self.output_trf = nn.Sequential(nn.Linear(nhidlast, n_experts*ninp), nn.Tanh())

        # always tie weights
        self.decoder = self.encoder.decode

        self.init_weights()

        self.ninp = ninp
        self.nhid = nhid
        self.nhidlast = nhidlast
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.ldropout = ldropout
        self.dropoutl = ldropout
        self.n_experts = n_experts
        self.ntoken = ntoken


    def init_weights(self):
        initrange = 0.1
        self.encoder.init_weights(initrange)

    def info_dict(self):
        if self.sparse_mode == 'direct':
            d = {'LSTM': {'layer%d'%i: str(rnn) for i, rnn in enumerate(self.rnns)}}
        else:
            d = {'SparseLSTM': {'layer%d'%i: rnn.info_dict() for i, rnn in enumerate(self.rnns)}}
        return d


    def forward(self, input, hidden, return_h=False, return_prob=False):
        #input:  (seq_length, batch_size)
        batch_size = input.size(1)
        emb = self.encoder(input)

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        if self.needs_output_trf:
            latent = self.output_trf(output)
            latent = self.lockdrop(latent, self.dropoutl)
        else:
            latent = output
        logit = self.decoder(latent.view(-1, self.ninp))

        # keep Mixture-of-Softmaxes output layer;
        # so far only tested sparse model with n_experts = 1
        prior_logit = self.prior(output).contiguous().view(-1, self.n_experts)
        prior = nn.functional.softmax(prior_logit, 1)

        prob = nn.functional.softmax(logit.view(-1, self.ntoken), 1).view(-1, self.n_experts, self.ntoken)
        prob = (prob * prior.unsqueeze(2).expand_as(prob)).sum(1)

        if return_prob:
            model_output = prob
        else:
            log_prob = torch.log(prob.add_(1e-8)) #original code from https://github.com/zihangdai/mos
            model_output = log_prob

        model_output = model_output.view(-1, batch_size, self.ntoken)

        if return_h:
            return model_output, hidden, raw_outputs, outputs
        return model_output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return [(Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.nhidlast).zero_()),
                 Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.nhidlast).zero_()))
                for l in range(self.nlayers)]

