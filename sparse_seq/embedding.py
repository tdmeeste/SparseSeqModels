import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable

#from embed_regularize import embedded_dropout
#from locked_dropout import LockedDropout

from scipy.optimize import bisect
import numpy as np

class RawEmbedding(nn.Module):
    """
    based on nn.Embedding
    but extended for applying F.embedding on larger embedding matrix, pre-padded with zeros.
    num_embeddings: total number of embeddings (prepad + actual)
    num_prepad: number of pre-padded zero embeddings
    for now: no measures to avoid storing these zeros.

    for simplicity: eliminated padding_idx functionality (but keep field for use
    with embedded_dropout)

    (Note: requires batch_first = False for variational dropouti)
    """
    def __init__(self, num_embeddings, embedding_dim, num_prepad=0, dropoute=0.1, dropouti=0.5,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 sparse=False, gpu=True):
        super(RawEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_prepad = num_prepad
        self.padding_idx = None
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.weight = Parameter(torch.Tensor(num_embeddings-num_prepad, embedding_dim))
        self.sparse = sparse

        if num_prepad > 0:
            prepad_weight = torch.zeros(num_prepad, embedding_dim, requires_grad=False)
#            prepad_weight = torch.Tensor(num_prepad, embedding_dim, requires_grad=False).fill_(0.)
            self.prepad_weight = prepad_weight if not gpu else prepad_weight.cuda()
        else:
            self.prepad_weight = None

        #todo: make robust for non-cuda running
        self.dropoute = dropoute
        self.dropouti = dropouti

        self.init_weights(0.1)

    def init_weights(self, init_range):
        self.weight.data.uniform_(-init_range, init_range)

    def n_trainable_params(self):
        return torch.numel(self.weight.data)

    def get_total_weight(self):
        return self.weight if self.prepad_weight is None else torch.cat((self.prepad_weight, self.weight), dim=0)


    def forward(self, input, mask_dropoute=None):
        """
        calling the embedding: input should contain indices over total (pre-padded) range.
        if mask_dropoute (1d tensor) is given, it overrules self.dropoute
        """
        total_weight = self.get_total_weight()
        if self.training and (mask_dropoute is not None or self.dropoute > 0):
            if mask_dropoute is None:
                mask_dropoute = self.weight.data.new(self.num_embeddings, 1).bernoulli_(1 - self.dropoute) / (1 - self.dropoute)
            total_weight = total_weight * Variable(mask_dropoute.expand_as(total_weight))

        emb = F.embedding(
            input, total_weight, max_norm=self.max_norm,
            norm_type=self.norm_type, scale_grad_by_freq=self.scale_grad_by_freq, sparse=self.sparse)
        #drop random embedding dimensions
        if self.training and self.dropouti > 0:
            #emb.data size:  (seq_len, batch, hidden): dropouti drops, random dimensions
            # (different per instance, but consistent along seq len of each instance, i.e., variational dropout)
            mask_dropouti = emb.data.new(1, emb.size(1), emb.size(2)).bernoulli_(1 - self.dropouti) / (1 - self.dropouti)
            emb = emb * Variable(mask_dropouti, requires_grad=False).expand_as(emb)

        return emb, mask_dropoute  #return mask_dropoute for using same mask on other embedding tensor for same batch




    def __repr__(self):
        s = '{name}({num_embeddings}, {embedding_dim}'
        if self.num_prepad>0:
            s += ', num_prepad={num_prepad}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        if self.dropoute > 0:
            s += ', dropoute={dropoute}'
        if self.dropouti > 0:
            s += ', dropouti={dropouti}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)



class SimpleEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dropoute=0.1, dropouti=0.5,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False, has_decoder=False, gpu=True):
        super(SimpleEmbedding, self).__init__()
        #RawEmbedder with num_prepad = 0
        self.embedder = RawEmbedding(num_embeddings, embedding_dim,
                                              num_prepad=0, dropoute=dropoute, dropouti=dropouti,
                                              max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq,
                                              sparse=sparse, gpu=gpu)
        self.has_decoder = has_decoder
        if has_decoder:
            self.decoder_bias = Parameter(torch.Tensor(num_embeddings))
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.init_weights(0.1)

    def n_trainable_params(self):
        return torch.numel(self.embedder.weight.data)

    def init_weights(self, init_range=0.1):
        self.embedder.init_weights(init_range)
        if self.has_decoder:
            self.decoder_bias.data.fill_(0)

    def decode(self, hidden):
        """input is 2d tensor (batch*seq, hidden), output (batch*seq, num_embeddings)"""
        if not self.has_decoder:
            raise IOError

        decoded = F.linear(hidden, self.embedder.get_total_weight(), bias=self.decoder_bias)
        #todo: make more memory-efficient
        return decoded


    def forward(self, input):
        return self.embedder(input)[0] #only return embedded input




class SparseEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, blocks=1, density=1, dropoute=0.1, dropouti=0.5,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False, has_decoder=False, gpu=True):
        super(SparseEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.dropoute = dropoute
        self.dropouti = dropouti

        blocks = min(blocks, embedding_dim)
        assert density > 0 and density <= 1
        if blocks > 1:
            assert density >= 1./blocks
        #if density == 1.:
        #    assert blocks == 1   #avoid needless simulations in sweep over hyperparams.
        if blocks == 1:
            assert density == 1.

        self.intended_density = density
        self.start_ids, self.block_widths, self.empirical_density = calc_sparse_geometry(self.num_embeddings, self.embedding_dim, density, blocks)
        self.blocks = len(self.start_ids)
        self.embedders = nn.ModuleList([RawEmbedding(num_embeddings, self.block_widths[i], num_prepad=self.start_ids[i],
                                                     dropoute=dropoute, dropouti=dropouti,
                                                     max_norm=max_norm, norm_type=norm_type,
                                                     scale_grad_by_freq=scale_grad_by_freq, sparse=sparse, gpu=gpu)
                                        for i in range(self.blocks)])
        #print('composition of sparse embedding layer:')
        #for i, e in enumerate(self.embedders):
        #    print('block', i, 'size', e.weight.data.size())

        self.has_decoder = has_decoder
        if has_decoder:
            self.decoder_bias = Parameter(torch.Tensor(num_embeddings))
        self.init_weights(0.1)

    def init_weights(self, init_range):
        #make consistent with other modules in LM experiments
        for e in self.embedders:
            e.init_weights(init_range)
        if self.has_decoder:
            self.decoder_bias.data.fill_(0)

    def n_trainable_params(self):
        return np.sum([torch.numel(e.weight.data) for e in self.embedders])

    def forward(self, input):

        embs = []
        e0, mask_dropoute = self.embedders[0](input)
        embs.append(e0)

        for i in range(1, self.blocks):
            ei, _ = self.embedders[i](input, mask_dropoute)
            embs.append(ei)
        return torch.cat(embs, dim=2)

    def decode(self, hidden):
        """input is 2d tensor (batch*seq, hidden), output (batch*seq, num_embeddings)"""
        if not self.has_decoder:
            raise IOError

        start_idx = 0
        end_idx = self.block_widths[0]
        decoded = F.linear(hidden[:, start_idx:end_idx], self.embedders[0].get_total_weight(), bias=self.decoder_bias)
        for i in range(1, self.blocks):
            start_idx = end_idx
            end_idx = start_idx + self.block_widths[i]
            decoded += F.linear(hidden[:, start_idx:end_idx], self.embedders[i].get_total_weight(), bias=None)

        #todo: more memory-efficient?
        return decoded


    def __repr__(self):
        s = '{name}({num_embeddings}, {embedding_dim}'
        if self.dropoute > 0:
            s += ', dropoute={dropoute}'
        if self.dropouti > 0:
            s += ', dropouti={dropouti}'
        s += ', blocks={blocks}'
        s += ', intended_density={intended_density}'
        s += ', empirical_density={empirical_density}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)




def calc_sparse_geometry(vocab, dim, density, blocks):

    vocab_remaining = vocab
    dim_remaining = dim
    blocks_remaining = blocks
    density_remaining = density

    block_widths = []
    start_ids = []
    all_params = 0

    #block 0
    w = int(np.floor(dim/blocks)) #compensate rounding error on last layer
    block_widths.append(w)
    start_ids.append(0)
    all_params += w * vocab
    print('all_params after layer 0:  {}/{}'.format(all_params, np.floor(vocab * dim * density)))
    #block 1 to blocks-1
    if blocks > 2:
        for k in range(1, blocks-1):
            block_widths.append(w)
            if density_remaining > 0:
                #calculate reduction factor for next block
                alpha_k = reduction_factor(density_remaining, blocks_remaining)
                print('alpha_{}: {}'.format(k, alpha_k))

                #update vocab_remaining, blocks_remaining, density_remaining
                vocab_reduction = int(np.ceil(vocab_remaining * (1 - alpha_k)))
                vocab_remaining -= vocab_reduction
                dim_remaining -= w
                blocks_remaining -= 1

                #update results
                start_ids.append(start_ids[-1] + vocab_reduction)
                if vocab_remaining * dim_remaining > 0 and vocab * dim * density - all_params > 0:
                    density_remaining = (vocab * dim * density - all_params) / (vocab_remaining * dim_remaining)
                    density_remaining = min(1., density_remaining)
                    all_params += vocab_remaining * w
                    print('density_remaining', density_remaining)
                else:
                    density_remaining = 0

            else:
                start_ids.append(vocab)

    if blocks > 1:
        #final block
        w_last = int(dim - np.sum(block_widths))
        #print('dim:', dim, 'sum', np.sum(block_widths), 'w_last', w_last)
        block_widths.append(w_last)
        remaining_params = max(0, np.floor(vocab * dim * density) - all_params)
        remaining_vocab_last = int(np.round(remaining_params / w_last))
        start_ids.append(vocab - remaining_vocab_last)
        all_params += w_last * remaining_vocab_last

    resulting_density = all_params / (vocab * dim)
    return list(start_ids), list(block_widths), float(resulting_density)



def reduction_factor(density, blocks):
    """
    calculate reduction factor alpha, for which
    density = 1/blocks * sum(alpha^k, k=0..blocks-1)
    """
    def delta(a):
        res = 1
        for b in range(1, blocks):
            res += a**b
        return res/blocks - density
    alpha = bisect(delta, 0, 1, xtol=1e-6)
    return alpha








if __name__=="__main__":

    print('\ntest RawEmbedding and SimpleEmbedding\n')
    V = 10
    d = 4

    e = SimpleEmbedding(V, d, dropoute=0.3, dropouti=0.5)
    for n,p in e.named_parameters():
        print('test parameter', n,type(p))

    input = torch.autograd.Variable(torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]))
    print('weight without dropout:', e.embedder.weight)
    print('trainable params: ', e.n_trainable_params())
    print('embedded input with dropout:', e(input))
    print('embedded input with dropout:', e(input))
    print('embedded input with dropout:', e(input))


    print('\ntest calculation of reduction factor \n')
    print('density = 0.75, blocks = 2 :', reduction_factor(0.75, 2))
    print('density = 2/20, blocks = 20 :', reduction_factor(0.1, 20))
    print('density = 0.5, blocks = 5 :', reduction_factor(0.5, 5))
    print('density = 1, blocks = 1 :', reduction_factor(1, 1))

    print('\ntest SparseEmbedding\n')
    V = 15
    d = 9
    blocks = 5
    density = 0.5

    e = SparseEmbedding(V, d, blocks=blocks, density=density, dropoute=0.1, dropouti=0.2, has_decoder=True, gpu=False)
    print(e)
    for n, p in e.named_parameters():
        print('test parameter', n, type(p), p.size())
    print('trainable params: ', e.n_trainable_params())
    print('empirical density: ', e.n_trainable_params()/(V*d))

    input = torch.autograd.Variable(torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]))

    print('embedded input with dropout:', e(input))
    #print('embedded input with dropout:', e(input))
    #print('embedded input with dropout:', e(input))

    batch = 3
    scores = torch.autograd.Variable(torch.Tensor(batch, d).uniform_())
    print(e.decode(scores))
