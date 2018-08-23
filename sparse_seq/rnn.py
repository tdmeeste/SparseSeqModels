import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter





class SparseLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, bidirectional=False, wdrop=0, blocks=0, density=0, reduce_in=-1, reduce_out=-1):
        """
        simple prototype of Sparse LSTM
        assumed:
            num_layers == 1 (multiple SparseLSTM's can be stacked though),
            bias == True,
            batch_first == False,
            dropout == 0  (instead: wdrop on hidden-hidden params), as proposed in Merity's awd_lstm_lm work
        """
        super(SparseLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.wdrop = wdrop

        if blocks > 0 and density >= 1./blocks and density <= 1:
            self.mode = 'given_blocks_density'
            self.k = blocks
            self.delta = density
        elif reduce_in > 0 and reduce_in <= 1 and reduce_out > 0 and reduce_out <= 1:
            self.mode = 'given_reduction'
            self.reduce_in = reduce_in
            self.reduce_out = reduce_out
            # determine lowest possible k (number of LSTM components)
            self.k = self._calc_minimal_k(self.reduce_in, self.reduce_out)
            # determine corresponding delta (input reduction density)
            self.delta = self._calc_input_density(self.k, self.reduce_in, self.reduce_out)
        else:
            raise IOError('incorrect input of blocks and density, or reduce_in and reduce_out')

        #determine indices for components inputs
        self.input_indices = self._calc_segments(self.input_size, self.k, self.delta)
        self.input_sizes = [self.input_indices[c][1]-self.input_indices[c][0] for c in range(self.k)]

        #determine indices for components outputs
        self.output_indices = self._calc_segments(self.hidden_size, self.k, 1./self.k)
        self.output_sizes = [self.output_indices[c][1]-self.output_indices[c][0] for c in range(self.k)]

        #initialize LSTMS
        self.LSTMs = [nn.LSTM(self.input_sizes[c], self.output_sizes[c], 1,
                              bidirectional=bidirectional, bias=True, batch_first=False, dropout=0)
                      for c in range(self.k)]
        #if wdrop > 0: wrap WeightDrop around LSTMs
        if self.wdrop > 0:
            self.LSTMs = [WeightDrop(LSTM, ['weight_hh_l0'], dropout=wdrop) for LSTM in self.LSTMs]

        self.LSTMs = nn.ModuleList(self.LSTMs)


    def info(self, pre='\t'):
        res = ''
        res += pre+'input size {}, hidden size {}\n'.format(self.input_size, self.hidden_size)
        res += pre+'wdrop {}, bidirectional {}\n'.format(self.wdrop, self.bidirectional)
        if self.mode == 'given_blocks_density':
            res += pre+'components k = {}, input density delta = {} (1/k = {})\n'.format(self.k, self.delta, 1./self.k)
        elif self.mode == 'given_reduction':
            res += pre+'reduce_in {}, reduce_out {}\n'.format(self.reduce_in, self.reduce_out)
            res += pre+'minimal required components k = {}, input density delta = {} (1/k = {})\n'.format(self.k, self.delta, 1./self.k)
        res += pre+'resulting input segment lengths: {}\n'.format(self.input_sizes)
        res += pre+'resulting hidden segment lengths: {}\n'.format(self.output_sizes)
        res += pre+'total sparseness (compared to dense with same dimensions): {}\n'.format(self._calc_sparseness())
        return res

    def info_dict(self):
        d = {'ninp': self.input_size, 'nhid': self.hidden_size, 'wdrop': self.wdrop, 'density': self.delta}
        if self.mode == 'given_blocks_density':
            pass
        elif self.mode == 'given_reduction':
            d = {**d, 'reduce_in': self.reduce_in, 'reduce_out': self.reduce_out}
            if self.k > 1:
                d = {**d, 'equiv_dense_lstm': self._calc_info_equivalent_dense_LSTM()}
        d = {**d, 'sizes_input': self.input_sizes, 'sizes_output': self.output_sizes, 'sparseness': self._calc_sparseness()}
        d = {**d, 'params': '%.2fM'%(self._calc_nparam()/1000000)}
        #d = {**d, 'info': self.info()}
        return d

    def _calc_nparam(self):
        np = 0
        for p in self.parameters():
            np += p.nelement()
        return np


    def forward(self, input, hidden):
        #apply all individual LSTMS
        #TODO: more efficient, parallel over multiple devices if available
        outputs = []
        hiddens = []

        for c, LSTM in enumerate(self.LSTMs):
            input_c = input[:, :, self.input_indices[c][0]: self.input_indices[c][1]]
            hidden_c = (hidden[0][:, :, self.output_indices[c][0]: self.output_indices[c][1]].contiguous(),
                        hidden[1][:, :, self.output_indices[c][0]: self.output_indices[c][1]].contiguous())
            output_c, new_hidden_c = LSTM(input_c, hidden_c)
            outputs.append(output_c)
            hiddens.append(new_hidden_c)

        #aggregate outputs and states
        output = torch.cat(outputs, dim=2)
        new_hidden = tuple(zip(*hiddens))
        new_hidden = (torch.cat(new_hidden[0], dim=2), torch.cat(new_hidden[1], dim=2))

        return output, new_hidden


    def _calc_info_equivalent_dense_LSTM(self):
        if self.mode == 'given_reduction':
            res = {'ninp': int(np.round(self.input_size * self.reduce_in)),
                   'nhid': int(np.round(self.hidden_size * self.reduce_out))}
            res['params'] = '%.2fM'%(4*(res['ninp']*res['nhid'] + res['nhid']*(2+res['nhid']))/1000000)
            return res
        else:
            return 'n/a'



    def _calc_minimal_k(self, reduce_in, reduce_out):
        hi, ho = self.input_size, self.hidden_size
        ai, ao = reduce_in, reduce_out
        denom = ai*ao*hi + ao*ao*ho + 2*ao - 2
        lowerbound = (hi + ho) / denom
        k = int(np.ceil(lowerbound))
        if denom <= 0 or lowerbound > ho or k >= self.input_size:
            raise IOError("Reduction factors too small for creating meaningful SparseLSTM")
        return k


    def _calc_input_density(self, k, reduce_in, reduce_out):
        hi, ho = self.input_size, self.hidden_size
        ai, ao = reduce_in, reduce_out
        input_density = (ai*ao*hi + ao*ao*ho + 2*ao - ho/k - 2)/hi
        if input_density > 1:
            input_density = 1. #required reduction too little to reach feasible configuration, even when full overlap
        if input_density < 1/k:
            raise IOError("Reduction factors too small for creating meaningful SparseLSTM")
        return input_density


    def _calc_sparseness(self):
        hi, ho = self.input_size, self.hidden_size
        current_params = 0
        for p in self.parameters():
            current_params += p.numel()
        #test_current_params = self.k * 4 * (self.delta*hi*ho/self.k + ho*ho/(self.k*self.k) + 2*ho/self.k)
        dense_params = 4 * (hi*ho + ho*ho + 2*ho)
        return current_params/dense_params


    def _calc_segments(self, total, k, delta):
        if k == 1 and delta == 1:
            return [(0, total)]
        elif k > 1 and delta >= 1/k:
            overlap = total * (k * delta - 1.) / (k - 1)
            begin_ids = [i*(total*delta - overlap) for i in range(k)]
            end_ids = [b + total*delta for b in begin_ids]
            #make integer
            begin_ids = [int(b) for b in begin_ids]
            end_ids = [int(e+1.e-7) for e in end_ids] # TODO, now hack to cure rare issue in floats
            end_ids[-1] = total
            #sanity checks
            assert np.min([e - b for e, b in zip(end_ids, begin_ids)]) > 0 #make sure no width-0 intervals
            all_segments = [list(range(b, e)) for b, e in zip(begin_ids, end_ids)]
            assert len(set([i for segm in all_segments for i in segm])) == total #make sure each index used

            return list(zip(begin_ids, end_ids))
        else:
            raise IOError("Incorrect blocks and coverage parameters to determine plausible segments")





class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def _dummy(self):
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self._dummy

        for name_w in self.weights:
            #print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.variational:
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda: mask = mask.cuda()
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
            else:
                w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)

