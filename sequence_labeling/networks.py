import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence, pack_sequence
from torch.autograd import Variable
from sparse_seq.embedding import SparseEmbedding
from dropout import WeightDrop, LockedDropout, embedded_dropout


class LSTMTagger(nn.Module):
    def __init__(self, paras):
        super(LSTMTagger, self).__init__()

        self.paras = paras

        self.word_embeddings = nn.Embedding(self.paras.vocab, self.paras.emsize)
        self.word_embeddings = SparseEmbedding(self.paras.vocab,
                                               self.paras.emsize,
                                               blocks=self.paras.emb_blocks,
                                               density=self.paras.emb_density,
                                               dropoute=paras.dropoute, dropouti=paras.dropouti,
                                               has_decoder=False)

        self.vardropout = LockedDropout(batch_first=True)
        self.wdrop = paras.wdrop
        self.dropouth = nn.Dropout(p=paras.dropouth)

        #self.embed_dropout = nn.Dropout(p=paras.dropout)
        #todo: decide on form of dropout for embs

        lstm_input_size = self.paras.emsize

        self.lstm = nn.LSTM(lstm_input_size, self.paras.nhid, num_layers=paras.layers,
                            batch_first=True, bidirectional=True, dropout=paras.dropouth)

        if self.wdrop > 0:
            self.lstm = WeightDrop(self.lstm, ['weight_hh_l0'], dropout=self.wdrop)

        # processes the hidden states of both directions
        self.hidden2intermediate = nn.Linear(2 * self.paras.nhid, self.paras.nhid)
        #self.intermediate_dropout = nn.Dropout(p=paras.dropout)
        self.intermediate2tag = nn.Linear(self.paras.nhid, self.paras.tagset_size)

        # init weights
        self.init_weights()


    def init_weights(self):
        initrange = 0.1
        self.word_embeddings.init_weights(initrange)


    def forward(self, sentences_word, lengths):


        sentence_inputs_words = Variable(torch.LongTensor(sentences_word).cuda())
        lengths_inputs = lengths[0]

        # word input; has to be [seq, batch] for variational dropout at emb level

        word_emb = torch.transpose(self.word_embeddings(torch.t(sentence_inputs_words)), 0, 1) #[batch, seq, emsize]
        #dropout already done at SparseEmbedding level; variational dropouti and embedding dropoute

        # sentence level processing
        packed_input = pack_padded_sequence(word_emb, lengths_inputs, batch_first=True)
        lstm_out, _ = self.lstm(packed_input)
        x = lstm_out.data
        x = self.dropouth(x)

        # Wang case needs this layer
        x = self.hidden2intermediate(x)
        x = F.tanh(x)

        x = self.dropouth(x)
        tag_space = self.intermediate2tag(x)

        return tag_space

    # helper function
    def prepare_targets(self, targets, lengths):
        targets_inputs = Variable(torch.LongTensor(targets).cuda())
        lengths_inputs = lengths[0]
        packed_input = pack_padded_sequence(targets_inputs, lengths_inputs, batch_first=True)
        return packed_input.data



