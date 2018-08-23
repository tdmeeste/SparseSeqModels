import os
import torch
import random
from collections import Counter


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def sort(self, mode):
        if mode == 'none':
            return
            #sorted_counts = list(self.counter.items())  # original order
        elif mode == 'rand':
            sorted_counts = list(self.counter.items())  # shuffled (fixed)
            random.shuffle(sorted_counts)
        elif mode == 'up':  # from low-freq to high-freq
            sorted_counts = sorted(self.counter.items(), key=lambda i: i[1], reverse=False)
        elif mode == 'down':  # from high-freq to low-freq
            sorted_counts = sorted(self.counter.items(), key=lambda i: i[1], reverse=True)

        sorted_ids = [sc[0] for sc in sorted_counts]
        new_idx2word = [self.idx2word[id] for id in sorted_ids]
        new_counter = Counter({i: sorted_counts[i][1] for i in range(len(sorted_ids))})
        new_word2idx = {w: i for i, w in enumerate(new_idx2word)}
        self.idx2word = new_idx2word
        self.counter = new_counter
        self.word2idx = new_word2idx

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, sorted='none'):
        random.seed(0)  # make sure same shuffling etc. for same corpus
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'), sorted=sorted)
        #print('Note: after loading training data: %d words (linear reading of training data)'%self.dictionary.total)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path, sorted='none'):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding='utf-8') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # sort tokens if need be
        if sorted in ['up', 'down', 'rand']:
            self.dictionary.sort(sorted)

        # Tokenize file content
        with open(path, 'r', encoding='utf-8') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1



        return ids


class SentCorpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding='utf-8') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        sents = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line:
                    continue
                words = line.split() + ['<eos>']
                sent = torch.LongTensor(len(words))
                for i, word in enumerate(words):
                    sent[i] = self.dictionary.word2idx[word]
                sents.append(sent)

        return sents


class BatchSentLoader(object):
    def __init__(self, sents, batch_size, pad_id=0, cuda=False, volatile=False):
        self.sents = sents
        self.batch_size = batch_size
        self.sort_sents = sorted(sents, key=lambda x: x.size(0))
        self.cuda = cuda
        self.volatile = volatile
        self.pad_id = pad_id

    def __next__(self):
        if self.idx >= len(self.sort_sents):
            raise StopIteration

        batch_size = min(self.batch_size, len(self.sort_sents) - self.idx)
        batch = self.sort_sents[self.idx:self.idx + batch_size]
        max_len = max([s.size(0) for s in batch])
        tensor = torch.LongTensor(max_len, batch_size).fill_(self.pad_id)
        for i in range(len(batch)):
            s = batch[i]
            tensor[:s.size(0), i].copy_(s)
        if self.cuda:
            tensor = tensor.cuda()

        self.idx += batch_size

        return tensor

    next = __next__

    def __iter__(self):
        self.idx = 0
        return self


if __name__ == '__main__':
    corpus = SentCorpus('../penn')
    loader = BatchSentLoader(corpus.test, 10)
    for i, d in enumerate(loader):
        print(i, d.size())
