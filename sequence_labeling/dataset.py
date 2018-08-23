import os
import codecs
import numpy as np
from collections import Counter
import math
import random

batching_seed = np.random.RandomState(1234)

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
    def __init__(self, path="./data/ptb/", sorted='none'):
        random.seed(0)  # make sure same shuffling etc. for same corpus

        source_tags = os.path.join(path, "tags.txt")
        self.tag_dict = self.read_tags(source_tags)

        # helping vars
        self._max_length_sentence = 0

        self.dictionary = Dictionary()
        self._UNK = "<unk>"
        self._PAD = "<pad>"
        self.dictionary.add_word(self._PAD)
        self.dictionary.add_word(self._UNK)

        self.train = self.parse_corpus(os.path.join(path, 'wsj_train.conll'), train=True, sorted=sorted)
        self.valid = self.parse_corpus(os.path.join(path, 'wsj_valid.conll'))
        self.test = self.parse_corpus(os.path.join(path, 'wsj_test.conll'))

    def read_tags(self,path):
        tag_dict = {}
        f = codecs.open(path, 'r', 'utf-8')
        for index, tag in enumerate(f.readlines()):
            tag_dict[tag.strip()] = index
        f.close()

        return tag_dict

    def parse_corpus(self, name, train=False, sorted='none'):

        x_set_word = []
        y_set = []
        lengths = []
        sentences = []
        sentence = []

        #load sentences and populate dictionary
        for line in codecs.open(name, "r"):

            parts = line.strip().split("\t")

            if len(parts) > 1:
                token = parts[0].strip()
                tag = parts[1].strip()

                sentence.append((token, tag))

                if train and sorted in ['up', 'down', 'rand']:
                    self.dictionary.add_word(token)

            else:

                len_sent = len(sentence)
                if len_sent > self._max_length_sentence:
                    self._max_length_sentence = len_sent
                sentences.append(sentence)
                sentence = []

        #sort dictionary if needed and in train mode
        if train:
            self.dictionary.sort(sorted)

        #encode sentences
        for sentence in sentences:

            x_word = np.zeros((self._max_length_sentence,), dtype=np.int64)
            y = np.empty((self._max_length_sentence,), dtype=np.int64)

            i = 0
            for word, label in sentence:
                if not train and word not in self.dictionary.word2idx:
                    word = self._UNK

                x_word[i] = self.dictionary.word2idx[word]
                y[i] = self.tag_dict[label]
                i += 1

            x_set_word.append(x_word)
            y_set.append(y)
            lengths.append(i)

        x_word_final = np.vstack(x_set_word)

        return {"x":x_word_final, "y":np.vstack(y_set), "lengths":np.asarray(lengths, dtype=np.int64)}

class DataIterator:
    def __init__(self, x, lengths, y, batch_size, train=False):

        self.x = x
        self.lengths = lengths
        self.y = y


        self.batch_size = batch_size
        self.number_of_sentences = lengths.shape[0]
        if train:
            self.n_batches = self.number_of_sentences // self.batch_size
        else:
            self.n_batches = math.ceil(self.number_of_sentences / self.batch_size)
        self.train = train


    def __iter__(self):
        if self.train:
            indexes = batching_seed.permutation(np.arange(self.number_of_sentences))
        else:
            indexes = np.arange(self.number_of_sentences)

        for i in range(self.n_batches):

            lengths = self.lengths[indexes[i * self.batch_size:(i + 1) * self.batch_size]]
            y_batch = self.y[indexes[i * self.batch_size:(i + 1) * self.batch_size]]

            x_batch = self.x[indexes[i * self.batch_size:(i + 1) * self.batch_size],
                           :]

            length_sentences = lengths

            perm_idx = length_sentences.argsort(axis=0)[::-1]
            length_sentences_ordered = length_sentences[perm_idx]
            length_words_ordered = lengths[perm_idx]

            x_batch_ordered = x_batch[perm_idx, :length_sentences_ordered.max()]
            y_batch_ordered = y_batch[perm_idx,:length_sentences_ordered.max()]
            tag_mask = np.zeros((y_batch.shape[0],length_sentences_ordered[0]),dtype=np.uint8)

            yield x_batch_ordered,  y_batch_ordered, \
                      [[int(length_sentences_ordered[j]) for j in range(len(length_sentences_ordered))],
                       length_words_ordered], tag_mask



