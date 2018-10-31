# Predefined Sparseness in Recurrent Sequence Models

This repository contains code to run the exeriments presented in our paper [Predefined Sparseness in Recurrent Sequence Models](https://arxiv.org/abs/1808.08720),
presented at CoNLL 2018.
The package `sparse_seq` contains the implementation of predefined sparse LSTM's and embedding layers, as described in that paper.
- **rnn.py**: contains *SparseLSTM*, a pytorch module that allows composing a sparse single-layer LSTM based on elementary dense LSTM's,
for a given parameter density, or given fractions in terms of input and hidden representation size
For example, with `reduce_in=0.5` and `reduce_out=0.5`, the sparse LSTM would have the same number of trainable parameters as a
dense LSTM with half the number of input and output dimensions.
Next step would be rewriting SparseLSTM for running in parallel on multiple devices,
to gain in speed and memory capacity compared to the dense LSTM.
- **embedding.py**: contains *SparseEmbedding*, a pytorch module that composes a sparse embedding layer by building the total embedding matrix
as a composition of a user-specified number individual trainable embedding blocks with smaller dimensions. As shown in the paper, this only behaves as intended,
if the vocabulary is sorted from least to most frequent terms.
Both embedding regularization mechanisms described in Merity's paper [Regularizing and Optimizing LSTM Language Models](https://arxiv.org/pdf/1708.02182.pdf)
are included in the code.

The folders `language_modeling` and `sequence_labeling` contain the code for the language modeling and part-of-speech tagging experiments
described in our paper.
The code was developed on Python 3.6.4, with pytorch 0.4.0 (CUDA V8.0, CuDNN 6.0) and all experiments were run on a GeForce GTX 1080 core.

The code is not heavily documented. I've cleaned it a little, but it's still dynamically grown research code (you know what I mean).
I'll be happy to provide more detailed descriptions if needed.
Don't hesitate to drop me an email if you have any questions:
<img src="https://tdmeeste.github.io/images/email.png" width="300px"/>


## Language modeling experiments

The `language_modeling` code is mostly based on https://github.com/salesforce/awd-lstm-lm,
but uses some parts from https://github.com/zihangdai/mos.
Given the strong dependence on hyperparameters and corresponding computational cost,
we only presented results for the Merity's model AWD-LSTM and its sparse counterpart.
Still, the code should be ready for use with the Yang et al.'s [Mixture-of-Softmaxes](https://arxiv.org/abs/1711.03953)
output layer, but we haven't tested it to avoid heavy hyperparameter tuning.
In any case, larger language modeling datasets would present a stronger test setup.

The code should work with a sparse embedding layer, but given the small relative number of embedding parameters in the setup,
and to keep the analysis untangled, we only ran experiments for AWD-LSTM with a sparse LSTM layer.

#### Baseline AWD-LSTM
The baseline can be run from the language_modeling folder as follows (after downloading the data by running `getdata.sh`):
```console
python main.py --seed 0 --save logs/awd-lstm
python finetune --save logs/awd-lstm
```
for the initial optimization run, and the finetune run, respectively.
The default parameter settings can be found in the file `args.py`.
The result, averaged over different seeds, is given in Table 1 in the paper.

The sparse model with wider middle LSTM layer (1725 dimensions instead of 1150) but predefined sparseness to maintain
the same number of recurrent layer parameters (also see Table 1) can be run with
```console
python main.py --sparse_mode sparse_hidden --sparse_fract 0.66666 --nhid 1725 --save logs/awd-lstm-sparse
python finetune.py --sparse_mode sparse_hidden --sparse_fract 0.66666 --nhid 1725 --save logs/awd-lstm-sparse
```

Finally, the *learning to recite* experiments can be run as follows.
The baseline with original dimensions and 24M parameters can be run with
```console
python main_overfit.py --save logs/awd-lstm-overfit-dense --epochs 150 --lr 5
```
where `main_overfit.py` is based on `main.py` and `args.py` in which for this particular
experiment all regularization parameters are set to 0.
The different setups with 7.07M unknowns in Table 3 can be run as follows
```console
python main_overfit.py --save logs/awd-lstm-overfit-dense_reduced --emsize 200 --nhid 575 --epochs 150 --lr 5
python main_overfit.py --save logs/awd-lstm-overfit-sparse1 --emsize 200 --sparse_mode sparse_hidden --sparse_fract 0.5 --epochs 150 --lr 5
python main_overfit.py --save logs/awd-lstm-overfit-sparse2 --emblocks 10 --emdensity 0.5 --sparse_mode sparse_all --sparse_fract 0.5 --epochs 150 --lr 5
```
in which `emblocks` and `emdensity` configure the sparse embedding layer, whereas `sparse_mode` and `sparse_fract` configure the stacked SparseLSTM layer.





## Sequence labeling experiments

The POS tagging baseline is based on code contributed by [Frederic Godin](https://www.fredericgodin.com/), augmented with the `SparseEmbedding`'s in the
`sparse_seq` package.

Dense model with reduced dimensions (Fig. 3), e.g., for embedding size 5, for one particular setting of the
regularization parameters (reported results were averaged over multiple random seeds, and tuned over a grid of hyperparameters)
```console
python main.py --emsize 5 --nhid 10 --epochs 50 --dropouti 0.2 --wdrop 0.2 --save logs/pos_dense
```

The counterpart with predefined sparse embedding layer (note that the vocab is sorted by default)
```console
python main.py --emsize 20 --emb_density 0.25 --emb_blocks 20 --nhid 10 --epochs 50 --dropouti 0.2 --wdrop 0.2 --save logs/pos_sparse
```

Finally, vocabulary sorting can be influenced with the flag `vocab_order`.
Simulating the effect of inversing the vocabulary order (such that predefined sparseness in the embedding layer
corresponds to shorter embeddings for more frequent terms, rather than the proposed ordering) can be done for instance as
```console
python main.py --emsize 20 --emb_density 0.25 --emb_blocks 20 --nhid 10 --epochs 50 --dropouti 0.2 --wdrop 0.2 --vocab_order down --save logs/pos_sparse_vocab_down
```






