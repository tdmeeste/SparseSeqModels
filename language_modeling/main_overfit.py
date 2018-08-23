
import os, sys
import time
import math
import numpy as np
import torch
import torch.nn as nn

import gc

import data
import model

from utils import batchify, get_batch, repackage_hidden, create_exp_dir, save_checkpoint
from tensorboard_logger import configure, log_value
import argparse


###############################################################################
# Load arguments (simplified from args.py for overfit experiment)
###############################################################################

parser = argparse.ArgumentParser(description='Language modeling overfit experiment')
parser.add_argument('--data', type=str, default='data/penn',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nhidlast', type=int, default=-1,
                    help='number of hidden units for the last rnn layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=500,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20,
                    metavar='N', help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=250, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='dev_runs_overfit',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=0,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=0,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=0.,
                    help='weight decay applied to all weights')
parser.add_argument('--continue_train', action='store_true',
                    help='continue train from a checkpoint')
parser.add_argument('--small_batch_size', type=int, default=-1,
                    help='the batch size for computation. batch_size should be divisible by small_batch_size.\
                     In our implementation, we compute gradients with small_batch_size multiple times, and accumulate the gradients\
                     until batch_size is reached. An update step is then performed.')
parser.add_argument('--max_seq_len_delta', type=int, default=40,
                    help='max sequence length')
parser.add_argument('--multi_gpu', default=False, action='store_true',
                    help='use all alvailable GPUs')

# MoS specific arguments
parser.add_argument('--n_experts', type=int, default=1,
                    help='number of experts')

# arguments for sparse embeddings
parser.add_argument('--emdensity', type=float, default=1.0,
                    help='density of sparse embeddings (1.0 = not sparse)')
parser.add_argument('--emblocks', type=int, default=1,
                    help='number of blocks in (sparse) embeddings')
parser.add_argument('--ordering', type=str, default='up', choices=['up', 'down', 'none', 'rand'],
                    help='ordering of vocab according to doc freq. (no actual influence if non-sparse embeddings)')

# arguments for sparse LSTMs
parser.add_argument('--sparse_mode', type=str, default='direct',
                    choices=['direct', 'sparse_hidden', 'sparse_emb', 'sparse_all'],
                    help='mode of sparsity in recurrent layers')
parser.add_argument('--sparse_blocks', type=int, default=1,
                    help='number of blocks in sparse RNN layers (in sparse_mode direct')  # clean up
parser.add_argument('--sparse_density', type=float, default=1.0,
                    help='density of input-hidden rnn matrices (in sparse_mode direct')  # clean up
parser.add_argument('--sparse_fract', type=float, default=1.0,
                    help='sparsity parameter for selected mode')

args = parser.parse_args()

if args.nhidlast < 0:
    args.nhidlast = args.emsize
if args.small_batch_size < 0:
    args.small_batch_size = args.batch_size


if not args.continue_train:
    #args.save = '{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(args.save, scripts_to_save=['main.py', 'model.py', 'args.py'])

def logging(*s, print_=True, log_=True):
    if print_:
        print(*s)
    if log_:
        with open(os.path.join(args.save, 'log.txt'), 'a+') as f_log:
            f_log.write(' '.join([str(ss) for ss in s]) + '\n')

# tensorboard visualization
configure(args.save)


# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed_all(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data, sorted=args.ordering)
logging('loaded corpus (vocab sorted: %s)'%args.ordering)
corpus_file = os.path.join(args.save, 'corpus.pt')
torch.save(corpus, corpus_file)
logging('saved corpus to', corpus_file)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
if args.continue_train:
    model = torch.load(os.path.join(args.save, 'model.pt'))
else:
    model = model.RNNModel(ntokens, args.emsize, args.nhid, args.nhidlast, args.nlayers,
                       0., 0., 0., 0., 0., 0., args.n_experts, args.emblocks, args.emdensity,
                       sparse_mode=args.sparse_mode,
                       sparse_fract=args.sparse_fract)

if args.cuda:
    if not args.multi_gpu:
        parallel_model = model.cuda()
    else:
        parallel_model = nn.DataParallel(model, dim=1).cuda()
else:
    parallel_model = model

logging('Args: {}'.format(args))

params_total, params_encoder, params_rnns = 0, 0, 0
for n, p in model.named_parameters():
    #print('param {}: {}'.format(n, p.nelement()))
    if 'encoder' in n:
        params_encoder += p.nelement()
    elif 'rnns' in n:
        params_rnns += p.nelement()
    params_total += p.nelement()
logging('params encoder: {}M'.format(params_encoder / 1.e6))
logging('params rnns: {}M'.format(params_rnns / 1.e6))
logging('params total: {}M'.format(params_total / 1.e6))

log_value('params rnn', params_rnns, 0)
log_value('params encoder', params_encoder, 0)
log_value('params total', params_total, 0)

#write out model
logging('model:\n'+str(model))

# info on sparse rnns:
logging('Recurrent layers info:')
info = model.info_dict()
logging(info)

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        targets = targets.view(-1)
        log_prob, hidden = parallel_model(data, hidden)
        loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets).data
        total_loss += loss * len(data)

        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)


def train():
    assert args.batch_size % args.small_batch_size == 0, 'batch_size must be divisible by small_batch_size'

    # Turn on training mode which enables dropout.
    total_loss = 0
    total_instances, total_correct = 0, 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = [model.init_hidden(args.small_batch_size) for _ in range(args.batch_size // args.small_batch_size)]
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        seq_len = min(seq_len, args.bptt + args.max_seq_len_delta)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len) #both (seq_length, batch_size)

        optimizer.zero_grad()

        start, end, s_id = 0, args.small_batch_size, 0
        while start < args.batch_size:
            cur_data, cur_targets = data[:, start: end], targets[:, start: end].contiguous().view(-1)

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden[s_id] = repackage_hidden(hidden[s_id])

            log_prob, hidden[s_id], rnn_hs, dropped_rnn_hs = parallel_model(cur_data, hidden[s_id], return_h=True)
            raw_loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), cur_targets)

            # determine whether max is correct (before updating)
            _, max_score_index = torch.max(log_prob, dim=2)
            max_score_index = max_score_index.view(-1)
            total_instances += max_score_index.size()[0]
            total_correct += torch.sum((max_score_index == cur_targets).int()).data.item()

            loss = raw_loss
            # Activiation Regularization
            loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
            # Temporal Activation Regularization (slowness)
            loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
            loss *= args.small_batch_size / args.batch_size
            total_loss += raw_loss.data * args.small_batch_size / args.batch_size
            loss.backward()

            s_id += 1
            start = end
            end = start + args.small_batch_size

            gc.collect()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        # total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            logging('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            log_value('train/ppl', math.exp(cur_loss), epoch)
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len

    #at end of epoch: report reconstruction accuracy
    acc = float(total_correct)/total_instances
    logging('Epoch %d  -   %d/%d train items correctly predicted;  accuracy %.3f'%(epoch, total_correct, total_instances, acc))
    log_value('overfit_acc', acc, epoch)
    return acc

# Loop over epochs.
lr = args.lr
eps = 1.e-4

# At any point you can hit Ctrl + C to break out of training early.
# only use SGD in overfit experiments, and add momentum
try:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10,
    #                                                       factor=0.5, threshold=0, min_lr=0.1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.97)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        acc = train()

        if (1-acc) < eps:
            log_value('perfectly_memorized', acc, epoch)
            break

        scheduler.step()

except KeyboardInterrupt:
    logging('-' * 89)
    logging('Exiting from training early')

