
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
from args import parse_args

args = parse_args()

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
                       args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, 
                       args.dropoutl, args.n_experts, args.emblocks, args.emdensity,
                       sparse_mode=args.sparse_mode,
                       sparse_blocks=args.sparse_blocks, sparse_density=args.sparse_density,
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

# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000
prev_loss = 100000000
epochs_overfit = 0
MAX_EPOCHS_OVERFIT = 20
best_epoch = 0

# At any point you can hit Ctrl + C to break out of training early.
try:
    if args.continue_train:
        optimizer_state = torch.load(os.path.join(args.save, 'optimizer.pt'))
        if 't0' in optimizer_state['param_groups'][0]:
            optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
        optimizer.load_state_dict(optimizer_state)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2 = evaluate(val_data)
            logging('-' * 89)
            logging('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss2, math.exp(val_loss2)))
            logging('-' * 89)
            log_value('valid/ppl', math.exp(val_loss2), epoch)

            if val_loss2 < stored_loss:
                save_checkpoint(model, optimizer, args.save)
                best_epoch = epoch
                logging('Saving Averaged!')
                stored_loss = val_loss2

            for prm in model.parameters():
                prm.data = tmp[prm].clone()

            # stop training if loss increasing again (over MAX_EPOCHS_OVERFIT epochs without interruptions)
            epochs_overfit = 0 if val_loss2 <= prev_loss else epochs_overfit + 1
            prev_loss = val_loss2
            if epochs_overfit >= MAX_EPOCHS_OVERFIT:
                logging('-' * 89)
                logging('breaking out of ASGD loop due to overfitting for {} epochs'.format(MAX_EPOCHS_OVERFIT))
                break

        else:
            val_loss = evaluate(val_data, eval_batch_size)
            logging('-' * 89)
            logging('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            logging('-' * 89)
            log_value('valid/ppl', math.exp(val_loss), epoch)

            if val_loss < stored_loss:
                save_checkpoint(model, optimizer, args.save)
                best_epoch = epoch
                logging('Saving Normal!')
                stored_loss = val_loss

            if 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                logging('Switching!')
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
                #optimizer.param_groups[0]['lr'] /= 2.
            best_val_loss.append(val_loss)

except KeyboardInterrupt:
    logging('-' * 89)
    logging('Exiting from training early')

# Load the best saved model.
model = torch.load(os.path.join(args.save, 'model.pt'))
parallel_model = nn.DataParallel(model, dim=1).cuda()

# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
logging('=' * 89)
logging('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
log_value('test/ppl', math.exp(test_loss), best_epoch)
logging('=' * 89)
