import argparse
import time
import os, sys
import math
import numpy as np
np.random.seed(331)
import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import model
import os

from utils import batchify, get_batch, repackage_hidden, create_exp_dir, save_checkpoint
from tensorboard_logger import configure, log_value
from args import parse_args
from shutil import copyfile



args = parse_args()


log_file = os.path.join(args.save, 'finetune_log.txt')
if not args.continue_train:
    if os.path.exists(log_file):
        os.remove(log_file)
    #to keep original trained model (with main.py): first copy into finetune_model.pt
    copyfile(os.path.join(args.save, 'model.pt'), os.path.join(args.save, 'finetune_model.pt'))


# tensorboard visualization
configure(args.save)


def logging(*s, print_=True, log_=True):
    if print_:
        print(*s)
    if log_:
        with open(log_file, 'a+') as f_log:
            f_log.write(' '.join([str(ss) for ss in s]) + '\n')

logging('finetune load path: {}/model.pt. '.format(args.save))
logging('log save path: {}/finetune_log.txt'.format(args.save))
logging('model save path: {}/finetune_model.pt'.format(args.save))


# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed_all(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = torch.load(os.path.join(args.save, 'corpus.pt'))

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
    model = torch.load(os.path.join(args.save, 'finetune_model.pt'))
else:
    model = torch.load(os.path.join(args.save, 'model.pt'))
if args.cuda:
    if not args.multi_gpu:
        parallel_model = model.cuda()
    else:
        parallel_model = nn.DataParallel(model, dim=1).cuda()
else:
    parallel_model = model
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model.parameters())
logging('Args: {}'.format(args))
logging('Model total parameters: {}'.format(total_params))

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        targets = targets.view(-1)
        
        log_prob, hidden = parallel_model(data, hidden)
        loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets).data

        total_loss += len(data) * loss
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
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

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
            log_value('finetune_train/ppl', math.exp(cur_loss), epoch)
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len

# Loop over epochs.
lr = args.lr
stored_loss = evaluate(val_data)
best_val_loss = []
min_epochs = 100 #to avoid early stopping due to noisy behavior at start

# At any point you can hit Ctrl + C to break out of training early.
try:
    #optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
    if args.continue_train:
        optimizer_state = torch.load(os.path.join(args.save, 'finetune_optimizer.pt'))
        optimizer.load_state_dict(optimizer_state)
        
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
            log_value('finetune_valid/ppl', math.exp(val_loss2), epoch)

            if val_loss2 < stored_loss:
                save_checkpoint(model, optimizer, args.save, finetune=True)
                logging('Saving Averaged!')
                stored_loss = val_loss2

            for prm in model.parameters():
                prm.data = tmp[prm].clone()

        if (len(best_val_loss)>max(min_epochs, args.nonmono) and val_loss2 > min(best_val_loss[:-args.nonmono])):
            logging('Done!')
            break
            optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
            #optimizer.param_groups[0]['lr'] /= 2.
        best_val_loss.append(val_loss2)

except KeyboardInterrupt:
    logging('-' * 89)
    logging('Exiting from training early')

# Load the best saved model.
model = torch.load(os.path.join(args.save, 'finetune_model.pt'))
parallel_model = nn.DataParallel(model, dim=1).cuda()
    
# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
logging('=' * 89)
logging('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
logging('=' * 89)
log_value('finetune_test/ppl', math.exp(test_loss), epoch)
