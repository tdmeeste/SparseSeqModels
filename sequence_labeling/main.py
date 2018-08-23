import argparse
import networks, dataset, evaluator
import torch.nn as nn
import torch.optim as optim
from tensorboard_logger import configure, log_value
from util import SimpleLogger
import torch



parser = argparse.ArgumentParser(description='PoS tagging Pytorch version.')

# bilstm word
parser.add_argument("--emsize", type=int, default=100, help="Word embedding size")
parser.add_argument("--nhid", type=int, default=50, help="number of LSTM units")
parser.add_argument("--layers", type=int, default=1, help="number of LSTM/conv layers")

# training
parser.add_argument("--batch_size", type=int, default=20, help=" batch size")
parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer function: sgd or adam")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--momentum", type=float, default=0., help="momentum in case optimizer = sgd")
parser.add_argument("--epochs", type=int, default=300, help="Number of epochs to run")

#regularization
parser.add_argument("--l2", type=float, default=0.0, help="L2_reg")
parser.add_argument("--dropouti", type=float, default=0., help="variational dropout mask on input embeddings")
parser.add_argument("--dropoute", type=float, default=0., help="variational dropout of entire input embeddings")
parser.add_argument("--wdrop", type=float, default=0., help="weight dropout of hidden-to-hidden transform in rnn")
parser.add_argument("--dropouth", type=float, default=0., help="standard dropout in classification layer")

# admin
parser.add_argument("--save", type=str, default='results', help="experiment logging folder")
parser.add_argument("--seed", type=float, default=1982, help="random seed")


# special args for sparse experiments
parser.add_argument('--emb_blocks', type=int, default=1,
                    help='number of blocks sparse embedding layer')
parser.add_argument('--emb_density', type=float, default=1.0,
                    help='density of embedding matrix')
parser.add_argument('--vocab_order', type=str, default='up', choices=['none', 'up', 'down', 'rand'],
                    help='sorting vocab after loading train corpus, for use with sparse embeddings (no actual influence if non-sparse embeddings')


paras = parser.parse_args()





# setup logging
logging = SimpleLogger(paras.save) #log to file
configure(paras.save) #tensorboard logging
logging('Args: {}'.format(paras))

# set random seed for reproducibility
torch.manual_seed(paras.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(paras.seed)



# load data
logging('creating vocab (sorted: {})'.format(paras.vocab_order))
corpus = dataset.Corpus(sorted=paras.vocab_order)
paras.vocab = len(corpus.dictionary)
print('number of tokens in vocab:', paras.vocab)
paras.tagset_size = len(corpus.tag_dict)
logging('tagset size', paras.tagset_size)

logging('creating dataset batchers')
train_it = dataset.DataIterator(x=corpus.train["x"], y=corpus.train["y"],lengths=corpus.train["lengths"],batch_size=paras.batch_size, train=True)
valid_it = dataset.DataIterator(x=corpus.valid["x"], y=corpus.valid["y"],lengths=corpus.valid["lengths"], batch_size=paras.batch_size)
test_it = dataset.DataIterator(x=corpus.test["x"], y=corpus.test["y"],lengths=corpus.test["lengths"], batch_size=paras.batch_size)


# build model
logging('build model')
model = networks.LSTMTagger(paras)
model.cuda()

#report word embeddings and number of parameters
logging('word embeddings:\n', model.word_embeddings)
if hasattr(model.word_embeddings, 'start_ids'):
    logging('sparse embeddings vocab per block\n', ','.join([str(model.word_embeddings.num_embeddings-i) for i in model.word_embeddings.start_ids]))
    logging('sparse embeddings block widths\n', ','.join([str(w) for w in model.word_embeddings.block_widths]))

params_total, params_embs, params_rnns = 0, 0, 0
for n, p in model.named_parameters():
    #print('param {}: {}'.format(n, p.nelement()))
    if 'word_embeddings' in n:
        params_embs += p.nelement()
    elif 'lstm' in n:
        params_rnns += p.nelement()
    params_total += p.nelement()
logging('word embeddings: %.3fM params; rnn %.3fM params; total %.3fM params.'%(params_embs/1e6,
                                                                             params_rnns/1e6,
                                                                             params_total/1e6))

log_value('params/emb', params_embs, 0)
log_value('params/rnn', params_rnns, 0)
log_value('params/total', params_total, 0)

# prepare training
loss_function = nn.CrossEntropyLoss()

if paras.optimizer == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=paras.lr, momentum=paras.momentum, weight_decay=paras.l2)
elif paras.optimizer == "adam":
    optimizer = optim.Adam(model.parameters(), lr=paras.lr, weight_decay=paras.l2)



# training
logging("Started training")
for epoch in range(paras.epochs):
    ##################
    # training       #
    ##################
    total_loss = 0
    model.train()
    for sentences_word, tags, lengths, mask in train_it:
        # set gradients zero
        model.zero_grad()
        # run model

        tag_scores = model(sentences_word, lengths)
        tag_gt = model.prepare_targets(tags, lengths)
        # calculate loss and backprop
        loss = loss_function(tag_scores, tag_gt)
        total_loss += loss.data.cpu().numpy()
        loss.backward()
        optimizer.step()
    log_value('train/loss', total_loss, epoch)


    ##################
    # validation     #
    ##################
    model.eval()
    total_valid = 0
    correct_valid = 0
    dev_acc = evaluator.evaluate_model(model, valid_it)
    log_value('dev/acc', dev_acc, epoch)

    logging("Epoch %s: train loss %s, valid acc %s" % (epoch + 1, total_loss / train_it.n_batches, dev_acc))

##################
# test           #
##################
#TODO: implement early stopping etc.
model.eval()
total_test = 0
correct_test = 0
test_acc = evaluator.evaluate_model(model, test_it)
log_value('test/acc', dev_acc, epoch)

logging("Finished: test accuracy after last epoch: %s" % (test_acc))



