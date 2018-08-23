import argparse
import sys

def parse_args():

    parser = argparse.ArgumentParser(description='PyTorch experiments on sparse embeddings / rnns for language modeling')
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
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--dropouth', type=float, default=0.25,
                        help='dropout for rnn layers (0 = no dropout)')
    parser.add_argument('--dropouti', type=float, default=0.4,
                        help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--dropoute', type=float, default=0.1,
                        help='dropout to remove words from embedding layer (0 = no dropout)')
    parser.add_argument('--wdrop', type=float, default=0.5,
                        help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
    parser.add_argument('--seed', type=int, default=141,
                        help='random seed')
    parser.add_argument('--nonmono', type=int, default=5,
                        help='random seed')
    parser.add_argument('--cuda', action='store_false',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=250, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str,  default='dev_runs',
                        help='path to save the final model')
    parser.add_argument('--alpha', type=float, default=2,
                        help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
    parser.add_argument('--beta', type=float, default=1,
                        help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
    parser.add_argument('--wdecay', type=float, default=1.2e-6,
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

    #MoS specific arguments
    parser.add_argument('--dropoutl', type=float, default=-1,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--n_experts', type=int, default=1,
                        help='number of experts')

    #arguments for sparse embeddings
    parser.add_argument('--emdensity', type=float, default=1.0,
                        help='density of sparse embeddings (1.0 = not sparse)')
    parser.add_argument('--emblocks', type=int, default=1,
                        help='number of blocks in (sparse) embeddings')
    parser.add_argument('--ordering', type=str, default='up', choices=['up', 'down', 'none', 'rand'],
                        help='ordering of vocab according to doc freq. (no actual influence if non-sparse embeddings)')

    #arguments for sparse LSTMs
    parser.add_argument('--sparse_mode', type=str, default='direct',
                        choices=['direct', 'sparse_hidden', 'sparse_emb', 'sparse_all'],
                        help='mode of sparsity in recurrent layers')
    parser.add_argument('--sparse_blocks', type=int, default=1,
                        help='number of blocks in sparse RNN layers (in sparse_mode direct') #clean up
    parser.add_argument('--sparse_density', type=float, default=1.0,
                        help='density of input-hidden rnn matrices (in sparse_mode direct') #clean up
    parser.add_argument('--sparse_fract', type=float, default=1.0,
                        help='sparsity parameter for selected mode')



    args = parser.parse_args()

    if args.nhidlast < 0:
        args.nhidlast = args.emsize
    if args.dropoutl < 0:
        args.dropoutl = args.dropouth
    if args.small_batch_size < 0:
        args.small_batch_size = args.batch_size

    return args