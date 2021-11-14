import torch
import argparse
from src.utils import *
from torch.utils.data import DataLoader
from src import training


print('Loading Corpus ...')

train_location = '/home/user/Desktop/AuxFormer/dataset/train_set.dt'
test_location = '/home/user/Desktop/AuxFormer/dataset/test_set.dt'
develop_location = '/home/user/Desktop/AuxFormer/dataset/develop_set.dt' 

train_data = torch.load(train_location)
test_data = torch.load(test_location)
develop_data = torch.load(develop_location)


print('Data loaded completed ... Training started!')    


parser = argparse.ArgumentParser(description='AuxFormer MSP@UTD')
parser.add_argument('-f', default='', type=str)

parser.add_argument('--model', type=str, default='AuxFormer',
                    help='name of the model')
parser.add_argument('--num_classes', type=int, default=6,
                    help='number of classes to predict')
parser.add_argument('--attn_dropout', type=float, default=0.1,
                    help='attention dropout')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.25,
                    help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.2,
                    help='output layer dropout (default: 0.2')
parser.add_argument('--layers', type=int, default = 5,
                    help='number of layers in the network (default: 5)')
parser.add_argument('--num_heads', type=int, default = 10,
                    help='number of heads for the transformer network (default: 10)')
parser.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for Transformer (default: true)')
parser.add_argument('--batch_size', type = int, default = 32,
                    help='batch size (default: 32)')
parser.add_argument('--clip', type = float, default = 0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type = float, default = .725e-3,
                    help='initial learning rate (default: .725e-3)')
parser.add_argument('--optim', type = str, default = 'Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--num_epochs', type=int, default = 30,
                    help='Number of Epochs (default: 30)')
parser.add_argument('--decay', type = int, default = 5,
                    help='When to decay learning rate (default: 5)')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--name', type=str, default='AuxFormer_model',
                    help='name of the saved model')
args = parser.parse_args()

use_cuda = True          

g_cuda = torch.Generator(device='cuda')

train_set = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator = g_cuda)
develop_set = DataLoader(develop_data, batch_size=args.batch_size, shuffle=False, generator = g_cuda)
test_set = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, generator = g_cuda)

args.a_dim, args.v_dim = train_data.get_dim()
args.a_len, args.v_len = train_data.get_seq_len()
args.use_cuda = use_cuda
args.when = args.decay

args.n_train = len(train_data)
args.n_valid =len(develop_data)
args.n_test = len(test_data)
args.model = str.upper(args.model.strip())
args.output_dim = args.num_classes 
args.criterion = 'CrossEntropyLoss'

if __name__ == '__main__':
    test_loss = training.initiate(args, train_set, develop_set, test_set)

print('Nepoch',args.num_epochs,' Layers', args.layers,' Outdrop', 
      args.out_dropout,' NumHeads', args.num_heads,' Batch_size', 
      args.batch_size,' Initial LR', args.lr,' Grad Clip', args.clip)

