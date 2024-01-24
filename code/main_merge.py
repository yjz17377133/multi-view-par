import argparse

import manager_merge

parser = argparse.ArgumentParser()

parser.add_argument('-c', '--cuda', default=[0], type=int, nargs='+', metavar='cuda',
                    help='gpu if cuda else cpu')

parser.add_argument('-l', '--lr', default=1e-4, type=float, metavar='learning_rate',
                    help='initial learning rate')

parser.add_argument('-b', '--batch_size', default=32, type=int, metavar='batch_size', # 64
                    help='batch_size for training')
                    
parser.add_argument('-b2', '--batch_size2', default=32, type=int, metavar='batch_size2', # 64
                    help='batch_size for val')

parser.add_argument('-rs', '--resolution', default=224, type=int, metavar='resolution', # 224
                    help='resolution for img')
                    
parser.add_argument('-s', '--train_step', default=30, type=int, metavar='train_step',
                    help='max training step')

parser.add_argument('-d', '--dataset', default='PA100k', type=str, metavar='dataset',
                    help="dataset (e.g. PETA,RAP)", choices=['PETA', 'RAP', 'PA100k'])

parser.add_argument('-bb', '--backbone', default='ConvBase', type=str, metavar='backbone',
                    help='backbone, e.g. Vgg19, resnet50, resnet101',
                    choices=['ConvBase', 'ConvLarge', 'ConvXlarge', 'ConvSmall', 'ConvTiny', 'Resnet50', 'Resnet101'])
                    
parser.add_argument('--ema-decay', default=0.9997, type=float, metavar='M',
                        help='decay of model ema')
parser.add_argument('--ema-epoch', default=0, type=int, metavar='M',
                        help='start ema epoch')

parser.add_argument('-o', '--optim', default='Adam', type=str, metavar='optim',
                    help='optim, e.g. Adam, optim', choices=['Adam', 'SGD'])

parser.add_argument('-cr', '--criterion', default='BCE', type=str, metavar='criterion',
                    help='criterion, e.g. BCE', choices=['BCE', 'MSE', 'Weight', 'Weight_MSE', 'Weight2', 'Weight3', 'Weight4', 'BCE_ALM', 'smooth'])

parser.add_argument('-sn', '--SubNet', default=True, type=bool, metavar='SubNet',
                    help='max training step')

parser.add_argument('-g', '--GCN', default=2, type=int, metavar='GCN',
                    help='GCN layers')

parser.add_argument('-lg', '--learn_gcn', default=False, type=bool, metavar='learn_gcn')

parser.add_argument('-info', '--information', default='save_peta', type=str)

parser.add_argument('-lga', '--learn_gcn_attention', default=True, type=bool, metavar='learn_gcn_attention')

parser.add_argument('-sp', '--saved_path', default="", type=str)

parser.add_argument('-se', '--embedding_se', default=False, type=bool, metavar='embedding_se')

parser.add_argument('-m', '--mark', default=0, type=int, metavar='mark',
                    help='0 - all 1 spatial 2 hiera ')
parser.add_argument('-n', '--net', default=0, type=int, metavar='net', help='0 ori, 1 vlad')

parser.add_argument('-p', '--platform', default="torch", type=str, metavar='platform', help='torch, paddle')

parser.add_argument('-detail', '--detail', default="False", type=bool, metavar='detail acc for label', help='true or false')

parser.add_argument('-aD', '--augData', default="normal", type=str, metavar='data Augment Operation', help='facon')

parser.add_argument('-sF', '--superFocus', default=0, nargs='+', type=int, metavar='super Focus label', help='num of index of attr')

parser.add_argument('-mB', '--multiBranch', default=False, type=bool, metavar='multi branche button', help='bool of multi branche of model')

parser.add_argument('-bu', '--but', default=False, type=bool, metavar='multi decoder object', help='index of multi decoder object of model')


def load_control(control):
    saved_param = 'model_peta.pt'

    restore_dict = control.load(saved_param)
    print('restored from %s' % saved_param)

    control.net.load_state_dict(restore_dict['model_state_dict'])   

if __name__ == '__main__':
    args = parser.parse_args()
    control = manager_merge.Manager(args)
    control.train()
