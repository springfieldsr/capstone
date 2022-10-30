import argparse
import const


def options():
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--dataset', dest='dataset', choices=const.DATASETS,
                        default='CIFAR10', type=str)
    parser.add_argument('--model', dest='model', choices=const.MODELS,
                        default='resnet18', type=str,
                        help='datasets')
    parser.add_argument('--bs', dest='batch_size', choices=const.BATCHSIZE,
                        default=32, type=int,
                        help='batch_size')
    parser.add_argument('--epochs', dest='epochs',
                        default=30, type=int,
                        help='epochs of training')
    parser.add_argument('--lr', dest='lr',
                        default=1e-2, type=float,
                        help='learning rate')
    parser.add_argument('--k', dest='top_k',
                        default=0.05, type=float,
                        help='track top k percentage of samples with highest loss')
    parser.add_argument('--ls', dest='label_shuffle',
                        default=True, type=bool,
                        help='whether to shuffle labels of k percent of training samples')
    parser.add_argument('--es', dest='early_stop',
                        default=True, type=bool,
                        help='early stop or not')
    args = parser.parse_args()
    return args
