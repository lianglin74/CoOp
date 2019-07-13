import argparse


def get_arg_parser(model_names):
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # necessary inputs
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    # DEBUG
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='DEBUG mode')

    # has default hyper parameter for ResNet
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--effective-batch-size', default=256, type=int,
                        metavar='N', help='num of samples in a batch given to all GPUs')
    parser.add_argument('--batch_size', default=0, type=int,
                        help='do not use this, use effective_batch_size')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')

    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to checkpoint to resume from (default: none)')
    parser.add_argument('--restore_latest_snapshot', action='store_true',
                        help='restore from latest snapshot')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--init_from', default='', type=str, metavar='PATH',
                        help='path to a pretrained model to initialize from')
    parser.add_argument('--skip_unmatched_layers', action='store_true',
                        help='used in pair with init_from, skip unmatched layers if True, otherwise raise error')

    # distributed training
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--init_method_type', default='tcp', type=str)
    parser.add_argument('--dist_url_tcp_port', default=23456, type=int)

    parser.add_argument('--random_seed', default=6, type=int)

    # need setup output dir
    parser.add_argument('--output-dir', default='./outputs/resnet18', type=str,
                        help='path to save checkpoint and log (default: ./outputs/resnet18)')
    # parser.add_argument('--prefix', default=None, type=str,
    #                     help='model prefix (default: same with model names)')

    # Optimization setting
    parser.add_argument('--lr-policy', default='STEP', type=str,
                        help='learning rate decay policy: STEP, MULTISTEP, EXPONENTIAL, PLATEAU, CONSTANT '
                             '(default: STEP)')
    parser.add_argument('--step-size', default=30, type=int,
                        help='step size for STEP decay policy (default: 30)')
    parser.add_argument('--milestones', default='30,60,90', type=str,
                        help='milestones for MULTISTEP decay policy (default: 30,60,90)')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='gamma for learning rate decay (default: 0.1)')
    parser.add_argument('--neg', dest='neg_weight_file', default=None,
                        help='weights of negative samples used in multi-label training. If specified, balanced loss will be used, otherwise, BCELoss will be used.')

    # force using customized hyper parameter
    parser.add_argument('-f', '--force', dest='force', action='store_true',
                        help='force using customized hyper parameter')

    parser.add_argument('--bn_no_weight_decay', dest='bn_no_weight_decay', action='store_true',
                        help='No weight decay for bn and bias during taining')
    parser.add_argument('--BatchNormEvalMode', dest='BatchNormEvalMode', action='store_true',
                        help='Use eval mode for batch normalization layer. Corresponding to Caffe use global statistics')

    parser.add_argument('--finetune', dest='finetune', action='store_true',
                        help='finetune last layer by using 0.1x lr for previous layers')
    parser.add_argument('--fixfeature', dest='fixfeature', action='store_true',
                        help='fix features, only optimize the fc layer')
    parser.add_argument('--fixpartialfeature', dest='fixpartialfeature', action='store_true',
                        help='only optimize the last feature block and fc layer')

    parser.add_argument('--ccs_loss_param', default=0.0, type=float,
                        help='ccl_loss_param * ccs_loss + classification loss')

    parser.add_argument('--balance_class', dest='balance_class', action='store_true',
                        help='use weighted CrossEntropyLoss, where weights[c] is n_samples / (n_classes * label_counts[c])')
    parser.add_argument('--balance_sampler', action='store_true',
                        help='class aware sampling for imbalanced dataset')
    parser.add_argument('--data_aug', default=0, type=int,
                        help='data augmentation type')
    parser.add_argument('--enlarge_bbox', default=1.0, type=float,
                        help='make bbox larger (factor*width, factor*height)')
    parser.add_argument('--input_size', default=224, type=int,
                        help='input image size')

    # cache policy
    parser.add_argument('--cache_tmp', action='store_true',
                        help='use cache policy tmp in TSVFile ')

    return parser
