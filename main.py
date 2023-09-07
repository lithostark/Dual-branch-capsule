#Noahs ark back up
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

from wideresnet import WideResNet
from wideresnet import SpreadLoss
from wideresnet import CapsNetv2
import numpy as np
import copy
import random

from tensorboardX import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3'

tensorboard_dir = 'log'
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)
writer = SummaryWriter(tensorboard_dir)

#netTYPE='RESsolo' # 1e-8
#netTYPE='CAPsolo' # 5e-4
netTYPE='incepCAP' # 5e-4
device = torch.device("cuda")

# used for logging to TensorBoard
#from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')


parser.add_argument('--dataset',                                  default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
# fashionmnist   cifar10   svhn   smallNORB   mnist
parser.add_argument('--epochs',                                   default=100, type=int,
                    help='number of total epochs to run')

parser.add_argument('--depth',                                    default=2, type=int, # 128
                    help='depth of network')


parser.add_argument('--start-epoch',                              default=0, type=int,
                    help='manual epoch number (useful on restarts)')


parser.add_argument('-b', '--batch-size',                          default=64, type=int, # 128
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate',                     default=1e-1, type=float, # 1e-1
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')


parser.add_argument('--weight-decay', '--wd',                     default= 0 , type=float,  # 5e-4
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    help='print frequency (default: 10)')


parser.add_argument('--layers', default=                           28, type=int,
                    help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=                     4, type=int,
                    help='widen factor (default: 10)')


parser.add_argument('--droprate', default=                         0.3, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')

parser.add_argument('--resume', default='./runs/t3/checkpoint.pth.tar', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='t3', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.set_defaults(augment=True)
parser.add_argument('--n_labeled_total', default=100, type=int,
                    help='number labeled data')
parser.add_argument('--seed', default=1, type=int,
                    help='random seed')

best_prec1 = 0

seed=1
cuda = torch.cuda.is_available()
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(seed)
if cuda:
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
device = torch.device("cuda" if cuda else "cpu")


def load_inds(n_labeled, n_classes, data, targets):
        x = data
        y = targets
        n_x = x.shape[0]
        n_classes = len(torch.unique(y))
        
        assert n_labeled % n_classes == 0, 'n_labeld not divisible by n_classes; cannot assure class balance.'
        n_labeled_per_class = n_labeled // n_classes
        
        x_labeled = [0] * n_classes
        x_unlabeled = [0] * n_classes
        y_labeled = [0] * n_classes
        y_unlabeled = [0] * n_classes
        
        idxallclass = []
        
        for i in range(n_classes):
            idxs = (y == i).nonzero().data.numpy()
            np.random.shuffle(idxs)
            idxallclass.append(idxs[:n_labeled_per_class])
        
        idxallclass = torch.tensor(idxallclass).flatten()
        randidx = torch.randperm(idxallclass.shape[0])
        idxallclassrand = idxallclass[randidx]
        
        return idxallclassrand

def create_data_label(model, nettype, trainlabeled):
    
    # --------------------------- collect data, label --------------------------- #
    #feature_collection = []
    mu_collection = []
    sigmasq_collection = []
    vote_collection = []
    label_collection = []
    
    for i, (inputs, target) in enumerate(trainlabeled):
        target = target.cuda(non_blocking=True)
        inputs = inputs.cuda(non_blocking=True)
        # compute output
        output, mu, sigmasq, vote = model(inputs, netTYPE=nettype)
        
        #feature = feature.cpu().detach()
        mu = mu.cpu().detach()
        sigmasq = sigmasq.cpu().detach()
        vote = vote.cpu().detach()
        #feature = model.module.class_caps.pose_out.cpu().detach()
        #mu = model.module.class_caps.mu.cpu().detach()
        #sigmasq = model.module.class_caps.sigma_sq.cpu().detach()
        #vote = model.module.class_caps.vote.cpu().detach()
        
        #feature_collection.append(feature)
        mu_collection.append(mu)
        sigmasq_collection.append(sigmasq)
        vote_collection.append(vote)
        label_collection.append(target.cpu().detach())
        del inputs, target, output
        print( i, len(trainlabeled) )
    
    #featureall = torch.cat(feature_collection,0).numpy()
    muall = torch.cat(mu_collection,0).numpy()
    sigmasqall = torch.cat(sigmasq_collection,0).numpy()
    voteall = torch.cat(vote_collection,0).numpy()
    labelall = torch.cat(label_collection,0).numpy()
    
    return muall, sigmasqall, voteall, labelall







def record_valid(model, nettype, trainlabeled, epoch):
    directory = 'valid_record/epoch_' + str(epoch) + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    model.eval()
    sample_count = 0
    running_loss = 0
    running_acc = 0

    output_collection = []
    mu_collection = []
    sigmasq_collection = []
    vote_collection = []
    label_collection = []


    with torch.no_grad():

        for i, (inputs, labels) in enumerate(dataloader):

            labels = labels.type(torch.LongTensor)
            # onehot_labels = torch.zeros(labels.size(0),
            #     args.n_classes).scatter_(1, labels.view(-1, 1), 1).cuda()
            inputs = inputs.type(torch.FloatTensor).cuda()

            #yhat = model(inputs)
            yhat, class_pose, class_sigma, vote_to_class = model(inputs, netTYPE=nettype)

            loss = F.cross_entropy(yhat, labels.cuda())

            sample_count += inputs.size(0)
            running_loss += loss.item() * inputs.size(0) # smaller batches count less
            running_acc += (yhat.argmax(-1) == labels.data.cuda()).sum().item() # n_corrects

            # record performance on valid dataset
            yhat = yhat.cpu().detach()
            class_pose = class_pose.cpu().detach()
            class_sigma = class_sigma.cpu().detach()
            vote_to_class = vote_to_class.cpu().detach()

            filename1 = directory + 'batch_' + str(i) + 'output_record.pt'
            torch.save(yhat, filename1)
            if nettype !='RESsolo':
                filename2 = directory + 'batch_' + str(i) + 'pose_record.pt'
                torch.save(class_pose,     filename2)
                filename3 = directory + 'batch_' + str(i) + 'sigmasq_record.pt'
                torch.save(class_sigma,filename3)
                filename4 = directory + 'batch_' + str(i) + 'vote_record.pt'
                torch.save(vote_to_class,   filename4)
                filename5 = directory + 'batch_' + str(i) + 'label_record.pt'
                torch.save(labels,  filename5)



            #output_collection.append(yhat)
            #mu_collection.append(class_pose)
            #sigmasq_collection.append(class_sigma)
            #vote_collection.append(vote_to_class)
            #label_collection.append(labels.cpu().detach())

            del inputs, labels, yhat, class_pose, class_sigma, vote_to_class










def main():
    global args, best_prec1
    args = parser.parse_args([])
    if args.tensorboard: configure("runs/%s"%(args.name))

    # Data loading code
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    if args.augment:
        transform_train = transforms.Compose([
        	transforms.ToTensor(),
        	transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
        						(4,4,4,4),mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    
    kwargs = {'num_workers': 1, 'pin_memory': True, 'worker_init_fn': np.random.seed(12)}
    path = './data'
#     assert(args.dataset == 'cifar10' or args.dataset == 'cifar100')
    if args.dataset == 'cifar10':
        num_class = 10
        datachan = 3
        train_loader = torch.utils.data.DataLoader(
            datasets.__dict__[args.dataset.upper()]('../data', train=True, download=True,
                                                    transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.__dict__[args.dataset.upper()]('../data', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        
        traindata = datasets.__dict__[args.dataset.upper()]('../data', train=True, download=True,
                                                            transform=transform_train)
        sup_inds = load_inds(args.n_labeled_total, num_class, torch.tensor(traindata.data), torch.tensor(traindata.targets))
        cifar10_labels = torch.tensor(traindata.targets)[sup_inds]
        cifar10_imgs = traindata.data[sup_inds]
        cifar10_size = cifar10_labels.shape[0] 
        print(type(cifar10_imgs), cifar10_labels.shape, cifar10_imgs.shape)
        labeled_dataset = copy.deepcopy(traindata)
        labeled_dataset.data = cifar10_imgs
        labeled_dataset.targets = cifar10_labels
        trainlabeled = torch.utils.data.DataLoader(
            labeled_dataset,
            batch_size=args.batch_size, shuffle=False)
    
    elif args.dataset == 'mnist':
        num_class = 10
        datachan = 1
        path = os.path.join('./data', args.dataset)
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        traindata = datasets.MNIST(path, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
        sup_inds = load_inds(args.n_labeled_total, num_class, traindata.data, traindata.targets)
        mnist_labels = traindata.targets[sup_inds]
        mnist_imgs = traindata.data[sup_inds]
        mnist_size = mnist_labels.shape[0] 
        print(type(mnist_imgs), mnist_labels.shape, mnist_imgs.shape)
        labeled_dataset = copy.deepcopy(traindata)
        labeled_dataset.data = mnist_imgs
        labeled_dataset.targets = mnist_labels

        trainlabeled = torch.utils.data.DataLoader(
            labeled_dataset,
            batch_size=8, shuffle=False)
        
    elif args.dataset == 'svhn':
        num_class = 10
        datachan = 3
        path = os.path.join('./data', args.dataset)
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(path, split='train', download=False,
#                           transform=transform_train
                          transform=transforms.Compose([
                              #transforms.Resize(64),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
                          ),
             batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.SVHN(path, split='test', download=True,
#                           transform=transform_test
                          transform=transforms.Compose([
                              #transforms.Resize(64),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
                          ),
             batch_size=args.batch_size, shuffle=True, **kwargs)
        traindata = datasets.SVHN(path, split='train', download=True,
                          #transform=transform_train,
                          transform=transforms.Compose([
                              #transforms.Resize(64),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
                          )
        sup_inds = load_inds(args.n_labeled_total, num_class, torch.tensor(traindata.data), torch.tensor(traindata.labels))
        svhn_labels = traindata.labels[sup_inds]
        svhn_imgs = traindata.data[sup_inds]
        svhn_size = svhn_labels.shape[0]
        print(type(svhn_imgs), svhn_labels.shape, svhn_imgs.shape)
        labeled_dataset = copy.deepcopy(traindata)
        labeled_dataset.data = svhn_imgs
        labeled_dataset.labels = svhn_labels
        trainlabeled = torch.utils.data.DataLoader(
            labeled_dataset,
            batch_size=args.batch_size, shuffle=False)

    elif args.dataset == 'smallnorb':
        from datasets.norb import smallNORB
        num_class = 5
        datachan = 1
#        path = os.path.join('./data', args.dataset)
        train_loader = torch.utils.data.DataLoader(
            smallNORB(path, train=True, download=True,
                      transform=transforms.Compose([
                          transforms.Resize(48),
                          transforms.RandomCrop(32),
                          transforms.ColorJitter(brightness=32./255, contrast=0.5),
                          transforms.ToTensor()
                      ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            smallNORB(path, train=False,
                      transform=transforms.Compose([
                          transforms.Resize(48),
                          transforms.CenterCrop(32),
                          transforms.ToTensor()
                      ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    
    elif args.dataset == 'fashionmnist':
        num_class = 10
        datachan = 1
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = datasets.FashionMNIST('./data', train=True, download=True, transform=train_transforms)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                  shuffle=True, **kwargs)
        valset = datasets.FashionMNIST('./data', train=False, transform=val_transforms)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                shuffle=True, **kwargs)
    
    
    
    
    

    # create model
#     model = WideResNet(args.layers, args.dataset == 'cifar10' and 10 or 100,
#                             args.widen_factor, dropRate=args.droprate, K=3, P=4, iters=3)
    #model = CapsNet(depth=args.layers, widen_factor=args.widen_factor, num_class=num_class, iters=2, datachan=datachan, dropRate=args.droprate)
    model = CapsNetv2(depth=args.layers, widen_factor=args.widen_factor, num_class=num_class, iters=2, datachan=datachan, dropRate=args.droprate, capslayer=args.depth)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    model = nn.DataParallel(model)
    model = model.to(device)
    model = model.cuda()

    # optionally resume from a checkpoint
    if args.start_epoch!=0:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    if netTYPE=='RESsolo':
        criterion = nn.CrossEntropyLoss().to(device)
    elif netTYPE=='incepCAP' or netTYPE=='CAPsolo':
        criterion = SpreadLoss(num_class=num_class, m_min=0.2, m_max=0.9)
        
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, nesterov = args.nesterov,
                                weight_decay=args.weight_decay)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # cosine learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*args.epochs)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,120], gamma=0.1, last_epoch=-1)
    feature_record = []
    mu_record = []
    sigmasq_record = []
    vote_record = []
    label_record = []

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, scheduler, epoch, netTYPE)

        # evaluate on validation set
        #prec1, featureall, muall, sigmasqall, voteall, labelall = validate(val_loader, model, \
        #                                                                   criterion, epoch, netTYPE, trainlabeled)
        prec1 = validate(val_loader, model, criterion, epoch, netTYPE, trainlabeled)
        #record_valid(model, netTYPE, trainlabeled, epoch)
        #if epoch < 50 :
            #feature_record.append(featureall)
            #mu_record.append(muall)
            #sigmasq_record.append(sigmasqall)
            #vote_record.append(voteall)
            #label_record.append(labelall)
            
            #torch.save(feature_record, args.dataset+netTYPE + 'feature_record.pt')
            #torch.save(mu_record, args.dataset+netTYPE + 'mu_record.pt')
            #torch.save(sigmasq_record, args.dataset+netTYPE + 'sigmasq_record.pt')
            #torch.save(vote_record, args.dataset+netTYPE + 'vote_record.pt')
            #torch.save(label_record, args.dataset+netTYPE + 'label_record.pt')

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)
    print('Best accuracy: ', best_prec1)

def train(train_loader, model, criterion, optimizer, scheduler, epoch, netTYPE):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        if args.dataset == 'smallNORB':
            target = target.squeeze(1)
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        # compute output
        output,_,_,_ = model(input, netTYPE=netTYPE)
        if netTYPE=='RESsolo':
            loss = criterion(output, target)
        elif netTYPE=='incepCAP' or netTYPE=='CAPsolo':
            r = (1.*i + (epoch-1)*len(train_loader)) / (args.epochs*len(train_loader))
            loss = criterion(output, target, r)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step() # when using consine annealing

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('current learning rate : {}'.format(optimizer.param_groups[0]['lr']))
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, top1=top1))

        iteration = epoch*len(train_loader) + i
        writer.add_scalar('train_loss', loss.item(), iteration)
        writer.add_scalar('train_acc', prec1.item(), iteration)
        #break
    #scheduler.step() # when using milestone

    # log to TensorBoard
    if args.tensorboard:
        log_value('train_loss', losses.avg, epoch)
        log_value('train_acc', top1.avg, epoch)
        

def validate(val_loader, model, criterion, epoch, netTYPE, trainlabeled):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if args.dataset == 'smallNORB':
            target = target.squeeze(1)
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            output,_,_,_= model(input,netTYPE=netTYPE)
        if netTYPE=='RESsolo':
            loss = criterion(output, target)
        else:
            loss = criterion(output, target, r=1)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

        writer.add_scalar('test_loss', losses.val, epoch)
        writer.add_scalar('test_acc', top1.val, epoch)

        #break
    #featureall, muall, sigmasqall, voteall, labelall = create_data_label(model, netTYPE, trainlabeled)
    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.avg, epoch)
        
    return top1.avg #, featureall, muall, sigmasqall, voteall, labelall


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
