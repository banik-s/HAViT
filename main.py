from utils.mix import cutmix_data, mixup_data, mixup_criterion
import numpy as np
import random
import logging as log
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from colorama import Fore, Style
from torchsummary import summary
from utils.losses import LabelSmoothingCrossEntropy
import os
from utils.sampler import RASampler
# import models.create_model as m
from utils.logger_dict import Logger_dict
from utils.print_progress import progress_bar
from utils.training_functions import accuracy
import argparse
from utils.scheduler import build_scheduler
from utils.dataloader import datainfo, dataload
from models.create_model import create_model
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=Warning)
from Optimizers.diffGrad import diffGrad
from Optimizers.Radam import Radam
from Optimizers.AdaBelief import AdaBelief
from Optimizers.AdamInject import AdamInject
from Optimizers.AdamNorm import AdamNorm

#------------------------------------------------------------------------------------------------------------------
import time




# Record start time
start_time = time.time()
print("Training started at:", time.ctime(start_time))
# Your model training code here

#--------------------------------------------------------------------------------------------------------------------


#tensor flow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow messages of level INFO and WARNING

lowest_inference_time = float('inf')  # Initialize with infinity


best_acc1 = 0
MODELS = ['vit', 'swin', 'pit', 'cait', 't2t', 'swinv2', 'ScalableViT', 'vitlucidrains' ]


def init_parser():
    parser = argparse.ArgumentParser(description='CIFAR10/100 quick training script')

    # Data args
    parser.add_argument('--data_path', default='./dataset', type=str, help='dataset path')
    
    parser.add_argument('--dataset', default='CIFAR100', choices=['CIFAR10', 'CIFAR100', 'T-IMNET', 'SVHN'], type=str, help='Image Net dataset path')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')

    parser.add_argument('--print-freq', default=1, type=int, metavar='N', help='log frequency (by iteration)')

    # Optimization hyperparams
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    
    parser.add_argument('--warmup', default=10, type=int, metavar='N', help='number of warmup epochs')
    
    parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)', dest='batch_size')
    
    parser.add_argument('--lr', default=0.003, type=float, help='initial learning rate')
    # parser.add_argument('--lr2', default=0.006, type=float, help='initial learning rate')
    
    parser.add_argument('--weight-decay', default=5e-2, type=float, help='weight decay (default: 1e-4)')

    # parser.add_argument('--model', type=str, default='vit', choices=MODELS)
    parser.add_argument('--model', type=str, default='vit')
    
    parser.add_argument('--exp_name', type=str, default='exp0')

    parser.add_argument('--disable-cos', action='store_true', help='disable cosine lr schedule')

    parser.add_argument('--enable_aug', action='store_true', help='disable augmentation policies for training')

    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--no_cuda', action='store_true', help='disable cuda')

    parser.add_argument('--ls', action='store_false', help='label smoothing')

    parser.add_argument('--channel', type=int, help='disable cuda')

    parser.add_argument('--heads', type=int, help='disable cuda')

    parser.add_argument('--depth', type=int, help='disable cuda')

    parser.add_argument('--tag', type=str, help='tag', default='')

    parser.add_argument('--seed', type=int, default=0, help='seed')
    
    parser.add_argument('--sd', default=0.1, type=float, help='rate of stochastic depth')
    
    # parser.add_argument('--resume', default=False, help='Version')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
       
    parser.add_argument('--aa', action='store_false', help='Auto augmentation used'),
    
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    
    parser.add_argument('--cm',action='store_false' , help='Use Cutmix')
    
    parser.add_argument('--beta', default=1.0, type=float,
                        help='hyperparameter beta (default: 1)')
    
    parser.add_argument('--mu',action='store_false' , help='Use Mixup')
    
    parser.add_argument('--alpha', default=1.0, type=float,
                        help='mixup interpolation coefficient (default: 1)')

    # parser.add_argument('--suppress', default=0, type=float, help='to supprress or not, if yes, the extent, 1, 1/2, 1/3, 1/4')
    # parser.add_argument('--thres', default=1, type=float, help='which elements to suppress, elements<max/thres, 1, 2, 3, 4')
    
    parser.add_argument('--mix_prob', default=0.5, type=float,
                        help='mixup probability')
    
    parser.add_argument('--ra', type=int, default=3, help='repeated augmentation')
    
    parser.add_argument('--re', default=0.25, type=float, help='Random Erasing probability')
    
    parser.add_argument('--re_sh', default=0.4, type=float, help='max erasing area')
    
    parser.add_argument('--re_r1', default=0.3, type=float, help='aspect of erasing area')
    

    return parser


def main(args):
    global best_acc1    
    start_epoch = 0
    torch.cuda.set_device(args.gpu)

    data_info = datainfo(logger, args)
    
    model = create_model(data_info['img_size'], data_info['n_classes'], args)
    if model is None:
        raise ValueError("Failed to create model. Exiting...")

    model.cuda(args.gpu)
        
    print(Fore.GREEN + '*' * 80)
    logger.debug(f"Creating model: {model_name}")    

    #  Print Model Summary & FLOPs
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_flops = model.flops(args.batch_size)  # Call the FLOPs method from the model
    logger.debug(f' Number of Parameters: {n_parameters:,}')
    logger.debug(f' FLOPs per forward pass (batch size {args.batch_size}): {total_flops / 1e9:.6f} GFLOPs')

    logger.debug(f'Initial learning rate: {args.lr:.6f}')
    logger.debug(f"Start training for {args.epochs} epochs")
    print('*' * 80 + Style.RESET_ALL)

    if args.ls:
        print(Fore.YELLOW + '*'*80)
        logger.debug('label smoothing used')
        print('*'*80+Style.RESET_ALL)
        criterion = LabelSmoothingCrossEntropy()
    
    else:
        criterion = nn.CrossEntropyLoss()    
        
    if args.sd > 0.:
        print(Fore.YELLOW + '*'*80)
        logger.debug(f'Stochastic depth({args.sd}) used ')
        print('*'*80+Style.RESET_ALL)         

    criterion = criterion.cuda(args.gpu)

    normalize = [transforms.Normalize(mean=data_info['stat'][0], std=data_info['stat'][1])]


    if args.cm:
        print(Fore.YELLOW+'*'*80)
        logger.debug('Cutmix used')
        print('*'*80 + Style.RESET_ALL)
    if args.mu:
        print(Fore.YELLOW+'*'*80)
        logger.debug('Mixup used')
        print('*'*80 + Style.RESET_ALL)
    if args.ra > 1:        
        
        print(Fore.YELLOW+'*'*80)
        logger.debug(f'Repeated Aug({args.ra}) used')
        print('*'*80 + Style.RESET_ALL)

    '''
        Data Augmentation
    '''
    augmentations = []
    
    augmentations += [                
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(data_info['img_size'], padding=4)
            ]
    
    if args.aa == True:
        print(Fore.YELLOW+'*'*80)
        logger.debug('Autoaugmentation used')      
        
        if 'CIFAR' in args.dataset:
            print("CIFAR Policy")
            from utils.autoaug import CIFAR10Policy
            augmentations += [   
                CIFAR10Policy()
            ]
            
        elif 'SVHN' in args.dataset:
            print("SVHN Policy")    
            from utils.autoaug import SVHNPolicy
            augmentations += [
                SVHNPolicy()
            ]
                    
        else:
            from utils.autoaug import ImageNetPolicy
            augmentations += [                
                ImageNetPolicy()
            ]
            
        print('*'*80 + Style.RESET_ALL)
    
    augmentations += [                
            transforms.ToTensor(),
            *normalize]  

    if args.re > 0:
        from utils.random_erasing import RandomErasing
        print(Fore.YELLOW + '*'*80)
        logger.debug(f'Random erasing({args.re}) used ')
        print('*'*80+Style.RESET_ALL)    
        
        augmentations += [     
            RandomErasing(probability = args.re, sh = args.re_sh, r1 = args.re_r1, mean=data_info['stat'][0])
            ]
    
    
    augmentations = transforms.Compose(augmentations)
      
    train_dataset, val_dataset = dataload(args, augmentations, normalize, data_info)

    # Check if data loaders are created successfully before using them
    train_loader, val_loader = dataload(args, augmentations, normalize, data_info)
    if train_loader is None or val_loader is None:
        raise ValueError("Failed to create data loaders. Exiting...")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,  num_workers=args.workers, pin_memory=True,
        batch_sampler=RASampler(len(train_dataset), args.batch_size, 1, args.ra, shuffle=True, drop_last=True))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    '''
        Training
    '''
    
    parameters = []
    parameters += [{'params': [p for n, p in model.named_parameters() if ('suppress'not in n and  'thres' not in n)   and p.requires_grad], 'lr': args.lr}]
    
    
    optimizer = torch.optim.AdamW(parameters, weight_decay=args.weight_decay); optimizer1='AdamW'

    
    
    print('Optimizer: ',optimizer1)
    scheduler = build_scheduler(args, optimizer, len(train_loader), args.lr)

    
    print()
    print("Beginning training")
    print()
    
    lr = optimizer.param_groups[0]["lr"]
    
    if args.resume:
        checkpoint = torch.load(os.path.join(save_path, 'checkpoint.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        final_epoch = args.epochs
        
        start_epoch = checkpoint['epoch'] + 1
    
    
    for epoch in tqdm(range(start_epoch, args.epochs)):
        lr = train(train_loader, model, criterion, optimizer, epoch, scheduler, args)
        acc1 = validate(val_loader, model, criterion, lr, args, epoch=epoch)
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 
            }, 
            os.path.join(save_path, 'checkpoint.pth'))
        
        logger_dict.print()
        
        if acc1 > best_acc1:
            print('* Best model upate *')
            best_acc1 = acc1
            
            torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'optimizer2_state_dict': optimizer2.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    # 'scheduler2': scheduler2.state_dict(),
                }, os.path.join(save_path, 'best.pth'))         
        
        print(f'Best acc1 {best_acc1:.2f}')
        print('*'*80)
        print(Style.RESET_ALL)        
        
        writer.add_scalar("Learning Rate1", lr, epoch) #, "Learning Rate2", lr2, epoch)
        # writer.add_scalar("Learning Rate2", lr2, epoch)
        
        
    print(Fore.RED+'*'*80)
    logger.debug(f'best top-1: {best_acc1:.2f}, final top-1: {acc1:.2f}')
    print('*'*80+Style.RESET_ALL)
    torch.save(model.state_dict(), os.path.join(save_path, 'checkpoint.pth'))


def train(train_loader, model, criterion, optimizer, epoch, scheduler, args):
    model.train()
    loss_val, acc1_val = 0, 0
    n = 0
        
    for i, (images, target) in enumerate(train_loader):
        if (not args.no_cuda) and torch.cuda.is_available():
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
                
        # Cutmix only
        if args.cm and not args.mu:
            r = np.random.rand(1)
            if r < args.mix_prob:
                slicing_idx, y_a, y_b, lam, sliced = cutmix_data(images, target, args)
                images[:, :, slicing_idx[0]:slicing_idx[2], slicing_idx[1]:slicing_idx[3]] = sliced
                output = model(images)
                # Print the input and output shapes before the addition operation
                print(f"Input shape to transformer: {images.shape}")
                print(f"Output shape from transformer: {output.shape}")

                
                loss =  mixup_criterion(criterion, output, y_a, y_b, lam)
                
                   
            else:
                output = model(images)
                
                loss = criterion(output, target)
                               
                
        # Mixup only
        elif not args.cm and args.mu:
            r = np.random.rand(1)
            if r < args.mix_prob:
                images, y_a, y_b, lam = mixup_data(images, target, args)
                output = model(images)
                
                loss =  mixup_criterion(criterion, output, y_a, y_b, lam)
                
                
            
            else:
                output = model(images)
                
                loss =  criterion(output, target)
                 
                
        # Both Cutmix and Mixup
        elif args.cm and args.mu:
            r = np.random.rand(1)
            if r < args.mix_prob:
                switching_prob = np.random.rand(1)
                
                # Cutmix
                if switching_prob < 0.5:
                    slicing_idx, y_a, y_b, lam, sliced = cutmix_data(images, target, args)
                    images[:, :, slicing_idx[0]:slicing_idx[2], slicing_idx[1]:slicing_idx[3]] = sliced
                    output = model(images)
                    
                    loss =  mixup_criterion(criterion, output, y_a, y_b, lam)
                    
                    
                # Mixup
                else:
                    images, y_a, y_b, lam = mixup_data(images, target, args)
                    output = model(images)
                    
                    loss = mixup_criterion(criterion, output, y_a, y_b, lam) 
                    
            else:
                output = model(images)
                
                loss = criterion(output, target) 
          
        # No Mix
        else:
            output = model(images)
                                
            loss = criterion(output, target)
            
        acc = accuracy(output, target, (1,))
        acc1 = acc[0]
        n += images.size(0)
        loss_val += float(loss.item() * images.size(0))
        acc1_val += float(acc1[0] * images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        if args.print_freq >= 0 and i % args.print_freq == 0:
            avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
            progress_bar(i, len(train_loader),f'[Epoch {epoch+1}/{args.epochs}][T][{i}]   Loss: {avg_loss:.4e}   Top-1: {avg_acc1:6.2f}   LR: {lr:.7f} '+' '*10)

    logger_dict.update(keys[0], avg_loss)
    logger_dict.update(keys[1], avg_acc1)
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Acc/train", avg_acc1, epoch)
    
    f = open(save_path+'/results.txt', 'a')
    f.write('Train | Epoch: %d | Loss: %.3f | Acc: %.3f\n'
                % (epoch+1, avg_loss, avg_acc1))
    f.close()
    
    return lr


def validate(val_loader, model, criterion, lr, args, epoch=None):
    global lowest_inference_time
    model.eval()
    loss_val, acc1_val = 0, 0
    n = 0
    inference_start = time.time()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if (not args.no_cuda) and torch.cuda.is_available():
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)
            acc = accuracy(output, target, (1,))
            acc1 = acc[0]

            n += images.size(0)
            loss_val += loss.item() * images.size(0)
            acc1_val += acc1[0] * images.size(0)

            if args.print_freq > 0 and i % args.print_freq == 0:
                avg_loss, avg_acc1 = loss_val / n, acc1_val / n
                progress_bar(i, len(val_loader), f'[Epoch {epoch+1}][V][{i}]   Loss: {avg_loss:.4e}   Top-1: {avg_acc1:.2f}   LR: {lr:.6f}')

    inference_end = time.time()
    inference_time = inference_end - inference_start

    #FLOPs Calculation
    batch_size = args.batch_size
    total_flops = model.flops(batch_size)  # Call model's FLOPs method

    # Logging & Updates
    avg_loss, avg_acc1 = loss_val / n, acc1_val / n
    print(f"\n Inference Time: {inference_time:.6f} seconds | FLOPs: {total_flops / 1e9:.6f} GFLOPs")

    # Update TensorBoard
    writer.add_scalar("Loss/val", avg_loss, epoch)
    writer.add_scalar("Acc/val", avg_acc1, epoch)
    writer.add_scalar("Inference Time", inference_time, epoch)
    writer.add_scalar("FLOPs", total_flops / 1e9, epoch)

    # Save Results
    with open(save_path + '/results.txt', 'a') as f:
        f.write(f'Test | Epoch: {epoch + 1} | Loss: {avg_loss:.3f} | Acc: {avg_acc1:.3f} | Inference: {inference_time:.2f}s | FLOPs: {total_flops / 1e9:.6f} GFLOPs\n')

    return avg_acc1



if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    global save_path
    global writer
    
    # random seed

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    model_name = args.model

        
    model_name += f"-{args.tag}-{args.dataset}-LR[{args.lr}]-Seed{args.seed}-{args.exp_name}"
    save_path = os.path.join(os.getcwd(), 'CheckpointsResults', model_name)
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        
    writer = SummaryWriter(os.path.join(os.getcwd(), 'tensorboard', model_name))
    
    # logger

    log_dir = os.path.join(save_path, 'history.csv')
    logger = log.getLogger(__name__)
    formatter = log.Formatter('%(message)s')
    streamHandler = log.StreamHandler()
    fileHandler = log.FileHandler(log_dir, 'a')
    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    logger.setLevel(level=log.DEBUG)

    
    global logger_dict
    global keys
    
    logger_dict = Logger_dict(logger, save_path)
    keys = ['T Loss', 'T Top-1', 'V Loss', 'V Top-1']
    
    main(args)

#----------------------------------------------------------------------------------------------------------------------------
# Record end time
end_time = time.time()

# Calculate total training time
total_time_seconds = end_time - start_time
total_time_minutes = total_time_seconds / 60

print("Total training time: {:.2f} seconds ({:.2f} minutes)".format(total_time_seconds, total_time_minutes))

#  Print final training summary
print(f"\n Training Completed!")
print(f" Best Top-1 Accuracy: {best_acc1:.2f}%")
print(f" Lowest Inference Time: {lowest_inference_time:.2f} seconds")
print(f" Total FLOPs (per batch): {total_flops / 1e9:.6f} GFLOPs")
#----------------------------------------------------------------------------------------------------------------------------
