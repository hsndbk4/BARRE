import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#import ipdb
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict


def add_path(path):
    if path not in sys.path:
        print('Adding {}'.format(path))
        sys.path.append(path)

add_path("../lib")
from architectures import get_architecture
from attack import arc_attack, apgd_attack
from utils import seed_torch, arr_to_str, proj_onto_simplex
from datasets import get_loaders, get_normalize, get_num_classes



parser = argparse.ArgumentParser(description='PyTorch BARRE Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '--r', default=-1, type=int)
parser.add_argument('--resume_iter', '--ri', default=-1, type=int)
parser.add_argument('--batch_size', '--b', type=int, default=256, help='batch size')
parser.add_argument('--total_epochs', "--te", type=int, default=100)
parser.add_argument("--model", type=str, default="res18", choices=["res18", "res20", "mbv1"])
parser.add_argument("--optimizer", "--opt", type=str, default="sgd", choices=["sgd", "adam"])
parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--normalize", action="store_true")
parser.add_argument("--no_aug", action="store_true")
parser.add_argument("--val_interval", "--vi", type=int, default=1)
parser.add_argument("--data", type=str, default="cifar10")
parser.add_argument('--outdir', default='outdir', type=str)
parser.add_argument('--num_workers', default=16, type=int)

parser.add_argument("--M", default=1, type=int)

parser.add_argument("--other_weight", "--ow", default=0, type=float, help='for MCE loss, set to 1')

## osp args
parser.add_argument('--osp_epochs', "--oe", type=int, default=10)
parser.add_argument('--osp_freq', "--of", type=int, default=10)
parser.add_argument('--osp_lr_max', "--olr", type=float, default=10)
parser.add_argument('--osp_batch_size', "--obm", type=int, default=512) #batch size used for osp
parser.add_argument('--osp_data_len', type=int, default=2048) #subset of trainset used for osp
args = parser.parse_args()


print("Args:", args)



outdir = args.outdir
if not os.path.exists(outdir):
    os.makedirs(outdir)

seed_torch(args.seed)
print('==> Preparing data..')

trainloader, testloader, osploader = get_loaders(args)
normalize = get_normalize(args)
num_classes = get_num_classes(args)



def train(epoch):
    train_loss = 0.
    train_adv_loss = 0.
    train_other_adv_loss = 0.
    correct = 0
    adv_correct = 0
    total = 0
    pbar = tqdm(trainloader)
    curr_lr = lr_scheduler.get_lr()[0]
    pbar.set_description("Train:{:3d} epoch lr {:.1e}".format(epoch, curr_lr))
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.cuda(), targets.cuda()
        adv_inp = apgd_attack(model_ls, inputs, targets, prob, 8 / 255.0, 2 / 255.0, 10, other_weight=args.other_weight, num_classes=num_classes, normalize=normalize)
        optimizer.zero_grad()
        model_ls[-1].train()
        pred = model_ls[-1](normalize(adv_inp))
        adv_loss = F.cross_entropy(pred, targets, reduction="none").mean()
        _, adv_predicted = pred.detach().max(1)

        other_pred = -model_ls[-1](normalize(adv_inp))
        other_advloss = - F.log_softmax(other_pred, dim=1) * (1 - F.one_hot(targets, num_classes=10))
        other_advloss = other_advloss.sum() / ((10 - 1) * len(targets))

        total_advloss = adv_loss + args.other_weight * other_advloss

        total_advloss.backward()
        total += targets.size(0)
        adv_correct += adv_predicted.eq(targets).sum().item()

        optimizer.step()

        train_adv_loss += adv_loss.item()
        train_other_adv_loss += other_advloss.item()

        pbar_dic = OrderedDict()
        pbar_dic['Adv Acc'] = '{:2.2f}'.format(100. * adv_correct / total)
        pbar_dic['adv loss'] = '{:.3f}'.format(train_adv_loss / (batch_idx + 1))
        pbar_dic['otheradv loss'] = '{:.3f}'.format(train_other_adv_loss / (batch_idx + 1))
        pbar.set_postfix(pbar_dic)



def osp_iter(epoch):

    M = len(prob)
    err = np.zeros(M)
    n = 0
    pbar = tqdm(osploader)
    osp_lr_init = args.osp_lr_max*lr_scheduler.get_lr()[0]
    curr_lr = osp_lr_init/(1+epoch)
    model_ls[-1].eval()
    pbar.set_description("OSP:{:3d} epoch lr {:.4f}".format(epoch, curr_lr))
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.cuda(), targets.cuda()
        adv_inp = arc_attack(model_ls, inputs, targets, prob, 8 / 255.0, 8 / 255.0, 10,  num_classes=num_classes, normalize=normalize, g=2)
        for m in range(M):
            t_m = model_ls[m](normalize(adv_inp))
            err[m]+= (t_m.max(1)[1] != targets).sum().item()

        n += targets.size(0)
        pbar_dic = OrderedDict()
        pbar_dic['Adv Acc'] = '{:2.2f}'.format(100. * (1-sum(err*prob)/n))
        pbar.set_postfix(pbar_dic)
    grad = err/n
    return grad

def test():
    for net in model_ls:
        net.eval()

    correct = 0
    adv_correct = 0
    total = 0
    pbar = tqdm(testloader)
    pbar.set_description('Evaluating')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            for i, net in enumerate(model_ls):
                outputs = net(normalize(inputs))
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()*prob[i]
        total += targets.size(0)
        adv_inp = arc_attack(model_ls, inputs, targets, prob, 8 / 255.0, 8 / 255.0, 10,  num_classes=num_classes, normalize=normalize, g=1)
        with torch.no_grad():
            for i, net in enumerate(model_ls):
                adv_outputs = net(normalize(adv_inp))
                _, adv_predicted = adv_outputs.max(1)
                adv_correct += adv_predicted.eq(targets).sum().item()*prob[i]

        pbar_dic = OrderedDict()
        pbar_dic['Acc'] = '{:2.2f}'.format(100. * correct / total)
        pbar_dic['Adv Acc'] = '{:2.2f}'.format(100. * adv_correct / total)
        pbar.set_postfix(pbar_dic)

    acc = 100. * correct / total
    advacc = 100. * adv_correct / total
    return acc, advacc



criterion = nn.CrossEntropyLoss()


if __name__ == "__main__":
    model_ls = []
    print_ls = []
    for iteration in range(args.M):
        print('==> Building model {}/{}'.format(iteration+1,args.M))
        model = get_architecture(args)
        model = nn.DataParallel(model).cuda()
        model_ls.append(model)
        prob = np.ones(len(model_ls))/len(model_ls)
        print('alpha = ',prob)
        if iteration >= 1:
            model_ls[-1].load_state_dict(model_ls[-2].state_dict())

        if iteration <= args.resume_iter:
            iter_save_path = os.path.join(outdir, "iter{:d}".format(iteration))
            ckpt_path = os.path.join(iter_save_path, 'model_best.pth')
            checkpoint = torch.load(ckpt_path)

            print('==> Resuming from checkpoint model_best.pth')
            if list(checkpoint['net'].keys())[0].startswith('module.'):
                model_ls[-1].load_state_dict(checkpoint['net'])
            else:
                model_ls[-1].module.load_state_dict(checkpoint['net'])
            best_advacc = checkpoint['best_advacc']
            best_correspond_acc = checkpoint["best_correspond_acc"]
            best_epoch = checkpoint["best_epoch"]
            advacc, acc = checkpoint['advacc'], checkpoint["acc"]

        else:
            best_advacc = -1.  # best test accuracy
            best_correspond_acc = -1  # corresponding clean accuracy
            start_epoch = -1  # start from epoch 0 or last checkpoint epoch
            best_epoch = -1

            if args.optimizer == "sgd":
                optimizer = optim.SGD(model_ls[-1].parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                    milestones=[int(0.5 * args.total_epochs), int(0.75 * args.total_epochs)], gamma=0.1)
            elif args.optimizer == "adam":
                optimizer = optim.Adam(model_ls[-1].parameters(), lr=args.lr, weight_decay=5e-4)

            iter_save_path = os.path.join(outdir, "iter{:d}".format(iteration))
            ckpt_path = os.path.join(iter_save_path, 'epoch{:}.pth'.format(args.resume))
            if not os.path.exists(iter_save_path):
                os.makedirs(iter_save_path)
            if os.path.exists(ckpt_path):
                print('==> Resuming from checkpoint {:d}..'.format(args.resume))
                checkpoint = torch.load(ckpt_path)
                model_ls[-1].load_state_dict(checkpoint['net'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                if args.optimizer == "sgd":
                    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                best_advacc = checkpoint['best_advacc']
                best_correspond_acc = checkpoint["best_correspond_acc"]
                best_epoch = checkpoint["best_epoch"]
                advacc, acc = checkpoint['advacc'], checkpoint["acc"]  
                start_epoch = args.resume

            print('==> Begin training for iteration {:d} ..'.format(iteration))
            for epoch in range(start_epoch + 1, args.total_epochs):

                train(epoch)
                if args.optimizer == "sgd":
                    lr_scheduler.step()
                if epoch >= args.total_epochs//2:
                    if ((epoch - args.total_epochs//2 + 1)% args.osp_freq == 0 or epoch == args.total_epochs-1) and iteration > 2:
                        eta_best = 1
                        osp_lr_init = args.osp_lr_max*lr_scheduler.get_lr()[0]
                        print('==> Begin OSP routine, starting alpha=' + arr_to_str(prob))
                        for t in range(args.osp_epochs):
                            osp_lr = osp_lr_init/(t+1)
                            g_t = osp_iter(t) #sub-gradient of eta(alpha_t)
                            eta_t = sum(g_t*prob) #eta(alpha_t)
                            if eta_t <= eta_best:
                                t_best = t
                                prob_best = np.copy(prob)
                                eta_best = eta_t
                            print("best acc = {:2.2f} @ alpha_best = ".format(100.*(1-eta_best)) + arr_to_str(prob_best))
                            prob = proj_onto_simplex(prob - osp_lr * g_t)
                        print('==> End OSP routine, final alpha=' + arr_to_str(prob_best))
                        prob = np.copy(prob_best)

                if (epoch + 1) % args.val_interval == 0 or epoch >= start_epoch + args.total_epochs - 10:
                    acc, advacc = test()
                    if advacc > best_advacc:
                        best_advacc = advacc
                        best_correspond_acc = acc
                        best_epoch = epoch
                        state = {'net': model_ls[-1].state_dict(),
                                 'alpha': prob,
                                 'optimizer': optimizer.state_dict(),
                                 'best_correspond_acc': best_correspond_acc,
                                 "best_advacc": best_advacc,
                                 'best_epoch': best_epoch,
                                 "advacc": advacc, "acc": acc,
                                 }
                        if args.optimizer == "sgd":
                            state["lr_scheduler"] = lr_scheduler.state_dict()
                        torch.save(state, os.path.join(iter_save_path, "model_best.pth"))

                    state = {'net': model_ls[-1].state_dict(),
                             'alpha': prob,
                             'optimizer': optimizer.state_dict(),
                             'best_correspond_acc': best_correspond_acc,
                             "best_advacc": best_advacc,
                             'best_epoch': best_epoch,
                             "advacc": advacc, "acc": acc,
                             }
                    if args.optimizer == "sgd":
                        state["lr_scheduler"] = lr_scheduler.state_dict()
                    torch.save(state, os.path.join(iter_save_path, "epoch{:}.pth".format(epoch)))

        print_str = "Iteration {:d}: Best advacc: {:.2f} at {:} epoch, cleanacc: {:.2f}; Last epoch cleanacc: {:.2f} advacc: {:.2f}".format(
            iteration, best_advacc, best_epoch, best_correspond_acc, acc, advacc)
        print(print_str)
        print_ls.append(print_str)
        print('Reloading best model of iteration {:d} at {:} epoch'.format(iteration,best_epoch ))
        checkpoint = torch.load(os.path.join(iter_save_path, "model_best.pth"))
        if list(checkpoint['net'].keys())[0].startswith('module.'):
            model_ls[-1].load_state_dict(checkpoint['net'])
        else:
            model_ls[-1].module.load_state_dict(checkpoint['net'])
    for s in print_ls:
        print(s)
