import argparse
import logging
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
from tqdm import tqdm
from utils import *
from architectures import get_architecture
from datasets import get_normalize, get_loaders, get_num_classes
from attack import arc_attack, apgd_attack
from collections import OrderedDict

upper_limit, lower_limit = 1,0

parser = argparse.ArgumentParser(description='Evaluating robustness')

parser.add_argument("--model", type=str, default="res18", choices=["res18", "res20", "mbv1"])
parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--num_workers', default=16, type=int)
parser.add_argument('--attack', default='pgd', type=str, choices=['pgd','arc'])

parser.add_argument('--epsilon', default=8, type=int)
parser.add_argument('--attack_iters', default=20, type=int)
parser.add_argument('--restarts', default=1, type=int)
parser.add_argument('--step_size', default=2, type=float)
parser.add_argument('--g', default=-1, type=int, help='how many hyperplanes to consider in the ARC algorithm, -1 means default arc')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--normalize', action='store_true', help='if true, no data normalization would be used for evaluating the model')

parser.add_argument('--use_osp',action='store_true')
parser.add_argument('--osp_epochs', "--oe", type=int, default=10)
parser.add_argument('--osp_lr_max', "--olr", type=float, default=10)
parser.add_argument('--osp_batch_size', "--obm", type=int, default=512) #batch size used for osp
parser.add_argument('--osp_data_len', type=int, default=2048) #subset of trainset used for osp

parser.add_argument('--logfilename', default='', type=str, help='choose the output filename')

parser.add_argument('--sourcedir', default='', type=str, help='directory/ies where models will be loaded from')
parser.add_argument('--outdir', default='cifar_model', type=str, help='directory where logfile will be stored')
parser.add_argument('--M', default=-1, type=int, help='number of classifiers to consider, if M<=M_read')
parser.add_argument('--ens-type', default='rand', type=str, choices=['rand', 'det'])
parser.add_argument('--eval-single',action='store_true', help='if true, the robust accuracies of the individual models will be evaluated as well.')
args = parser.parse_args()


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)





def get_models(args):
    model_ls = []
    list_dir = os.listdir(args.sourcedir)
    cnt = 0
    for path in sorted(list_dir):
        file_path = os.path.join(args.sourcedir,path, 'model_best.pth')
        if os.path.exists(file_path):
            ckpt =  torch.load(file_path)
            if list(ckpt['net'].keys())[0].startswith('module.'):
                state_dict = remove_module(ckpt['net'])
            else:
                state_dict = ckpt['net']
            model = get_architecture(args)
            model.load_state_dict(state_dict)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            model_ls.append(model)
            cnt+=1
            if cnt == args.M:
                break
    return model_ls

def mean_mse(x,y):
    n = len(x)
    return sum((x-y)**2)/n

def osp_iter(epoch, model_ls, prob, osploader, epsilon, step_size, attack_iters, num_classes, normalize, g, curr_lr):

    M = len(prob)
    err = np.zeros(M)
    n = 0
    pbar = tqdm(osploader)
    model_ls[-1].eval()
    pbar.set_description("OSP:{:3d} epoch lr {:.4f}".format(epoch, curr_lr))
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.cuda(), targets.cuda()
        adv_inp = arc_attack(model_ls, inputs, targets, prob, epsilon, step_size, attack_iters,  num_classes=num_classes, normalize=normalize, g=2)
        for m in range(M):
            t_m = model_ls[m](normalize(adv_inp))
            err[m]+= (t_m.max(1)[1] != targets).sum().item()

        n += targets.size(0)
        pbar_dic = OrderedDict()
        pbar_dic['Adv Acc'] = '{:2.2f}'.format(100. * (1-sum(err*prob)/n))
        pbar.set_postfix(pbar_dic)
    grad = err/n
    return grad


def test(model_ls, prob, testloader, epsilon, step_size, attack_iters, num_classes, normalize, g):
    for net in model_ls:
        net.eval()
    correct = 0
    adv_correct = 0
    total = 0
    return correct, adv_correct
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
        adv_inp = arc_attack(model_ls, inputs, targets, prob, epsilon, step_size, attack_iters,  num_classes=num_classes, normalize=normalize, g=1)
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


def main():
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    # Copies files to the outdir to store complete script with each experiment
    #copy_code(args.fname)


    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    epsilon = (args.epsilon / 255.)
    step_size = (args.step_size / 255.)
    trainloader, testloader, osploader = get_loaders(args)
    num_classes = get_num_classes(args)
    normalize = get_normalize(args)
    print('loading the models ...')

    models = get_models(args)
    for m, model in enumerate(models):
        models[m] = nn.DataParallel(model).cuda()
        models[m].eval()
    M = len(models)

    if args.use_osp:
        ## load pre-trained alphas
        list_dir = os.listdir(args.sourcedir)
        cnt = 0
        loaded = False
        for path in sorted(list_dir):
            file_path = os.path.join(args.sourcedir,path, 'model_best.pth')
            if os.path.exists(file_path):
                ckpt =  torch.load(file_path)
                cnt+=1
                if cnt == M:
                    if 'alpha' in ckpt:
                        alpha = proj_onto_simplex(ckpt['alpha'])
                        #print(alpha)
                        alpha = np.round_(alpha,decimals=3)
                        #print(alpha)
                        alpha = alpha/sum(alpha)
                        #print(alpha)
                        #if mean_mse(alpha,np.ones(M)/M)<1e-5:
                        #    alpha=np.ones(M)/M
                        print('Loaded sampling probability alpha = ' + arr_to_str(alpha))
                        loaded = True
        if loaded == False:
            eta_best = 1
            osp_lr_init = 2
            print('==> Begin OSP routine, starting alpha=' + arr_to_str(np.ones(M)/M))
            prob = np.ones(M)/M
            for t in range(args.osp_epochs):
                osp_lr = osp_lr_init/(t+1)
                g_t = osp_iter(t, models, prob, osploader, epsilon, step_size, args.osp_epochs, num_classes, normalize, args.g, osp_lr) #sub-gradient of eta(alpha_t)
                eta_t = sum(g_t*prob) #eta(alpha_t)
                if eta_t <= eta_best:
                    t_best = t
                    prob_best = np.copy(prob)
                    eta_best = eta_t
                print("best acc = {:2.2f} @ alpha_best = ".format(100.*(1-eta_best)) + arr_to_str(prob_best))
                prob = proj_onto_simplex(prob - osp_lr * g_t)
            print('==> End OSP routine, final alpha=' + arr_to_str(prob_best))
            alpha = np.copy(prob_best)

    else:
        alpha = np.ones(M)/M


    logfilename = os.path.join(args.outdir, 'eval-attack_'+args.attack+'-M_'+str(M)+'.txt')
    if os.path.exists(logfilename):
        log(logfilename, "")
        log(logfilename, "+++++++++++++++++++++++++++++++++")
        log(logfilename, "++++++++++   New Run   ++++++++++")
        log(logfilename, "+++++++++++++++++++++++++++++++++")
        log(logfilename, "")
        log(logfilename, "---------- Static Args ----------")
    else:
        init_logfile(logfilename, "---------- Static Args ----------")
    log(logfilename, 'epsilon'+'\t'+"{:.8f}".format(epsilon))
    log(logfilename, 'K'+'\t'+"{}".format(args.attack_iters))
    log(logfilename, 'step_size'+'\t'+"{}".format(step_size))
    log(logfilename, 'topk'+'\t'+"{}".format(args.g+1))
    log(logfilename, 'attack'+'\t'+"{}".format(args.attack))
    log(logfilename, 'ens-type'+'\t'+"{}".format(args.ens_type))
    log(logfilename, 'use osp'+'\t'+"{}".format(args.use_osp))
    log(logfilename, 'alpha'+'\t'+arr_to_str(alpha))
    log(logfilename, "---------- Ensemble Model Stats ----------")
    log(logfilename, "model\ta_nat\ta_rob")

    test_acc, test_robust_acc = test(models, alpha, testloader, epsilon, step_size, args.attack_iters, num_classes, normalize, args.g)
    log(logfilename, "ENS\t{:.3f}\t{:.3f}".format(test_acc,test_robust_acc))

if __name__ == "__main__":
    main()
