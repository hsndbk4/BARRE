import torch
import torch.nn.functional as F

import numpy as np
import os
import sys

def clamp(x, lower_limit, upper_limit):
    return torch.max(torch.min(x, upper_limit), lower_limit)

def expected_acc(models, prob, x, y, normalize, reduction='mean'):
    acc = 0
    for i in range(len(prob)):
        y_i = models[i](normalize(x))
        t_i = (y_i.max(1)[1] == y).float()
        acc += prob[i]*t_i
    return acc.mean() if reduction == 'mean' else acc

# ARC attack from [https://arxiv.org/pdf/2206.06737.pdf], with modified option for randomized order looping over the classifiers

def arc_attack(models, x_clean, label, prob, epsilon = 8 / 255.0, step_size = 2 / 255.0, attack_iter=10, rand=False, num_classes=10, normalize=None, g=1, restarts=1, use_rand_loop=True):
    for net in models:
        net.eval()
    M = len(prob)
    upper_limit = 1
    custom_beta = True
    lower_limit = 0
    topk = g + 1
    min_acc = torch.ones(x_clean.shape[0]).cuda()
    min_delta = torch.zeros_like(x_clean).cuda()
    for _ in range(restarts):
        delta_min = torch.zeros_like(x_clean).cuda()
        C = num_classes
        beta_m = step_size
        rho = 0.05*epsilon

        loss_min = expected_acc(models,prob,x_clean,label,normalize,reduction='none')
        if rand:
            delta_min.uniform_(-epsilon / 2, epsilon / 2)

        delta_min = clamp(delta_min, lower_limit-x_clean, upper_limit-x_clean)
        I = np.flip(np.argsort(prob))
        for k in range(attack_iter):
            if use_rand_loop:
                I = np.random.choice(M, M, replace=False, p=prob)
            delta_local = torch.zeros_like(x_clean).cuda()
            loss_min_local = loss_min.clone()
            for i in I:
                f_i = models[i]
                delta = (delta_min+delta_local).clone()
                delta.requires_grad = True
                out = f_i(normalize(x_clean+delta))
                out_top_k,_ = torch.topk(out,topk,dim=1)
                out_m = out_top_k[:,0]
                out_m.backward(torch.ones_like(out_m),retain_graph=True)
                del_y = delta.grad.detach().clone()
                delta.grad.zero_()
                zeta = torch.ones_like(out_m)*np.infty
                g = torch.zeros_like(x_clean).cuda()
                g_2 = torch.zeros_like(x_clean).cuda()
                for j in range(1,topk):
                    out_j = out_top_k[:,j]
                    out_j.backward(torch.ones_like(out_j),retain_graph=True)
                    del_j = delta.grad.detach()

                    with torch.no_grad():
                        zeta_j = torch.abs(out_m-out_j)
                        w_j = del_y-del_j
                        del_norm = torch.norm(w_j.view(w_j.shape[0],-1),p=1,dim=1) #q =  p/p-1 = 1

                        zeta_j = zeta_j/(del_norm+1e-10)

                        g_j = -torch.sign(w_j)
                        g_2_j = -w_j/(del_norm.view(-1,1,1,1)+1e-10)
                        update_indx = zeta>zeta_j
                        zeta[update_indx]=zeta_j[update_indx]
                        g[update_indx] = g_j[update_indx]
                        g_2[update_indx] = g_2_j[update_indx]
                    delta.grad.zero_()

                beta = torch.ones_like(label).float().cuda()
                beta = beta*beta_m


                if i!=I[0] and (custom_beta==True): #not the first classifier in the ensemble
                    vec_dot = (g_2.view(g_2.size(0),-1)*delta_local.view(delta_local.size(0),-1)).sum(dim=1)
                    beta_c = (beta_m)/(beta_m-zeta+1e-10)*(torch.abs(-vec_dot+zeta)) + rho
                    beta[zeta<beta_m]=beta_c[zeta<beta_m]

                delta_local_hat = delta_local + beta.view(-1,1,1,1)*g
                delta_local_hat = beta_m*delta_local_hat/(torch.norm(delta_local_hat.view(w_j.shape[0],-1),p=float('inf'),dim=1).view(-1,1,1,1)+1e-10)

                d = torch.clamp(delta_min + delta_local_hat, min=-epsilon, max=epsilon)

                d = clamp(d, lower_limit - x_clean, upper_limit - x_clean)

                new_loss =  expected_acc(models,prob,x_clean+d,label,normalize,reduction='none')
                delta_local[new_loss <= loss_min_local] = delta_local_hat.detach()[new_loss <= loss_min_local]
                loss_min_local = torch.min(loss_min_local, new_loss)


            d = torch.clamp(delta_min + delta_local, min=-epsilon, max=epsilon)

            d = clamp(d, lower_limit - x_clean, upper_limit - x_clean)

            new_loss =  expected_acc(models,prob,x_clean+d,label,normalize,reduction='none')
            delta_min[new_loss <= loss_min] = d.detach()[new_loss <= loss_min]
            loss_min = torch.min(loss_min, new_loss)

        min_delta[loss_min <= min_acc] = delta_min.detach()[loss_min <= min_acc]
        min_acc = torch.min(min_acc, loss_min)
    return torch.clamp(x_clean + min_delta[:x_clean.size(0)], min=lower_limit, max=upper_limit)



# adaptive PGD attack from [https://arxiv.org/pdf/2206.03362.pdf], which averages the post-softmax outputs, using the MCE loss
def apgd_attack(models, x_clean, label, prob, epsilon = 8 / 255.0, step_size = 2 / 255.0, attack_iter=10, other_weight=0., rand=True, inv=False, num_classes=10, normalize=None):
    criterion = torch.nn.CrossEntropyLoss().to(torch.device('cuda'))
    for net in models:
        net.eval()
    x = x_clean.detach()
    if rand:
        x = x + torch.zeros_like(x_clean).uniform_(-epsilon / 2, epsilon / 2)
        x = torch.clamp(x, 0., 1.)

    for _ in range(attack_iter):
        x.requires_grad_()
        with torch.enable_grad():
            pred = torch.log(sum([F.softmax(net(normalize(x)) ,dim=1)*prob[i] for i, net in enumerate(models)]))

            loss = criterion(pred, label)
            other_pred = -pred
            other_advloss = - F.log_softmax(other_pred, dim=1) * (1 - F.one_hot(label, num_classes=num_classes))
            other_advloss = other_advloss.sum() / ((num_classes - 1) * len(label))
            loss += other_weight * other_advloss

            if inv:
                loss = - loss

        grad_sign = torch.autograd.grad(loss, x, only_inputs=True, retain_graph=False)[0].detach().sign()
        x = x.detach() + step_size * grad_sign
        x = torch.min(torch.max(x, x_clean - epsilon), x_clean + epsilon)
        x = torch.clamp(x, 0., 1.)
    return x
