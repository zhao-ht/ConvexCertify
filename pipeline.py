# import argparse
# import numpy as np
# import torch as t
# from torch.optim import Adam
# import torch.nn.functional as F
# from utils.batchloader import BatchLoader
# from utils.parameters import Parameters
# from model.vae import VAE
# from torch.autograd import Variable
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import sys
import torch
#sys.path.append("..")
from Arguments import parser
from data import make_synthesized_data
from model import BaseModel
from tqdm import tqdm
from utils import ModifiedBertTokenizer
from utils import DiscreteChoiceTensor
from model.certify_vae import Certify_VAE
from aux_function import DataGather,DataParallel,correct_rate_func
import os
import pickle
import json
import numpy as np
# import nltk
# nltk.download('punkt')
import torch.distributed as dist
# from apex.parallel import DistributedDataParallel as DDP
# from apex import amp
from data import make_synthesized_data, make_attack_data
from torch.utils.data import (DataLoader, RandomSampler)
from torch.utils.data.distributed import DistributedSampler




def save_model(model_dic,save_path,local_rank=-1):
    model_state_dic={}
    for key in model_dic.keys():
        if not 'tb_step' in key:
            model_state_dic[key]=model_dic[key].state_dict() if local_rank==-1 else model_dic[key].module.state_dict()
        else:
            model_state_dic[key] = model_dic[key]
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    path=os.path.join(save_path,'model' + '.pth')
    print('saveing to {}'.format(path))
    torch.save(model_state_dic, path)

def load_model(model_dic,tb_step,load_path,local_rank=-1,device=0):
    path = os.path.join(load_path, 'model' + '.pth')
    print('loading from {}'.format(path))
    checkpoint=torch.load(path,map_location=torch.device(device))
    for key in model_dic.keys():
        if not 'tb_step' in key:
            try:
                model_dic[key].load_state_dict(checkpoint[key])
            except:
                model_dic[key].module.load_state_dict(checkpoint[key])
        else:
            tb_step=checkpoint[key]
    return tb_step


def train_epoch(model,optimizer,scheduler,train_loader,dev_loader,args,gather,device,step,count,epoch):
    gather.write('train' + '-' * 28)
    gather.flush()
    if args.train_on_dev:
        train = dev_loader
    else:
        train = train_loader
    if args.use_tqdm:
        gene = tqdm(train, disable=args.local_rank not in [-1, 0])
    else:
        gene = train
    for batch in gene:
        model.train()
        step = step + 1
        for item in batch:
            batch[item] = batch[item].to(device[0])
        model.zero_grad()

        coef = model.coef_function(args.Generator_fashion, step, args.Generator_max_coef, args.Generator_max_step)
        loss_cls, accu, loss_info, NLL_loss1, NLL_loss2, KL_loss, ibp_loss, radius, recons1, recons2, z, mu, log_sigma, loss_clas_hidden, cr_hidden = model(
            batch, coef)
        radius_loss = torch.relu(ibp_loss - radius + args.radius_margin)
        indices = ~torch.isnan(radius_loss) & ~torch.isinf(
            radius_loss)
        radius_loss = radius_loss[indices]
        loss_cls, accu, loss_info, NLL_loss1, NLL_loss2, \
        KL_loss, ibp_loss, radius, radius_loss, loss_clas_hidden, cr_hidden = \
            loss_cls.mean(), accu.mean(), loss_info.mean(), NLL_loss1.mean(), \
            NLL_loss2.mean(), KL_loss.mean(), ibp_loss.mean(), radius.mean(), radius_loss.mean(), loss_clas_hidden.mean(), cr_hidden.mean()
        KL_weight = model.coef_function(args.IBP_fashion, step, args.IBP_max_coef, args.IBP_max_step,
                                        args.IBP_start_step)

        if KL_weight == 0:
            loss = loss_cls + args.generator_coef * (NLL_loss1 + NLL_loss2 + loss_info)
        else:
            loss = loss_cls + args.generator_coef * (NLL_loss1 + NLL_loss2 + loss_info) + KL_weight * radius_loss
        if args.classify_on_hidden or args.only_classify_on_hidden:
            loss = loss + loss_clas_hidden
        if args.warm_up:
            if args.warm_up_method == 'train_encoder_first':
                if step == 1:
                    params = [
                        {"params": model.encoder.parameters(), "lr": args.learning_rate_2},
                        {"params": model.embedding_encoder.parameters(), "lr": args.learning_rate_2},
                        {"params": model.decoder.parameters(), "lr": args.learning_rate},
                        {"params": model.base_model.parameters(), "lr": args.learning_rate}
                    ]
                    optimizer = torch.optim.Adam(params)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
                    gather.write('warming up start, lr: ' + str(optimizer.param_groups[0]['lr']))
                if step <= args.warm_up_step:
                    loss = loss_clas_hidden
                if step == args.warm_up_step:
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
                    gather.write('warming up end, lr: ' + str(optimizer.param_groups[0]['lr']))
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
                    args.classify_on_hidden = False
            else:
                raise ValueError("warm up method wrong")
        loss.backward()

        optimizer.step()

        gather.insert(['loss', 'loss_CLS', 'accu', 'loss_info', 'NLL1', 'NLL2', 'KL_loss',
                       'KL_weight', 'Ibp_loss', 'radius', 'radius_loss', 'coef', 'loss_clas_hidden', 'cr_hidden'],
                      [float(loss), float(loss_cls), float(accu), float(loss_info), float(NLL_loss1), float(NLL_loss2),
                       float(KL_loss), float(KL_weight), float(ibp_loss), float(radius), float(radius_loss),
                       float(coef), float(loss_clas_hidden), float(cr_hidden)])
        count += 1
        if count % args.print_interval == 0:
            gather.report()

    gather.report('train ' + str(epoch) + ': ')

    tem = gather.get_report()
    if args.not_use_scheduler:
        pass
    else:
        scheduler.step(tem[0])
    gather.write('lr: ' + str(scheduler.optimizer.param_groups[-1]['lr']))
    tem = gather.get_report()

    return optimizer,scheduler,args,gather,step,count,epoch

def adver_train_epoch(model,optimizer,train_loader,dev_loader,args,gather,device,step,count,epoch):
    gather.write('adv_train' + '-' * 28)
    gather.flush()
    if args.train_on_dev:
        train = dev_loader
    else:
        train = train_loader
    if args.use_tqdm:
        gene = tqdm(train, disable=args.local_rank not in [-1, 0])
    else:
        gene = train
    for batch in gene:
        model.train()
        step = step + 1
        for item in batch:
            batch[item] = batch[item].to(device[0])
        model.zero_grad()

        if args.adv_method == 'freelb':
            # ============================ Code for adversarial training=============
            # initialize delta
            embeds_init = model.get_emb(batch['sent'])
            if args.adv_init_mag > 0:

                input_mask = batch['mask'].to(embeds_init)
                input_lengths = torch.sum(input_mask, 1)
                # check the shape of the mask here..

                if args.norm_type == "l2":
                    delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                    dims = input_lengths * embeds_init.size(-1)
                    mag = args.adv_init_mag / torch.sqrt(dims)
                    delta = (delta * mag.view(-1, 1, 1)).detach()
                elif args.norm_type == "linf":
                    delta = torch.zeros_like(embeds_init).uniform_(-args.adv_init_mag,
                                                                   args.adv_init_mag) * input_mask.unsqueeze(2)

            else:
                delta = torch.zeros_like(embeds_init)

            # the main loop
            dp_masks = None
            for astep in range(args.adv_steps):
                # (0) forward
                delta.requires_grad_()
                batch['input_emb'] = delta + embeds_init
                batch['dp_masks'] = dp_masks

                coef = model.coef_function(args.Generator_fashion, step, args.Generator_max_coef,
                                           args.Generator_max_step)
                loss_cls, accu, loss_info, NLL_loss1, NLL_loss2, KL_loss, ibp_loss, radius, recons1, recons2, z, mu, log_sigma, loss_clas_hidden, cr_hidden = model(
                    batch, coef, input_preprocess=False, idirect_nput_to_bert_by_sent=False)
                radius_loss = torch.relu(ibp_loss - radius + args.radius_margin)
                indices = ~torch.isnan(radius_loss) & ~torch.isinf(
                    radius_loss)
                radius_loss = radius_loss[indices]
                loss_cls, accu, loss_info, NLL_loss1, NLL_loss2, \
                KL_loss, ibp_loss, radius, radius_loss, loss_clas_hidden, cr_hidden = \
                    loss_cls.mean(), accu.mean(), loss_info.mean(), NLL_loss1.mean(), \
                    NLL_loss2.mean(), KL_loss.mean(), ibp_loss.mean(), radius.mean(), radius_loss.mean(), loss_clas_hidden.mean(), cr_hidden.mean()
                KL_weight = model.coef_function(args.IBP_fashion, step, args.IBP_max_coef, args.IBP_max_step,
                                                args.IBP_start_step)

                if KL_weight == 0:
                    loss = loss_cls + args.generator_coef * (NLL_loss1 + NLL_loss2 + loss_info)
                else:
                    loss = loss_cls + args.generator_coef * (
                            NLL_loss1 + NLL_loss2 + loss_info) + KL_weight * radius_loss
                if args.classify_on_hidden or args.only_classify_on_hidden:
                    loss = loss + loss_clas_hidden

                # loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
                # # (1) backward
                # if args.n_gpu > 1:
                #     loss = loss.mean()  # mean() to average on multi-gpu parallel training
                # if args.gradient_accumulation_steps > 1:
                #     loss = loss / args.gradient_accumulation_steps

                loss = loss / args.adv_steps

                loss.backward()

                if astep == args.adv_steps - 1:
                    # further updates on delta
                    break

                # (2) get gradient on delta
                delta_grad = delta.grad.clone().detach()

                # (3) update and clip
                if args.norm_type == "l2":
                    denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                    denorm = torch.clamp(denorm, min=1e-8)
                    delta = (delta + args.adv_lr * delta_grad / denorm).detach()
                    if args.adv_max_norm > 0:
                        delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                        exceed_mask = (delta_norm > args.adv_max_norm).to(embeds_init)
                        reweights = (args.adv_max_norm / delta_norm * exceed_mask \
                                     + (1 - exceed_mask)).view(-1, 1, 1)
                        delta = (delta * reweights).detach()
                elif args.norm_type == "linf":
                    denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1,
                                                                                                             1)
                    denorm = torch.clamp(denorm, min=1e-8)
                    delta = (delta + args.adv_lr * delta_grad / denorm).detach()
                    if args.adv_max_norm > 0:
                        delta = torch.clamp(delta, -args.adv_max_norm, args.adv_max_norm).detach()
                else:
                    print("Norm type {} not specified.".format(args.norm_type))
                    exit()

                embeds_init = model.get_emb(batch['sent'])

        elif args.adv_method == 'ascc':
            attack_type_dict = {'num_steps': 10, 'loss_func': 'ce', 'w_optm_lr': 1, 'sparse_weight': 15,
                                'out_type': "text"}
            coef = model.coef_function(args.Generator_fashion, step, args.Generator_max_coef,
                                       args.Generator_max_step)
            input_ori = model.ascc_certify(batch, attack_type_dict)
            loss_cls, accu, loss_info, NLL_loss1, NLL_loss2, KL_loss, ibp_loss, radius, recons1, recons2, z, mu, log_sigma, loss_clas_hidden, cr_hidden = model(
                input_ori, coef, input_preprocess=False, idirect_nput_to_bert_by_sent=False)
            radius_loss = torch.relu(ibp_loss - radius + args.radius_margin)
            indices = ~torch.isnan(radius_loss) & ~torch.isinf(
                radius_loss)
            radius_loss = radius_loss[indices]
            loss_cls, accu, loss_info, NLL_loss1, NLL_loss2, \
            KL_loss, ibp_loss, radius, radius_loss, loss_clas_hidden, cr_hidden = \
                loss_cls.mean(), accu.mean(), loss_info.mean(), NLL_loss1.mean(), \
                NLL_loss2.mean(), KL_loss.mean(), ibp_loss.mean(), radius.mean(), radius_loss.mean(), loss_clas_hidden.mean(), cr_hidden.mean()
            KL_weight = model.coef_function(args.IBP_fashion, step, args.IBP_max_coef, args.IBP_max_step,
                                            args.IBP_start_step)

            if KL_weight == 0:
                loss = loss_cls + args.generator_coef * (NLL_loss1 + NLL_loss2 + loss_info)
            else:
                loss = loss_cls + args.generator_coef * (
                        NLL_loss1 + NLL_loss2 + loss_info) + KL_weight * radius_loss
            if args.classify_on_hidden or args.only_classify_on_hidden:
                loss = loss + loss_clas_hidden

            loss.backward()

        optimizer.step()

        gather.insert(['loss', 'loss_CLS', 'accu', 'loss_info', 'NLL1', 'NLL2', 'KL_loss',
                       'KL_weight', 'Ibp_loss', 'radius', 'radius_loss', 'coef', 'loss_clas_hidden', 'cr_hidden'],
                      [float(loss), float(loss_cls), float(accu), float(loss_info), float(NLL_loss1), float(NLL_loss2),
                       float(KL_loss), float(KL_weight), float(ibp_loss), float(radius), float(radius_loss),
                       float(coef), float(loss_clas_hidden), float(cr_hidden)])
        count += 1
        if count % args.print_interval == 0:
            gather.report()
        return optimizer, args, gather, step, count, epoch


def verify_epoch(model,optimizer,train_loader,dev_loader,args,gather,device,step,count,epoch):
    gather.write('verify' + '-' * 28)
    gather.flush()
    with torch.no_grad():
        model.eval()
        if args.use_tqdm:
            gene = tqdm(dev_loader, disable=args.local_rank not in [-1, 0])
        else:
            gene = dev_loader
        for batch in gene:

            for item in batch:
                batch[item] = batch[item].to(device[0])

            coef = model.coef_function(args.Generator_fashion, step, args.Generator_max_coef,
                                       args.Generator_max_step)
            loss_cls, accu, loss_info, NLL_loss1, NLL_loss2, KL_loss, ibp_loss, radius, recons1, recons2, z, mu, log_sigma, loss_clas_hidden, cr_hidden = model(
                batch, coef)
            radius_loss = torch.relu(ibp_loss - radius + args.radius_margin)
            indices = ~torch.isnan(radius_loss) & ~torch.isinf(
                radius_loss)
            radius_loss = radius_loss[indices]
            loss_cls, accu, loss_info, NLL_loss1, NLL_loss2, \
            KL_loss, ibp_loss, radius, radius_loss, loss_clas_hidden, cr_hidden = \
                loss_cls.mean(), accu.mean(), loss_info.mean(), NLL_loss1.mean(), \
                NLL_loss2.mean(), KL_loss.mean(), ibp_loss.mean(), radius.mean(), radius_loss.mean(), loss_clas_hidden.mean(), cr_hidden.mean()
            KL_weight = model.coef_function(args.IBP_fashion, step, args.IBP_max_coef, args.IBP_max_step,
                                            args.IBP_start_step)

            if KL_weight == 0:
                loss = loss_cls + args.generator_coef * (NLL_loss1 + NLL_loss2 + loss_info)
            else:
                loss = loss_cls + args.generator_coef * (
                        NLL_loss1 + NLL_loss2 + loss_info) + KL_weight * radius_loss
            if args.classify_on_hidden or args.only_classify_on_hidden:
                loss = loss + loss_clas_hidden
            if args.warm_up:
                if args.warm_up_method == 'classify_on_hidden':
                    if step == 1:
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = args.learning_rate_2
                        gather.write('warming up start, lr: ' + str(optimizer.param_groups[0]['lr']))
                    if step <= args.warm_up_step:
                        loss = loss_clas_hidden
                    if step == args.warm_up_step:
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = args.learning_rate
                        gather.write('warming up end, lr: ' + str(optimizer.param_groups[0]['lr']))

            gather.insert(['loss', 'loss_CLS', 'accu', 'loss_info', 'NLL1', 'NLL2', 'KL_loss',
                           'KL_weight', 'Ibp_loss', 'radius', 'radius_loss', 'coef', 'loss_clas_hidden',
                           'cr_hidden'],
                          [float(loss), float(loss_cls), float(accu), float(loss_info), float(NLL_loss1),
                           float(NLL_loss2),
                           float(KL_loss), float(KL_weight), float(ibp_loss), float(radius), float(radius_loss),
                           float(coef), float(loss_clas_hidden), float(cr_hidden)])
            count += 1
            if count % args.print_interval == 0:
                gather.report()

    gather.report('verify ' + str(epoch) + ': ')

    return optimizer, args, gather, step, count, epoch


def test_epoch(model,optimizer,train_loader,test_loader,args,gather,device,step,count,epoch):
    gather.write('test' + '-' * 28)
    gather.flush()
    if args.use_tqdm:
        gene = tqdm(test_loader, disable=args.local_rank not in [-1, 0])
    else:
        gene = test_loader
    with torch.no_grad():
        model.eval()
        for batch in gene:
            for item in batch:
                batch[item] = batch[item].to(device[0])

            coef = model.coef_function(args.Generator_fashion, step, args.Generator_max_coef,
                                       args.Generator_max_step)
            loss_cls, accu, loss_info, NLL_loss1, NLL_loss2, KL_loss, ibp_loss, radius, recons1, recons2, z, mu, log_sigma, loss_clas_hidden, cr_hidden = model(
                batch, coef)
            radius_loss = torch.relu(ibp_loss - radius + args.radius_margin)
            indices = ~torch.isnan(radius_loss) & ~torch.isinf(
                radius_loss)
            radius_loss = radius_loss[indices]
            loss_cls, accu, loss_info, NLL_loss1, NLL_loss2, \
            KL_loss, ibp_loss, radius, radius_loss, loss_clas_hidden, cr_hidden = \
                loss_cls.mean(), accu.mean(), loss_info.mean(), NLL_loss1.mean(), \
                NLL_loss2.mean(), KL_loss.mean(), ibp_loss.mean(), radius.mean(), radius_loss.mean(), loss_clas_hidden.mean(), cr_hidden.mean()
            KL_weight = model.coef_function(args.IBP_fashion, step, args.IBP_max_coef, args.IBP_max_step,
                                            args.IBP_start_step)

            if KL_weight == 0:
                loss = loss_cls + args.generator_coef * (NLL_loss1 + NLL_loss2 + loss_info)
            else:
                loss = loss_cls + args.generator_coef * (
                        NLL_loss1 + NLL_loss2 + loss_info) + KL_weight * radius_loss
            if args.classify_on_hidden or args.only_classify_on_hidden:
                loss = loss + loss_clas_hidden
            if args.warm_up:
                if args.warm_up_method == 'classify_on_hidden':
                    if step == 1:
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = args.learning_rate_2
                        gather.write('warming up start, lr: ' + str(optimizer.param_groups[0]['lr']))
                    if step <= args.warm_up_step:
                        loss = loss_clas_hidden
                    if step == args.warm_up_step:
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = args.learning_rate
                        gather.write('warming up end, lr: ' + str(optimizer.param_groups[0]['lr']))

            gather.insert(['loss', 'loss_CLS', 'accu', 'loss_info', 'NLL1', 'NLL2', 'KL_loss',
                           'KL_weight', 'Ibp_loss', 'radius', 'radius_loss', 'coef', 'loss_clas_hidden',
                           'cr_hidden'],
                          [float(loss), float(loss_cls), float(accu), float(loss_info), float(NLL_loss1),
                           float(NLL_loss2),
                           float(KL_loss), float(KL_weight), float(ibp_loss), float(radius), float(radius_loss),
                           float(coef), float(loss_clas_hidden), float(cr_hidden)])
            count += 1
            if count % args.print_interval == 0:
                gather.report()

    gather.report('test ' + str(epoch) + ': ')

    return optimizer, args, gather, step, count, epoch

def certify_epoch(model,certify_loader,args,gather,device,epoch):
    count = 0
    verify = 0
    verify_num = 0

    gather.write('certify' + '-' * 28)
    gather.flush()
    with torch.no_grad():
        model.eval()
        if args.use_tqdm:
            gene = tqdm(certify_loader, disable=args.local_rank not in [-1, 0])
        else:
            gene = certify_loader
        for batch in gene:
            for item in batch:
                batch[item] = batch[item].to(device[0])
            res, radius, mu_ibp = model.certify(batch, args.certify_first_sample_size, args.certify_second_sample_size,
                                                args.alpha)
            acc = (res == batch['label']) * (res != -1).float()
            bounded = (radius > torch.norm(mu_ibp.ub - mu_ibp.lb, dim=1)).float()
            if args.verify_certification:
                _, acc_em = model.emperical_certify(batch)
                if acc * bounded >= 1:
                    verify_num += 1
                    verify += float(acc_em.mean() >= 1)
            gather.insert(['acc_certify', 'bounded_certify', 'ratio_certify'],
                          [float(acc.mean()), float(bounded.mean()), float((acc * bounded).mean())])
            count += 1
            if count % 20 == 0:
                gather.report('certify ' + str(epoch) + ': ')
                if args.verify_certification:
                    print(verify / verify_num)
    gather.report('certify ' + str(epoch) + ': ')
    return args, gather,  count, epoch



def empirical_certify_epoch(model,certify_loader,args,gather,device,epoch):
    count = 0
    gather.write('empirical_certify' + '-' * 28)
    gather.flush()
    with torch.no_grad():
        model.eval()
        if args.use_tqdm:
            gene = tqdm(certify_loader)
        else:
            gene = certify_loader
        for batch in gene:
            for item in batch:
                batch[item] = batch[item].to(device[0])
            # bounded,acc = model.emperical_certify(batch)
            attack_type_dict = {'num_steps': 10, 'loss_func': 'ce', 'w_optm_lr': 1, 'sparse_weight': 15,
                                'out_type': "text"}
            input_ori = model.ascc_certify(batch, attack_type_dict)
            loss_cls, acc, _, _, _, _, _, _, _, _, _, _, _, _, _ = model(input_ori, 1)
            # print(acc)
            gather.insert(['acc_certify'], [float(acc.mean() >= 1)])
            count += 1
            if count % 20 == 0:
                gather.report('certify ' + str(epoch) + ': ')
    gather.report('certify ' + str(epoch) + ': ')
    return args, gather, count, epoch



def sentence_level_certify_epoch(model,empirical_loader,args,gather,device,epoch):
    count = 0
    gather.write('sentence level empirical certify' + '-' * 28)
    gather.flush()
    if args.use_tqdm:
        gene = tqdm(empirical_loader)
    else:
        gene = empirical_loader
    with torch.no_grad():
        model.eval()
        for batch in gene:
            for item in batch:
                batch[item] = batch[item].to(device[0])
            if args.direct_input_to_bert:
                loss_cls, accu, loss_info, NLL_loss1, NLL_loss2, KL_loss, ibp_loss, radius, recons1, recons2, z, mu, log_sigma, loss_clas_hidden, cr_hidden = model(
                    batch, 1)
                gather.insert(['acc_certify'], [accu.mean()])
            else:
                acc = model.sentence_certify(batch, 1, args.certify_second_sample_size)
                gather.insert(['acc_certify'], [acc])
            count += 1
            if count % 20 == 0:
                gather.report('certify ' + str(epoch) + ': ')
    gather.report('certify ' + str(epoch) + ': ')
    return args, gather, count, epoch



def save_epoch(model,optimizer,args,gather,step,loss_best):
    tem = gather.get_report()
    loss = tem[args.criterion_id]
    flag = True
    if 'best' in args.save_criterion and ((loss > loss_best) ^ args.criterion_maximize):
        flag = False
    if (loss < loss_best) ^ args.criterion_maximize:
        loss_best = loss
    if flag:
        if args.save_model:
            if step % args.save_frequency == 0:
                load_dict_bert = {'model': model,
                                  'optimizer': optimizer,
                                  'tb_step': step}
                save_model(load_dict_bert, args.save_path)
                gather.write('model saved')
    gather.write('loss_best: ' + str(loss_best))
