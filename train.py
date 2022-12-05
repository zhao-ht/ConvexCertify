
import torch.distributed as dist

from torch.utils.data import (DataLoader, RandomSampler)

from pipeline import *



if __name__ == "__main__":
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if args.save_path is not None:
        args.save_path = os.path.join('backup_checkpoint',args.save_path)
    if args.load_path is not None:
        args.load_path = os.path.join('backup_checkpoint',args.load_path)
    if args.save_path is not None:
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)

    with open(os.path.join(args.save_path, 'args.pkl'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    # from argparse import ArgumentParser
    # parser = ArgumentParser()
    # args = parser.parse_args()
    # with open('commandline_args.txt', 'r') as f:
    #     args.__dict__ = json.load(f)

    # tokenizer = ModifiedBertTokenizer.from_pretrained(args.pretrained_name, do_lower_case=True)
    # print(tokenizer.encode_plus("he's here"))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        if args.use_cpu:
            args.device = ['cpu']
            args.parallel = False
        device = args.device
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
        args.device = [device]
        args.parallel = False


      # need to write into arguments.py

    train_data, dev_data, test_data, syn_data = make_synthesized_data(args)
    if args.sentence_level_certify_model:
        empirical_data = make_attack_data(args)
    if args.train_number is not None:
        length = len(train_data)
        start=int(length/2-args.train_number/2)
        end=int(length/2+args.train_number/2)
        train_data.x=train_data.x[start:end]
        train_data.y = train_data.y[start:end]
    if args.test_number is not None:
        length = len(test_data)
        start=int(length/2-args.test_number/2)
        end=int(length/2+args.test_number/2)
        test_data.x=test_data.x[start:end]
        test_data.y = test_data.y[start:end]

    tokens_to_ids = train_data.tokens_to_ids
    ids_to_tokens = train_data.ids_to_tokens
    bert_base_tokens_to_ids = train_data.bert_base_tokens_to_ids
    bert_base_ids_to_tokens = train_data.bert_base_ids_to_tokens
    args.tokens_to_ids = tokens_to_ids
    args.ids_to_tokens=ids_to_tokens
    args.bert_base_tokens_to_ids = bert_base_tokens_to_ids
    args.bert_base_ids_to_tokens = bert_base_ids_to_tokens


    train_loader = torch.utils.data.DataLoader(train_data, args.batch_size, sampler=RandomSampler(train_data) \
        if args.local_rank == -1 else DistributedSampler(train_data, shuffle=True))
    dev_loader = torch.utils.data.DataLoader(dev_data, args.batch_size, sampler=RandomSampler(dev_data) \
        if args.local_rank == -1 else DistributedSampler(dev_data, shuffle=True))
    test_loader = torch.utils.data.DataLoader(test_data, args.test_batch_size, sampler=RandomSampler(test_data) \
        if args.local_rank == -1 else DistributedSampler(test_data, shuffle=True))
    certify_loader = torch.utils.data.DataLoader(test_data, args.certify_batch_size, sampler=RandomSampler(test_data) \
        if args.local_rank == -1 else DistributedSampler(test_data, shuffle=True))
    if args.sentence_level_certify_model:
        empirical_loader = torch.utils.data.DataLoader(empirical_data, args.certify_batch_size, shuffle=True, num_workers=0)
    if args.certify_on_transferred_data:
        certify_loader=empirical_loader
        test_loader=empirical_loader
    args.vocab_size=len(tokens_to_ids)

    model = Certify_VAE(args).to(device[0])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = get_scheduler(args,optimizer)

    step=0
    gather=DataGather(['loss','loss_CLS','accu','loss_info','NLL1','NLL2',
                       'KL_loss','KL_weight','Ibp_loss','radius','radius_loss','coef',
                       'loss_clas_hidden','cr_hidden','acc_certify','bounded_certify','ratio_certify'],
                      [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],args.save_path)


    load_dict_bert = {'model': model,
                 'optimizer': optimizer,
                 'tb_step': step}


    if args.load_path is not None:
        step=load_model(load_dict_bert,step,args.load_path,args.device[0])
        gather.write('loading from {}'.format(args.load_path))
    if args.init_lr is not None:
        for param_group in optimizer.param_groups:  # 在每次更新参数前迭代更改学习率
            param_group["lr"] = args.init_lr
    if args.init_step is not None:
        step=args.init_step


    if args.parallel:
        model=DataParallel(model,device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[torch.distributed.get_rank()], output_device=args.local_rank, find_unused_parameters=True)


    if args.criterion_maximize:
        loss_best=-100000000
    else:
        loss_best = 100000000

    if args.local_rank == 0:
        torch.distributed.barrier()


    for epoch in range(args.epoch):

        count=0
        if args.train_model:
            optimizer, scheduler, args, gather, step, count, epoch=train_epoch(model,optimizer,scheduler,train_loader,dev_loader,args,gather,device,step,count,epoch)

        if args.adv_train_model:
            optimizer, args, gather, step, count, epoch=adver_train_epoch(model,optimizer,train_loader,dev_loader,args,gather,device,step,count,epoch)

        if args.verify_model:
            optimizer, args, gather, step, count, epoch = verify_epoch(model, optimizer, train_loader, dev_loader,
                                                                            args, gather, device, step, count, epoch)
        if args.test_model:
            optimizer, args, gather, step, count, epoch = test_epoch(model, optimizer, train_loader, test_loader,
                                                                        args, gather, device, step, count, epoch)

################# saveing model
        if args.local_rank in [-1, 0]:
            loss_best=save_epoch(model, optimizer, args, gather, step, loss_best)

        if args.certify_model or args.sentence_level_certify_model:
            args, gather, count, epoch=certify_epoch(model, certify_loader, args, gather, device, epoch)

        if args.empirical_certify_model:
            args, gather, count, epoch=empirical_certify_epoch(model,certify_loader,args,gather,device,epoch)

        if args.sentence_level_certify_model:
            args, gather, count, epoch=sentence_level_certify_epoch(model,empirical_loader,args,gather,device,epoch)











