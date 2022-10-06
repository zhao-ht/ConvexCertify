from model.decoder import Decoder,Decoder_RNN
from model.encoder import Encoder,Encoder_IBP,TextCNN,CNNModel,BOWModel,CNNModel_bypass
from model.base import BaseModel
import torch
from model.ibp import max_diff_norm
import numpy as np
import model.ibp as ibp
import os
from aux_function import correct_rate_func
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import norm
from scipy.stats import binom_test
from torch.distributions.normal import Normal
from utils import IntervalBoundedTensor
import torch.nn.functional as F
def grl_hook(coeff):
    def fun1(grad):
        return coeff*grad.clone()
    return fun1

class Certify_VAE(BaseModel):

    def __init__(self, args):
        super(Certify_VAE, self).__init__(args)
        self.args=args
        self.args.ibp_encode = False
        if args.encoder_type=='ibp':
            self.encoder=Encoder_IBP(args.embedding_dim,args.latent_size)
            self.args.ibp_encode=True
        elif args.encoder_type=='cnn':
            self.encoder = Encoder(args.embedding_dim, args.latent_size)
        elif args.encoder_type=='textcnn':
            self.encoder = TextCNN(args.embedding_dim, args.n_filters, args.filter_sizes,
                 args.dropout)
        elif args.encoder_type=='cnnibp':
            if args.bypass_premise:
                self.encoder=CNNModel_bypass(args.embedding_dim,args.latent_size,3,'mean',args.dropout,args.no_wordvec_layer)
            else:
                self.encoder = CNNModel(args.embedding_dim,args.latent_size,3,'mean',args.dropout,args.no_wordvec_layer)
            self.args.ibp_encode = True
        elif args.encoder_type=='BOW_SNLI':
            self.encoder = BOWModel(args.embedding_dim, args.latent_size, args.dropout, args.num_layer)
            self.args.ibp_encode = True
        weight=self.bert_model.embeddings.word_embeddings.weight.detach()
        if args.decoder_type=='cnn':
            self.decoder=Decoder(args.vocab_size, args.latent_size, args.rnn_size, args.rnn_num_layers, args.embedding_dim,weight,args)
        elif args.decoder_type=='rnn':
            self.decoder = Decoder_RNN(args.vocab_size, args.latent_size, args.rnn_size, args.rnn_num_layers,
                                   args.embedding_dim, weight, args)
        self.NLL = torch.nn.CrossEntropyLoss(reduction='none')
        self.STD=args.std

    def coef_function(self,anneal_function, step, k, x0,start=0):
        if anneal_function == 'logistic':
            return float(1 / (1 + np.exp(-k * (step - x0))))
        elif anneal_function == 'linear':
            return k*min(1, max((step-start),0) / x0)
        elif anneal_function == 'zero':
            return 0.0

    def reparameterize(self,mu,sigma):
        z=torch.randn(mu.shape).to(mu.device)
        # return mu+sigma*z
        return mu + self.STD*z

    def classify_on_hidden(self,input,label):
        mu,log_sigma,mu_ibp = self.encoder(input)
        if self.args.ibp_encode:
            mu_val=mu.val
            log_sigma_val=log_sigma.val
        else:
            mu_val = mu
            log_sigma_val = log_sigma
        z = self.reparameterize(mu_val, log_sigma_val.exp())
        logit=self.decoder.hidden_to_label(z)
        tem = torch.tensor(0.0).to(input["sent"].device)
        if self.args.radius_on_hidden_classify:
            radius = self.soft_radius(logit, input['label'])
        else:
            radius=tem

        loss_clas_hidden = self.NLL(logit, label).mean()
        if self.args.ibp_loss:
            ibp_loss=self.compute_IBP_loss(mu)
        else:
            ibp_loss=tem
        return loss_clas_hidden,correct_rate_func(logit,label),ibp_loss,radius


    def encode(self,input,sample_num=1):
        mu,log_sigma,_=self.encoder(input)
        if self.args.ibp_encode:
            mu_val=mu.val
            log_sigma_val=log_sigma.val
        else:
            mu_val = mu
            log_sigma_val = log_sigma

        mu_val = mu_val.repeat(sample_num, 1)
        log_sigma_val = log_sigma_val.repeat(sample_num, 1)
        z = self.reparameterize(mu_val, log_sigma_val.exp())

        return z,mu_val,log_sigma_val,mu

    def decode(self,z,input_decode):
        return self.decoder(z,input_decode)

    def loss_fn(self,logp1,logp2, target, mask, mean, logv,mu_ibp):
        tem = torch.tensor(0.0).to(logp1.device)
        batch_size=logp2.shape[0]
        # cut-off unnecessary padding from target, and flatten

        if self.args.reconstruct_by_RNN:
            logp1 = logp1.view(-1, logp1.size(2))
            target=target.view(-1)
            mask=mask.view(-1)
            # Negative Log Likelihood
            NLL_loss1 = (self.NLL(logp1, target)*mask).sum()/batch_size
        else:
            NLL_loss1=tem

        logp2 = logp2.reshape([-1, logp2.size(2)])
        target = target.reshape([-1])
        mask = mask.reshape([-1])
        # Negative Log Likelihood
        NLL_loss2 = (self.NLL(logp2, target) * mask).sum() / batch_size

        # KL Divergence
        # KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())/batch_size
        KL_loss=tem

        return NLL_loss1,NLL_loss2, KL_loss

    def generate(self,latent_variable):
        cnn_out = self.decoder.conv_decoder(latent_variable)
        state=None
        init_input_decoder =self.args.tokens_to_ids['[PAD]'] * torch.ones([latent_variable.shape[0], 1]).to(latent_variable.device).type(torch.long)
        res=[]
        input=init_input_decoder
        for i in range(cnn_out.shape[1]):
        # input_decoder=sent
            input_emb = self.get_emb(input)
            logits1,state = self.decoder.rnn_decoder(cnn_out[:,i,:].unsqueeze(1), input_emb, initial_state=state,return_final=True)
            next = logits1.argmax(2)
            res.append(next)
            input=next
        res=torch.cat(res,1)
        return res

    def generate_from_text(self,input):
        mask = None
        ibp_input = None
        sent = input["sent"]
        text_like_syn = input["text_like_syn"]
        text_like_syn_valid = input["text_like_syn_valid"]
        if "mask" in input:
            mask = input["mask"]

        if self.args.ibp_encode:
            # if "ibp" in input:
            #     ibp_input = input["ibp_input"]

            if ibp_input == None:
                ibp_input = self.ibp_input_from_convex_hull(sent, text_like_syn, text_like_syn_valid, mask)
                # input["ibp_input"] = ibp_input
            input = ibp_input
        else:
            input = self.get_emb(sent)
        z, mu, log_sigma,_ = self.encode(input)
        res=self.generate(z)

        return res

    def input_preprocess(self, input, convex_hull=True):

        sent = input["sent"]

        if self.args.ibp_encode:
            mask = None
            ibp_input = None
            text_like_syn = input["text_like_syn"]
            text_like_syn_valid = input["text_like_syn_valid"]
            if "mask" in input:
                mask = input["mask"]

            # if "ibp" in input:
            #     ibp_input = input["ibp_input"]
            if convex_hull:
                if ibp_input == None:
                    ibp_input = self.ibp_input_from_convex_hull(sent, text_like_syn, text_like_syn_valid, mask)
                    # input["ibp_input"] = ibp_input
            else:
                ibp_input = self.ibp_input_from_convex_hull(sent, None, None, mask)

            input_emb = ibp_input
        else:
            input_emb = self.get_emb(sent)

        input['input_emb'] = input_emb

        if 'premises' in input:
            sent = input["premises"]
            mask = None
            ibp_input = None
            text_like_syn = input["text_like_syn_premises"]
            text_like_syn_valid = input["text_like_syn_valid_premises"]
            if "mask_premises" in input:
                mask = input["mask_premises"]

            # if "ibp" in input:
            #     ibp_input = input["ibp_input"]

            if ibp_input == None:
                ibp_input = self.ibp_input_from_convex_hull(sent, text_like_syn, text_like_syn_valid, mask)
                # input["ibp_input"] = ibp_input
            input_emb = ibp_input
            input['input_emb_premises'] = input_emb

        if 'hypothesis' in input:
            sent = input["hypothesis"]
            mask = None
            ibp_input = None
            text_like_syn = input["text_like_syn_hypothesis"]
            text_like_syn_valid = input["text_like_syn_valid_hypothesis"]
            if "mask_hypothesis" in input:
                mask = input["mask_hypothesis"]
            #
            # if "ibp" in input:
            #     ibp_input = input["ibp_input"]

            if ibp_input == None:
                ibp_input = self.ibp_input_from_convex_hull(sent, text_like_syn, text_like_syn_valid, mask)
                # input["ibp_input"] = ibp_input
            input_emb = ibp_input
            input['input_emb_hypothesis'] = input_emb

        return input

    def compute_IBP_loss(self,mu_ibp):
        if self.args.IBP_loss_type == 'mean':
            tem2 = mu_ibp.ub - mu_ibp.lb
            ibp_loss = tem2.mean(1).mean(0)
        elif self.args.IBP_loss_type == 'l2':
            ibp_loss = torch.norm(torch.maximum(
                (mu_ibp.ub - mu_ibp.val).abs(), (mu_ibp.val - mu_ibp.lb).abs()
            ), dim=1)
        return ibp_loss

    def forward_vae(self,input,coeff=1):

        tem = torch.tensor(0.0).to(input["label"].device)

        if self.args.encoder_type != 'cnnibp':
            z,mu,log_sigma,mu_ibp=self.encode(input)
        else:
            z, mu, log_sigma, mu_ibp = self.encode(input)
        # input_decoder=torch.cat([self.args.tokens_to_ids['[PAD]']*torch.ones([input["sent"].shape[0],1])
        #                         .to(input["label"].device),input["sent"][:,0:-1]],1).type(torch.long)
        # # input_decoder=sent
        # input_decoder=self.get_emb(input_decoder)
        input_decoder=None
        recons1,recons2,emb1,emb2,clas=self.decode(z,input_decoder)


        if self.args.info_loss=='reconstruct':
            NLL_loss1, NLL_loss2, KL_loss = self.loss_fn(recons1, recons2, input["sent"],
                                                                               input["mask"], mu, log_sigma, mu_ibp,
                                                                               )
        else:
            NLL_loss1=tem
            NLL_loss2=tem
            KL_loss=tem

        if self.args.ibp_loss:
            ibp_loss=self.compute_IBP_loss(mu_ibp)
        else:
            ibp_loss=tem
        return NLL_loss1, NLL_loss2, KL_loss,ibp_loss,recons1, recons2, z, mu, log_sigma,emb1,emb2,clas



    def data_to_loss(self,input):
        NLL_loss1, NLL_loss2, KL_loss, ibp_loss, recons1, recons2, z, mu, log_sigma=self.forward_vae(input)
        return NLL_loss1.mean(), NLL_loss2.mean(), KL_loss.mean(), ibp_loss.mean(), recons1, recons2, z, mu, log_sigma


    def classify_by_bert(self,input,recons1,recons2,emb1,emb2,coeff,sample_num=1):
        if not self.args.direct_input_to_bert:

            weight = self.bert_model.embeddings.word_embeddings.weight.detach()

            if self.args.reconstruct_by_vocab:
                if self.args.reconstruct_by_RNN:
                    sents_recons = torch.nn.Softmax(2)(recons1)
                else:
                    sents_recons = torch.nn.Softmax(2)(recons2)
                new_emb = sents_recons @ weight
            else:
                if self.args.reconstruct_by_RNN:
                    new_emb = emb1
                else:
                    new_emb = emb2
            if self.args.detach_reconstruct:
                new_emb = new_emb.detach()
            if self.args.renormlize_embedding:
                new_emb = torch.nn.functional.normalize(new_emb, p=2, dim=2) * 1.4

            try:
                new_emb.register_hook(grl_hook(coeff))
            except:
                pass

            tem = weight[self.args.tokens_to_ids['[CLS]'], :].unsqueeze(0).repeat(emb2.shape[0], 1).unsqueeze(
                1)
            tem = torch.cat([tem, new_emb], 1)
            mask = torch.cat([torch.ones([input['label'].shape[0], 1]).to(input['mask'].device), input['mask']], 1)

            if self.args.bypass_premise:
                tem=torch.cat([tem,input['input_emb_premises'].val],1)
                mask=torch.cat([torch.ones([input['sent'].shape[0], self.args.recovery_length]).to(input['mask'].device), input['mask_premises']], 1)
            if tem.shape[1]>512:
                tem=tem[:,0:512,:]
                mask=mask[:,0:512]
            if self.args.inconsistent_recorvery_length:
                clas = self.base_model(inputs_embeds=tem,labels=input['label'].repeat([sample_num, 1]))
            else:
                clas = self.base_model(inputs_embeds=tem, attention_mask=mask.repeat([sample_num,1]), labels=input['label'].repeat([sample_num,1]))

        else:

            tem = self.args.tokens_to_ids['[CLS]'] * torch.ones([input['sent'].shape[0], 1]).to(
                input['mask'].device).type(torch.long)
            tem = torch.cat([tem, input['sent']], 1)
            mask = torch.cat([torch.ones([input['sent'].shape[0], 1]).to(input['mask'].device), input['mask']], 1)
            clas = self.base_model(tem, attention_mask=mask, labels=input['label'])

        return clas

    def soft_radius(self,logits,label):
        pred = torch.softmax(logits, 1)
        acc = (pred.argmax(1) == label).float()
        pred_sorted, _ = pred.sort(1, descending=True)
        PA = pred_sorted[:, 0] * acc + pred_sorted[:, 1] * (1 - acc)
        PB = pred_sorted[:, 1] * acc + pred_sorted[:, 0] * (1 - acc)
        PA = torch.minimum(PA, self.args.soft_upper_bound * torch.ones_like(PA))
        PB = torch.maximum(PB, (1-self.args.soft_upper_bound) * torch.ones_like(PB))
        m = Normal(torch.tensor([0.0]).to(pred.device),
                   torch.tensor([1.0]).to(pred.device))
        radius = (m.icdf(PA) - m.icdf(PB)) * self.STD / 2
        return radius

    def forward(self,input,coeff,input_preprocess=True,idirect_nput_to_bert_by_sent=True):

        if 'sent' in input and input_preprocess:
            input = self.input_preprocess(input)
        tem = torch.tensor(0.0).to(input['label'].device)

        if self.args.only_classify_on_hidden:
            loss,accuracy,ibp_loss,radius=self.classify_on_hidden(input,input['label'])

            return tem, tem,tem, tem, tem, tem, ibp_loss,radius, tem, tem, tem, tem, tem, loss,accuracy
        else:
            loss_info=tem
            if self.args.direct_input_to_bert:
                if 'sent' in input and idirect_nput_to_bert_by_sent:

                    tem_input = self.args.tokens_to_ids['[CLS]'] * torch.ones([input['sent'].shape[0], 1]).to(
                        input['mask'].device).type(torch.long)
                    tem_input = torch.cat([tem_input, input['sent']], 1)
                    mask = torch.cat([torch.ones([input['sent'].shape[0], 1]).to(input['mask'].device), input['mask']], 1)
                    clas = self.base_model(tem_input, attention_mask=mask, labels=input['label'])
                else:
                    if isinstance(input['input_emb'],IntervalBoundedTensor):
                        new_emb=input['input_emb'].val
                    else:
                        new_emb = input['input_emb']
                    weight = self.bert_model.embeddings.word_embeddings.weight
                    tem_input = weight[self.args.tokens_to_ids['[CLS]'], :].unsqueeze(0).repeat(new_emb.shape[0], 1).unsqueeze(
                        1)
                    tem_input = torch.cat([tem_input, new_emb], 1)
                    mask = torch.cat([torch.ones([input['mask'].shape[0], 1]).to(input['mask'].device), input['mask']],
                                     1)  # 改过
                    if tem_input.shape[1] > 512:
                        tem_input = tem_input[:, 0:512, :]
                        mask = mask[:, 0:512]
                    clas = self.base_model(inputs_embeds=tem_input, attention_mask=mask,
                                           labels=input['label'])
                return clas.loss,correct_rate_func(clas.logits,input['label']),tem, tem, tem, tem, tem, tem, tem, tem, tem, tem, tem, tem, tem
            else:
                NLL_loss1, NLL_loss2, KL_loss, ibp_loss, recons1, recons2, z, mu, log_sigma,emb1,emb2,clas_hidden=self.forward_vae(input,coeff)

                if self.args.info_loss is not None:
                    if self.args.info_loss=='mean':
                        loss_info = (input['input_emb'].val.mean(1).detach() - emb2.mean(1)).abs().sum(1).mean(0)


                if self.args.classify_on_hidden:
                    loss_clas_hidden=self.NLL(clas_hidden, input['label']).mean()
                else:
                    loss_clas_hidden=torch.tensor(0.0).to(NLL_loss1.device)

                if not self.args.not_train_bert:
                    clas=self.classify_by_bert(input,recons1,recons2,emb1,emb2,coeff)

                    radius=self.soft_radius(clas.logits,input['label'])
                    loss_cls=clas.loss

                    return loss_cls,correct_rate_func(clas.logits,input['label']), loss_info, NLL_loss1, NLL_loss2, KL_loss, ibp_loss,radius, recons1, \
                           recons2, z, mu, log_sigma,loss_clas_hidden,correct_rate_func(clas_hidden,input['label'])

                else:

                    loss_cls=torch.tensor(0.0).to(NLL_loss1.device)
                    accu =torch.tensor(0.0).to(NLL_loss1.device)
                    radius=torch.tensor(0.0).to(NLL_loss1.device).repeat(ibp_loss.shape[0])
                    return loss_cls,accu, NLL_loss1,loss_info, NLL_loss2, KL_loss, ibp_loss, radius,recons1, recons2, z, mu, log_sigma,loss_clas_hidden,correct_rate_func(clas_hidden,input['label'])





    def certify_prediction(self,input,sample_num,alpha=0.05):

        input=self.input_preprocess(input)
        # first sampling
        z, mu, log_sigma, mu_ibp = self.encode(input, sample_num)
        input_decoder = torch.cat([self.args.tokens_to_ids['[PAD]'] * torch.ones([input['sent'].shape[0], 1])
                                  .to(input['sent'].device), input['sent'][:, 0:-1]], 1).type(torch.long)

        input_decoder = self.get_emb(input_decoder)
        input_decoder = input_decoder.repeat(sample_num, 1, 1)
        recons1, recons2, emb1, emb2, clas_hidden = self.decode(z, input_decoder)

        clas = self.classify_by_bert(input, recons1, recons2, emb1, emb2, 0, sample_num)

        count = torch.nn.functional.one_hot(clas.logits.argmax(dim=1).reshape([input['sent'].shape[0], sample_num]),num_classes=self.args.num_classes).sum(1)
        _,count_sort = count.sort(dim=1,descending=True)
        count_max=count_sort[:,0]
        cont_second=count_sort[:,1]
        count_1=count.gather(1,count_max.unsqueeze(1)).detach().cpu().numpy()
        count_2=count.gather(1,cont_second.unsqueeze(1)).detach().cpu().numpy()
        P=self.binomial_test(count_1,count_2,z.device,0.5)
        res = (P < alpha) * count_max + (P>alpha) * -1
        return res




    def certify(self,input,sample_num,sample_num_2,alpha=0.05):
        input = self.input_preprocess(input)
        # first sampling
        z, mu, log_sigma, mu_ibp = self.encode(input,sample_num)
        input_decoder = torch.cat([self.args.tokens_to_ids['[PAD]'] * torch.ones([input['sent'].shape[0], 1])
                                  .to(input['sent'].device), input['sent'][:, 0:-1]], 1).type(torch.long)

        input_decoder = self.get_emb(input_decoder)
        input_decoder=input_decoder.repeat(sample_num, 1,1)
        recons1, recons2, emb1, emb2, clas_hidden = self.decode(z, input_decoder)

        clas = self.classify_by_bert(input, recons1, recons2, emb1, emb2, 0,sample_num)

        if self.args.soft_verify:
            pred = torch.softmax(clas.logits, 1).reshape([sample_num,input['sent'].shape[0],self.args.num_classes]).sum(0)

            count_max=pred.argmax(dim=1)
        else:
            count=torch.nn.functional.one_hot(clas.logits.argmax(dim=1).reshape([sample_num,input['sent'].shape[0]]),self.args.num_classes).sum(0)
            pred = torch.softmax(clas.logits, 1).reshape(
                [sample_num, input['sent'].shape[0], self.args.num_classes]).sum(0)
            count_max=count.argmax(dim=1)

        #second sampleing
        assert sample_num_2>=300 and sample_num_2%300 ==0
        sample_num_each=300
        iter_num=sample_num_2//sample_num_each
        result=[]
        for _ in range(iter_num):
            z, mu, log_sigma, mu_ibp = self.encode(input,sample_num_each)
            input_decoder = torch.cat([self.args.tokens_to_ids['[PAD]'] * torch.ones([input['sent'].shape[0], 1])
                                      .to(input['sent'].device), input['sent'][:, 0:-1]], 1).type(torch.long)

            input_decoder = self.get_emb(input_decoder)
            input_decoder=input_decoder.repeat(sample_num_each,1, 1)
            recons1, recons2, emb1, emb2, clas_hidden = self.decode(z, input_decoder)

            clas = self.classify_by_bert(input, recons1, recons2, emb1, emb2,0,sample_num_each)
            result.append(clas.logits)
        result=torch.cat(result,dim=0)
        if self.args.soft_verify:
            count_soft=torch.softmax(self.args.soft_beta*result, 1).reshape([sample_num_2,input['sent'].shape[0],self.args.num_classes]).sum(0)
            sum_square=torch.softmax(self.args.soft_beta*result, 1).reshape([sample_num_2,input['sent'].shape[0],self.args.num_classes]).square().sum(0)
            P_A, radius = self.lower_confidence_bound_soft(count_soft.gather(1, count_max.unsqueeze(1)).round(), sample_num_2, alpha,sum_square.gather(1, count_max.unsqueeze(1)))
        else:
            count=torch.nn.functional.one_hot(result.argmax(dim=1).reshape([sample_num_2,input['sent'].shape[0]]),self.args.num_classes).sum(0)
            P_A,radius=self.lower_conf_bound(count.gather(1,count_max.unsqueeze(1)),sample_num_2,alpha)

        res=(P_A>0.5)*count_max+(P_A<0.5)*-1

        return res, radius, mu_ibp






    def input_pertubation(self,input,sample_num):

        input['sent']=input['sent'].repeat(sample_num,1)
        input['mask'] = input['mask'].repeat(sample_num,1)
        input['token_type_ids'] = input['token_type_ids'].repeat(sample_num,1)
        input['label'] = input['label'].repeat(sample_num)
        input["text_like_syn"]=input["text_like_syn"].repeat(sample_num,1,1)
        input["text_like_syn_valid"] = input["text_like_syn_valid"].repeat(sample_num, 1, 1)

        seed = torch.rand(input["text_like_syn"].shape).to(input["text_like_syn"].device)*input["text_like_syn_valid"]
        index=torch.argmax(seed,dim=2)
        sent=torch.gather(input["text_like_syn"],2,index.unsqueeze(-1)).squeeze(-1)

        if self.args.ibp_encode:
            mask = None
            ibp_input = None
            text_like_syn = input["text_like_syn"]
            text_like_syn_valid = input["text_like_syn_valid"]
            if "mask" in input:
                mask = input["mask"]

            if "ibp" in input:
                ibp_input = input["ibp_input"]

            if ibp_input == None:
                ibp_input = self.ibp_input_from_convex_hull(sent, text_like_syn, text_like_syn_valid, mask)
                input["ibp_input"] = ibp_input
            input_emb = ibp_input
        else:
            input_emb = self.get_emb(sent)

        input['input_emb'] = input_emb

        return input

    def lower_conf_bound(self,count,num,alpha=0.05):
        P_A= proportion_confint(count.detach().cpu().numpy(),num,alpha=2*alpha,method='beta')[0]
        radius = self.STD*norm.ppf(P_A)
        return torch.Tensor(P_A).to(count.device).squeeze(),torch.Tensor(radius).to(count.device).squeeze()

    def binomial_test(self,count_max,count_second,device,P=0.5):
        res=[]
        for i in range(count_max.shape[0]):
            res.append(binom_test(count_max[i],count_max[i]+count_second[i],P))
        return torch.Tensor(res).to(device).squeeze()





    def lower_confidence_bound_soft(self, NA, N, alpha, ss):
        NA=NA.detach().cpu().numpy()
        ss=ss.detach().cpu().numpy()
        sample_variance = (ss - NA * NA / N) / (N - 1)
        if sample_variance < 0:
            sample_variance = 0
        t = np.log(2 / alpha)
        P_A=NA / N - np.sqrt(2 * sample_variance * t / N) - 7 * t / 3 / (N - 1)
        radius = self.STD * norm.ppf(P_A)
        return  torch.Tensor(P_A).to(self.args.device[0]).squeeze(),torch.Tensor(radius).to(self.args.device[0]).squeeze()


    def ascc_certify(self, input, attack_type_dict):
        text_like_syn = input["text_like_syn"]
        text_like_syn_valid = input["text_like_syn_valid"]
        device = text_like_syn.device
        y = input["label"]
        emb_ibp = self.input_preprocess(input, convex_hull=False)['input_emb']

        num_steps = attack_type_dict['num_steps']
        loss_func = attack_type_dict['loss_func']
        w_optm_lr = attack_type_dict['w_optm_lr']
        sparse_weight = attack_type_dict['sparse_weight']
        out_type = attack_type_dict['out_type']

        syn, _ = self.build_convex_hull(text_like_syn, text_like_syn_valid)
        batch_size, text_len, embd_dim = emb_ibp.val.shape
        batch_size, text_len, syn_num, embd_dim = syn.shape

        w = torch.empty(batch_size, text_len, syn_num, 1).to(device).float()
        torch.nn.init.kaiming_normal_(w)
        w.requires_grad_()
        params = [w]
        optimizer = torch.optim.Adam(params, lr=w_optm_lr, weight_decay=2e-5)

        def get_comb_p(w, syn_valid):
            ww = w * syn_valid.reshape(batch_size, text_len, syn_num, 1) + 500 * (
                        syn_valid.reshape(batch_size, text_len, syn_num, 1) - 1)
            return F.softmax(ww, -2)

        def get_comb_ww(w, syn_valid):
            ww = w * syn_valid.reshape(batch_size, text_len, syn_num, 1) + 500 * (
                        syn_valid.reshape(batch_size, text_len, syn_num, 1) - 1)
            return ww

        def get_comb(p, syn):
            return (p * syn.detach()).sum(-2)

        input_ori = dict([])
        input_ori["input_emb"] = emb_ibp
        input_ori["mask"] = input["mask"].detach()
        input_ori["label"] = input["label"]

        for step in range(num_steps):
            optimizer.zero_grad()
            with torch.enable_grad():
                ww = get_comb_ww(w, text_like_syn_valid)
                embd_adv = get_comb(F.softmax(ww, -2), syn)
                embd_adv_ibp = IntervalBoundedTensor(embd_adv, embd_adv, embd_adv)
                input_ori["input_emb"] = embd_adv_ibp
                loss_cls, accu, _, _, _, _, _, _, _, _, _, _, _, _, _ = self.forward(input_ori, 1)
                loss = -loss_cls
                loss.backward()
                optimizer.step()

        ww_discrete = ww
        embd_adv = get_comb(F.softmax(ww * (1e10), -2), syn).detach()
        embd_adv_ibp = IntervalBoundedTensor(embd_adv, embd_adv, embd_adv)
        input_ori["input_emb"] = embd_adv_ibp

        return input_ori

    def emperical_certify(self,input):
        input = self.input_preprocess(input)
        input_per = self.input_pertubation(input,20)
        # first sampling
        z, mu, log_sigma, mu_ibp = self.encode(input)
        z_per, mu_per, log_sigma_per, mu_ibp_per = self.encode(input_per)
        ub = mu_per.max(0)[0]
        lb = mu_per.min(0)[0]
        ub_ibp=mu_ibp.ub
        lb_ibp=mu_ibp.lb
        ind1=ub_ibp[0,:]>=ub
        ind2=lb_ibp[0,:]<=lb
        bounded=ind1.prod()*ind2.prod()
        input_decoder = torch.cat([self.args.tokens_to_ids['[PAD]'] * torch.ones([input_per['sent'].shape[0], 1])
                                  .to(input_per['sent'].device), input_per['sent'][:, 0:-1]], 1).type(torch.long)

        input_decoder = self.get_emb(input_decoder)
        recons1, recons2, emb1, emb2, clas_hidden = self.decode(mu_per, input_decoder)

        clas = self.classify_by_bert(input_per, recons1, recons2, emb1, emb2, 0)

        return bounded.float(),correct_rate_func(clas.logits, input_per[
            'label'])
