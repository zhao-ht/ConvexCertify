import torch
import torch.nn as nn

import sys
from transformers import BertForSequenceClassification
sys.path.append('..')
from model.convex_hull import ConvexHull
from model.certify_model import Certify_CNN_1D
from utils import IntervalBoundedTensor
from model.ibp import max_diff_norm

class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.model_name = 'Base_model'
        self.args = args
        if self.args.base_model == 'bert':
            # bert def and args
            self.base_model = BertForSequenceClassification.from_pretrained(args.pretrained_name,num_labels=args.num_classes, return_dict=True)
            self.bert_model = self.base_model.bert
            self.embedding_encoder = self.bert_model.embeddings.word_embeddings
            self.position_encoder = self.bert_model.embeddings.position_embeddings
            self.position_ids = self.bert_model.embeddings.position_ids 
            self.token_type_encoder = self.bert_model.embeddings.token_type_embeddings
            self.embedding_layer_norm = self.bert_model.embeddings.LayerNorm
            self.embedding_dropout = self.bert_model.embeddings.dropout
            self.args.embedding_dim = 768
        else:
            self.embedding_encoder = nn.Embedding(args.vocab_size, args.embedding_dim)
            if self.args.use_pretrained_embed:
                self.embedding_encoder.weight=nn.Parameter(args.embeddings,requires_grad=args.embedding_training)
        
        self.properties = {"model_name":self.__class__.__name__,   # help: more could be added in new classes
                "embedding_dim":self.args.embedding_dim,
                "embedding_training":self.args.embedding_training,
                "max_seq_len":self.args.max_len,
                "batch_size":self.args.batch_size,
                "learning_rate":self.args.learning_rate,
                "dropout":self.args.dropout,
                }

        self.certify_model = Certify_CNN_1D(args.embedding_dim)
        
        if self.args.vocab_size > 0:
            self.update_embedding()
        
    def update_embedding(self):
        tokens_to_ids = self.args.tokens_to_ids 
        ids_to_tokens = self.args.ids_to_tokens 
    
        bert_base_tokens_to_ids = self.args.bert_base_tokens_to_ids
        bert_base_ids_to_tokens = self.args.bert_base_ids_to_tokens
    
        vocab_size = len(bert_base_tokens_to_ids)
        #assert vocab_size == len(tokens_to_ids)
        
        embedding_encoder = nn.Embedding(vocab_size, self.args.embedding_dim)
        for i in range(vocab_size):
            old_id = bert_base_tokens_to_ids[bert_base_ids_to_tokens[i]]
            embedding_encoder.weight.data[i] = self.embedding_encoder.weight.data[old_id]
            #embedding_encoder.weight[i].requires_grad = True

        self.embedding_encoder = embedding_encoder
        
        return
    
    def get_properties(self):
        print(self.properties)
        return self.properties
    
    def get_emb(self, tokens):
        if self.args.embedding_training:
            return self.embedding_encoder(tokens)
        else:
            return self.embedding_encoder(tokens).detach()
    
    
    # don't need ,just input into bert_model with inputs_embeds = inputs_embeds
    '''def bert_emb_transform(self, input_embeds): 
        input_shape = input_embeds.size()[:-1]
        print(input_shape)
        seq_length = input_shape[1]
        position_ids = self.position_ids[:, : seq_length].to(self.args.device)
        token_type_ids = torch.zeros(input_shape,dtype=torch.long, device=self.args.device)
        
        position_embeddings = self.position_encoder(position_ids)
        token_type_embeddings = self.token_type_encoder(token_type_ids)
        embeddings = input_embeds + position_embeddings + token_type_embeddings
        embeddings = self.embedding_layer_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)

        return embeddings
    '''
        
    def build_convex_hull(self, text_like_syn, syn_valid): # help: build convex hull for input, origin text is contained in synonyms
        n,l,s = text_like_syn.shape
        text_like_syn_embd = self.get_emb(text_like_syn.reshape(n,l*s)).reshape(n,l,s,-1) # help: temporary input doesn't include mask, token_ids
        valid_mask = syn_valid.unsqueeze(-1).repeat(1, 1, 1, text_like_syn_embd.size()[-1])
        text_like_syn_embd = text_like_syn_embd * valid_mask
        node_embd = text_like_syn_embd.reshape(n*l, s, -1)
        node_valid = syn_valid.reshape(n*l,s)
        return text_like_syn_embd, ConvexHull(node_embd, node_valid)
    
    def sample_from_convex_hull(self, sent, text_like_syn, text_like_syn_valid, max_sample_num = 2):
        _, convex_hull = self.build_convex_hull(text_like_syn, text_like_syn_valid)
        n,l,s = text_like_syn.shape
        para_sent = convex_hull.sampler(max_sample_num = max_sample_num).reshape(n,l,-1,self.args.embedding_dim).transpose(1,2)
        return para_sent

    def ibp_input_from_convex_hull(self, sent, text_like_syn, text_like_syn_valid, mask, eps=1.0):
        if text_like_syn is not None:
            text_like_syn_embd, _ = self.build_convex_hull(text_like_syn, text_like_syn_valid) # [n, l, s, d], mask [n, l]
            n, l, s, d = text_like_syn_embd.shape
            tmp_mask = (mask.unsqueeze(-1)).unsqueeze(-1)  
            tmp_mask = tmp_mask.repeat(1,1,s,d)
            
            syn_mask = text_like_syn_valid.unsqueeze(-1)
            syn_mask = syn_mask.repeat(1,1,1,d)
            tmp_mask = tmp_mask * syn_mask
            
            bound_temp = text_like_syn_embd * tmp_mask
            
            reverse_mask = 1 - tmp_mask
            origin_emb = text_like_syn_embd[:,:,0,:].unsqueeze(dim = -2)
            broadcast_origin = origin_emb.repeat(1, 1, s, 1)
            broadcast_origin = broadcast_origin * reverse_mask
            
            bound_temp = bound_temp + broadcast_origin
            
            ub = torch.max(bound_temp, dim = -2).values  # [n, l, d] # note the origin sentence is contained in convex hull
            lb = torch.min(bound_temp, dim = -2).values # [n, l, d]
            val = self.get_emb(sent)  # [n, l ,d]
            if eps != 1.0:
                lb = val - (val - lb) * eps
                ub = val + (ub - val) * eps
            return IntervalBoundedTensor(val, lb, ub)
        else:
            val = self.get_emb(sent)
            return IntervalBoundedTensor(val, val, val)
        
        
        


    def forward(self, input): # hope there are no bugs here...
        # only a demonstration how to use functions !!!!!!!
        mask = None
        ibp_input = None
        sent = input["sent"]
        text_like_syn = input["text_like_syn"]
        text_like_syn_valid = input["text_like_syn_valid"]
        if "mask" in input:
            mask = input["mask"]
        if "ibp" in input:
            ibp_input = input["ibp_input"]
            
        if ibp_input == None:
            ibp_input = self.ibp_input_from_convex_hull(sent, text_like_syn, text_like_syn_valid, mask)
            input["ibp_input"] = ibp_input
        output = self.certify_model(input)
        output_max = max_diff_norm(output)
        #pert_sent = self.sample_from_convex_hull(sent, text_like_syn, text_like_syn_valid)
        #sent_emb = self.get_emb(sent)
        return output_max
