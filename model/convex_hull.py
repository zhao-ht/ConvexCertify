import torch
import torch.nn.functional as F

class ConvexHull:
    def __init__(self, nodes_emb, sample_valid): 
        # help: nodes [sample, nodes for each sample, emb_dim], sample size = batch * max_len
        self.nodes_emb = nodes_emb
        self.device = nodes_emb.device
        self.s, self.n, self.e = nodes_emb.shape
        self.valid_mask = sample_valid.reshape(self.s, self.n, 1)
        
        #self.w = torch.empty(num_samples, num_nodes, 1).to(device).to(nodes_emb.dtype)
    
    
    def masked_softmax(self, vec, mask, dim = 1):
        masked_vec = vec * mask.float()
        max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
        exps = torch.exp(masked_vec-max_vec)
        masked_exps = exps * mask.float()
        masked_sums = masked_exps.sum(dim, keepdim=True)
        zeros=(masked_sums == 0)
        masked_sums += zeros.float()
        return masked_exps/masked_sums
    
    def sampler(self, max_sample_num = 50, mode='softmax_sampler'): 
        # help: sample max_sample_num random samples from convex hull
        if mode == 'softmax_sampler':
            self.w = torch.rand(self.s, self.n, max_sample_num).to(self.device)
            #self.w = self.valid_mask * self.w
            self.ww = self.masked_softmax(self.w, self.valid_mask, -2).reshape(self.s, self.n, max_sample_num,1)  # help: make sure the sum of weight is 1
            samples = (self.ww * self.nodes_emb.detach().reshape(self.s, self.n, 1, self.e)).sum(dim = 1)
        
        return samples
