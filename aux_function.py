from __future__ import print_function

import numpy as np

import os
import torch
import torch.nn.parallel

import torch.utils.data

from torch import nn

class DataGather(object):
    def __init__(self, keys,options,save_path=None):
        self.keys = keys
        self.data = self.get_empty_data_dict()
        self.options=options
        assert len(self.keys)==len(self.options)

        self.save_path= save_path
        if save_path is not None:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            self.save_path=os.path.join(save_path,'log.txt')

    def get_empty_data_dict(self):
        dic={}
        for key in self.keys:
            dic[key]=[]
        return dic

    def insert(self, keys,data):
        assert len(keys)==len(data)
        for i in range(len(keys)):
            if isinstance(data[i],torch.Tensor):
                tem=data[i].item()
            else:
                # assert isinstance(data[i],float)
                tem=data[i]
            self.data[keys[i]].append(tem)

    def flush(self):
        self.data = self.get_empty_data_dict()

    def get_mean(self):
        res=[]
        for key in self.keys:
            if len(self.data[key])>0:
                res.append(np.mean(self.data[key]))
            else:
                res.append(0)
        return res

    def get_min(self):
        res = []
        for key in self.keys:
            if len(self.data[key])>0:
                res.append(np.min(self.data[key]))
            else:
                res.append(None)
        return res

    def get_max(self):
        res = []
        for key in self.keys:
            if len(self.data[key])>0:
                res.append(np.max(self.data[key]))
            else:
                res.append(None)
        return res

    def get_sum(self):
        res=[]
        for key in self.keys:
            if len(self.data[key])>0:
                res.append(np.sum(self.data[key]))
            else:
                res.append(0)
        return res

    def get_report(self,):
        mins=self.get_min()
        means=self.get_mean()
        maxs=self.get_max()
        res=[]
        for i in range(len(mins)):
            res.append([mins[i],maxs[i],means[i]])
        res1=[]
        for i in range(len(mins)):
            res1.append(res[i][self.options[i]])
        return res1

    def report(self,additional=None):
        res=self.get_report()
        string=''
        for ind,key in enumerate(self.keys):
            string=string+str(key)+' '
            string=string+str(res[ind])+' '
        if additional is not None:
            string=str(additional)+' '+string
        print(string)
        if self.save_path is not None:
            with open(self.save_path, 'a+') as f:
                f.write(string+'\n')
    def write(self,string):
        print(string)
        if self.save_path is not None:
            with open(self.save_path, 'a+') as f:
                f.write(str(string)+'\n')



#
# def infinit_iter(loader):
#     while 1:
#         for j in loader:
#             yield j
#
# def check_nan(names,data_lis,tb_step,args,break_flag=True):
#     if args['check_nan_flags']:
#         for i  in range(len(data_lis)):
#             if isinstance(data_lis[i],torch.Tensor):
#                 if torch.any(torch.isnan(data_lis[i])):
#                     print(names[i]+' is nan at step '+str(tb_step))
#                     if break_flag:
#                         raise
#                     return True
#             elif np.any(np.isnan(data_lis[i])):
#                 print(names[i] + ' is nan at step ' + str(tb_step))
#                 if break_flag:
#                     raise
#                 return True
#
# def check_nan_model(model,tb_step,args,break_flag=True):
#     if args['check_nan_flags']:
#         for name, parms in model.named_parameters():
#             check_nan([name],[parms],tb_step,break_flag)
#             if not (parms.grad is None):
#                 check_nan([name+' grad'],[parms],tb_step,break_flag)
#
#
# def multi_tb_writer(writer,names,datas,tb_step,prefix=''):
#     assert len(names)==len(datas)
#     for i in range(len(names)):
#         writer.add_scalar(prefix+'/'+names[i],datas[i],tb_step)
#
# def multi_tb_writer_hist(writer,names,datas,tb_step,prefix=''):
#     assert len(names) == len(datas)
#     for i in range(len(datas)):
#         writer.add_histogram(prefix+'/'+names[i],datas[i],tb_step)
#
# def model_params_tb_writer(writer,model,tb_step,prefix=''):
#     for name, parms in model.named_parameters():
#         try:
#             writer.add_histogram(prefix+'_parms/' + name, parms, tb_step)
#         except:
#             pass
#         if not (parms.grad is None):
#             writer.add_scalar(prefix+'_grad/' + name, torch.norm(parms.grad), tb_step)
#
#
# def im_writer(writer,images,tb_step,args,prefix='',normalization=False):
#     assert images.shape[0]>=args['figure_num']
#     if images.dim()==2:
#         images=images.view(-1,args['channel_input'],args['imagesize'],args['imagesize'])
#     if normalization:
#         images=(images-images[:].min())/(images[:].max()-images[:].min())
#     img_grid = torchvision.utils.make_grid(images[0:args['figure_num'],:,:,:],args['figure_each_row'])
#     writer.add_image(prefix+'/'+str(tb_step),img_grid)
#
# def plot_matrix(cm, class_names='',add_value=False):
#
#     figure = plt.figure()
#     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
#     plt.colorbar()
#     if len(class_names)>0:
#         tick_marks = np.arange(len(class_names))
#         plt.xticks(tick_marks, class_names, rotation=45)
#         plt.yticks(tick_marks, class_names)
#     if add_value:
#         labels = np.around(cm.astype('float'),decimals=2)
#         threshold = cm.max() / 2.
#         for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#             color = "white" if cm[i, j] > threshold else "black"
#             plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)
#
#     plt.show()
#
# def plot_embedding(X, title=None, y=None):
#     x_min, x_max = np.min(X, 0), np.max(X, 0)
#     X = (X - x_min) / (x_max - x_min)
#
#     plt.figure()
#     ax = plt.subplot(111)
#     if y is None:
#         plt.scatter(X[:, 0], X[:, 1], alpha=0.3
#                     )
#     else:
#         for i in range(X.shape[0]):
#             plt.scatter(X[i, 0], X[i, 1],
#                         color=plt.cm.Set1(y[i] / 10.), alpha=0.3
#                         )
#
#     plt.xticks([]), plt.yticks([])
#     if title is not None:
#         plt.title(title)
#     plt.show()
#
# def plot_matrix_to_tensor(cm, class_names='',add_value=False):
#
#     figure = plt.figure()
#     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
#     plt.colorbar()
#     if len(class_names)>0:
#         tick_marks = np.arange(len(class_names))
#         plt.xticks(tick_marks, class_names, rotation=45)
#         plt.yticks(tick_marks, class_names)
#     if add_value:
#         labels = np.around(cm.astype('float'),decimals=2)
#         threshold = cm.max() / 2.
#         for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#             color = "white" if cm[i, j] > threshold else "black"
#             plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)
#
#     buf = io.BytesIO()
#     plt.savefig(buf, format='jpeg')
#     plt.close()
#     buf.seek(0)
#     image = PIL.Image.open(buf)
#     image = ToTensor()(image)
#     return image
#
#
# # def plot_to_image(figure):
# #   buf = io.BytesIO()
# #   plt.savefig(buf, format='png')
# #   plt.close(figure)
# #   buf.seek(0)
# #   image = tf.image.decode_png(buf.getvalue(), channels=4)
# #   image = tf.expand_dims(image, 0)
# #   return image.numpy()
#
#
#
# def matrix_writer(writer,matrixs,tb_step,prefix=''):
#     ind=0
#     if isinstance(matrixs,torch.Tensor) and matrixs.dim==3:
#         matrixs=[to_numpy(matrixs[i]) for i in range(matrixs.shape[0])]
#     for matrix in matrixs:
#         if isinstance(matrix, torch.Tensor):
#             matrix=to_numpy(matrix)
#         image = plot_matrix_to_tensor(matrix)
#         writer.add_image(prefix+'/'+str(ind)+'/'+str(tb_step),image)
#         ind=ind+1
#
#
# def permute_dims(z_list):
#     assert z_list[0].dim() == 2
#     res=[]
#     for z in z_list:
#         B, _ = z.size()
#
#         perm_z = []
#         for z_j in z.split(1, 1):
#             perm = torch.randperm(B).to(z.device)
#             perm_z_j = z_j[perm].detach()
#             perm_z.append(perm_z_j)
#         perm_z=torch.cat(perm_z, 1)
#         res.append(perm_z)
#     return res
#
#
#
#
# def save_model(model_dic,args,epoch):
#     model_state_dic={}
#     for key in model_dic.keys():
#         if not 'tb_step' in key:
#             model_state_dic[key]=model_dic[key].state_dict()
#         else:
#             model_state_dic[key] = model_dic[key]
#     if not os.path.exists(args['save_path']):
#         os.mkdir(args['save_path'])
#     print('saveing to {}'.format(args['save_path'] + '/Epoch_' + str(epoch) + '.pth'))
#     torch.save(model_state_dic, args['save_path'] + '/Epoch_' + str(epoch) + '.pth')
#
#
# def load_model(model_dic,tb_step,args):
#     print('loading from {}'.format(args['reload_path'] + '/Epoch_' + str(args['reload_from']) + '.pth'))
#     checkpoint=torch.load(args['reload_path'] + '/Epoch_' + str(args['reload_from']) + '.pth')
#     for key in model_dic.keys():
#         if not 'tb_step' in key:
#             model_dic[key].load_state_dict(checkpoint[key])
#         else:
#             tb_step=checkpoint[key]
#     return tb_step
#
#
# def specified_load(model_dic,tb_step,model_path):
#     print('loading from {}'.format(model_path))
#     checkpoint = torch.load(model_path)
#     for key in model_dic.keys():
#         if not 'tb_step' in key:
#             model_dic[key].load_state_dict(checkpoint[key])
#         else:
#             tb_step=checkpoint[key]
#     return tb_step
#
#
# def kaiming_init(m):
#     if isinstance(m, (nn.Linear, nn.Conv2d)):
#         torch.nn.init.kaiming_normal_(m.weight)
#         if m.bias is not None:
#             m.bias.data.fill_(0)
#     elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
#         m.weight.data.fill_(1)
#         if m.bias is not None:
#             m.bias.data.fill_(0)
#
#
# def normal_init(m):
#     if isinstance(m, (nn.Linear, nn.Conv2d)):
#         torch.nn.init.normal_(m.weight, 0, 0.02)
#         if m.bias is not None:
#             m.bias.data.fill_(0)
#     elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
#         m.weight.data.fill_(1)
#         if m.bias is not None:
#             m.bias.data.fill_(0)
#
#
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         torch.nn.init.normal_(m.weight, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         torch.nn.init.normal_(m.weight, 1.0, 0.02)
#         torch.nn.init.zeros_(m.bias)
#
#
#
# class Add_noise(nn.Module):
#     def __init__(self,mean=0,std=1.0):
#         self.mean=mean
#         self.std=std
#         super(Add_noise,self).__init__()
#
#     def forward(self,input,device):
#         assert isinstance(input,torch.Tensor)
#         res=input+torch.randn(size=input.shape).to(device)
#         return res
#
# def recon_loss(x_recon,x):
#
#     assert isinstance(x_recon, torch.Tensor)
#     assert x.shape == x_recon.shape
#     assert x.dim() == 4 and x_recon.dim() == 4
#
#
#     loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='none')
#     assert loss.dim()==4
#
#     loss=loss.sum([1,2,3]).mean()
#
#     return loss
#
#
# def kl_divergence(mu, logvar):
#     assert mu.dim()==4 and logvar.dim()==4
#     assert mu.shape==logvar.shape
#
#     kld = -0.5*(1+logvar-mu**2-logvar.exp()).sum([1,2,3]).mean()
#
#     return kld
#
# def kl_divergence_hard(mu, logvar):
#     assert mu.shape == logvar.shape
#     if mu.dim()==4 and logvar.dim()==4:
#         kld = -0.5*(-mu**2).sum([1,2,3]).mean()
#     elif mu.dim()==2 and logvar.dim()==2:
#         kld = -0.5 * (-mu ** 2).sum([1]).mean()
#     return kld
#
#
def correct_rate_func(out,label):
    return (torch.argmax(out,1)==label).float().mean()

class DataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class DistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


