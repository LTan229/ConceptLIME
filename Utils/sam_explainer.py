from __future__ import print_function, division
import torch
import numpy as np

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim


def gen_concept_masks(gen_model,target_img):
    return gen_model.generate(target_img)


def feat_prob(model,feat,target_label):
    
    with torch.no_grad():
        if model == None:
            probabilities = torch.nn.functional.softmax(feat,dim=1)
        else:
            output = model(feat)
            probabilities = torch.nn.functional.softmax(output,dim=1)
        return probabilities[:,target_label]
    

class NeuralNet(nn.Module): ### a simple NN network
    def __init__(self, in_size,bs,head,lr,ftdim):
        super(NeuralNet, self).__init__()
        torch.manual_seed(0)
        self.model = nn.Sequential(nn.Linear(in_size,ftdim))
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        self.loss = torch.nn.BCELoss()
        self.bs = bs
        self.fc = head

    def change_lr(self,lr):
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

    def forward(self,x):
        if self.fc != None:
            return self.fc(self.model(x))
        else:
            return self.model(x)
    
    def forward_feat(self,x):
        return self.model(x)
    
    def step(self, x,y):
        self.optimizer.zero_grad()
        loss = self.loss
        output = loss(x, y)
        output.backward()
        self.optimizer.step()
        return output.detach().cpu().numpy()
    
    def step_val(self, x,y):
        self.optimizer.zero_grad()
        loss = self.loss
        output = loss(x, y)
        return output.detach().cpu().numpy()


def learning_feat(target_model,full_model,concept_mask,target_img,target_label,fc,image_norm=None):
    target_img = np.expand_dims(target_img,0)
    concept_mask = np.expand_dims(concept_mask,3)
    masked_image = target_img*concept_mask
    batch_img = []
    for i in range(masked_image.shape[0]):
        input_tensor = image_norm(masked_image[i])
        batch_img.append((input_tensor.cpu()))
    tmp_dl = DataLoader(dataset = batch_img, batch_size = 500, shuffle =False)
    
    output = None
    fc_res = None
    for x in tmp_dl:
        with torch.no_grad():
            if output == None:
                output = target_model(x.cuda())
                output = output['avgpool'].squeeze().squeeze()
                fc_res = torch.nn.functional.softmax(fc(output), dim=1)[:,target_label]
            else:
                tmp_out = target_model(x.cuda())
                tmp_out = tmp_out['avgpool'].squeeze().squeeze()
                output = torch.cat((output,tmp_out))
                fc_res = torch.cat((fc_res,torch.nn.functional.softmax(fc(tmp_out), dim=1)[:,target_label]))
        
    return output, fc_res

    
def learn_PIE(target_model,full_model,concept_mask,target_img,target_label,fc,lr,epochs,image_norm=None):
    masks_tmp = concept_mask.copy()
    
    num_feat = masks_tmp.shape[0]
    only_feat = np.zeros((num_feat, num_feat), int)
    np.fill_diagonal(only_feat,1)
    bin_x = np.random.binomial(1,0.5,size=(2500,num_feat)).astype(bool) #### generate 2500 samples to learn PIE by default
    new_mask = np.array([masks_tmp[i].sum(0) for i in bin_x]).astype(bool)
    feat, probs = learning_feat(target_model,full_model,new_mask,target_img,target_label,fc,image_norm)
    feat = feat.detach().clone().cpu()
    probs = probs.detach().clone().cpu()
    bin_x_torch = torch.tensor(bin_x.tolist(),dtype=torch.float)
    data = [[x,y] for x,y in zip(bin_x_torch,probs)]
    bs = 100
    losses = []
    
    net = NeuralNet(num_feat,bs,fc,lr,feat.shape[1]).cuda()
    
    net.change_lr(lr)
    data_comb_train = DataLoader(dataset = data[num_feat:], batch_size = bs, shuffle =True)
    
    ##### learning combin
    for epoch in range(50):
        loss = 0
        for x,y in data_comb_train:
            pred = torch.nn.functional.softmax(net(x.cuda()), dim=1)[:,target_label]
            loss += net.step(pred,y.cuda())*x.shape[0]
    
    net.eval()
    return feat, probs,losses,net,bin_x_torch



### model: the target endtoend model you wish to explain
### img_numpy: image in numpy version, the same file you pass to the sam
### image_class: the class id you wish to explain
### concept_masks: concept_masks gen by sam
### fc: the fully-connect layer in the target endtoend model
### feat_exp: the feature extrator in the target endtoend model
### image_norm: transform the img_numpy into torch version with the target model normlize

def samshap(model,img_numpy,image_class,concept_masks,fc,feat_exp,image_norm=None,lr=0.008):
    
    
    feat, probs,losses,net,bin_x_torch = learn_PIE(feat_exp,model,concept_masks,img_numpy,image_class,fc,lr=lr,epochs=100,image_norm=image_norm) ### learning the PIE module 
    feat_num = len(concept_masks)
    shap_val = []
    mc = 50000
    
    
    for i in range(feat_num):
        bin_x_tmp = np.random.binomial(1,0.5,size=(mc,feat_num)) ### mc shapley computing 
        bin_x_tmp_sec = bin_x_tmp.copy()

        bin_x_tmp[:,i] = 1
        bin_x_tmp_sec[:,i] = 0

        bin_x_tmp = torch.from_numpy(bin_x_tmp).type(torch.float)
        bin_x_tmp_sec = torch.from_numpy(bin_x_tmp_sec).type(torch.float)

        pre_shap = (feat_prob(fc,net.forward_feat(bin_x_tmp.cuda()),image_class) - feat_prob(fc,net.forward_feat(bin_x_tmp_sec.cuda()),image_class)).detach().cpu().numpy()

        shap_val.append(pre_shap.sum()/mc)
    ans = shap_val.index(max(shap_val))
    shap_list = shap_val

    shap_list = np.array(shap_list)
    shap_arg = np.argsort(-shap_list)
    auc_mask = np.expand_dims(np.array([concept_masks[shap_arg[:i+1]].sum(0) for i in range(len(shap_arg))]).astype(bool),3)


    return auc_mask, shap_list