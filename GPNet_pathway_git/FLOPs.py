#%%
import torch
import torch.nn as nn
import torchprofile


act_fn = nn.LeakyReLU(0.2)
dropout_fn = nn.Dropout(0.5)
def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.fill_(0.01)
class pathwayblock(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        if hidden_dim == 0:
            self.block = nn.Sequential(
                nn.Linear(input_dim, 1, bias = False),
                act_fn,
                dropout_fn
            )
        else: #layer_num = btw pathway and biological factor number. if layer_num ==0, then hidden num should be zero for setting. we do not allow =0
            modules = []
            modules.append(nn.Linear(input_dim, hidden_dim, bias = False))
            modules.append(act_fn)
            modules.append(dropout_fn)
            for i in range(layer_num-1):
                modules.append(nn.Linear(hidden_dim, hidden_dim, bias = False))
                modules.append(act_fn)
                modules.append(dropout_fn)
            modules.append(nn.Linear(hidden_dim, 1, bias = False))
            modules.append(act_fn)
            modules.append(dropout_fn)
            self.block = nn.Sequential(*modules)
        self.block.apply(init_weights)

    def forward(self, x):
        return self.block(x)



class DeepHisCoM(nn.Module):
    def __init__(self,  nvar, width, layer, covariate, device):
        super(DeepHisCoM, self).__init__()
        self.nvar = nvar
        self.width = width
        self.layer = layer 
        self.pathway_nn = nn.ModuleList([pathwayblock(nvar[i], width[i], layer[i]) for i in range(len(self.nvar))])
        self.bn_path = nn.BatchNorm1d(len(nvar))
        self.dropout_path = dropout_fn
        self.covariate = covariate
        self.fc_path_disease=nn.Linear(len(nvar) +covariate ,1)
        self.fc_path_disease.weight.data.fill_(0)
        self.fc_path_disease.bias.data.fill_(0.001)
        self.device = device

    def forward(self, x):
        kk=0
        nvarlist = list()
        nvarlist.append(kk)
        for i in range(len(self.nvar)):
            k=kk
            kk=kk+self.nvar[i]
            nvarlist.append(kk)
        nvarlist.append(kk + self.covariate)
        pathway_layer = torch.cat([self.pathway_nn[i](x[:,nvarlist[i]:nvarlist[i+1]]) for i in range(len(self.nvar))],1)
        pathway_layer = self.bn_path(pathway_layer)
        pathway_layer = pathway_layer/(torch.norm(pathway_layer,2))
        x = torch.cat([pathway_layer, x[:, nvarlist[len(self.nvar)]:nvarlist[len(self.nvar) + 1]]], 1)
        x = self.dropout_path(x)
        x = self.fc_path_disease(x)
        x = torch.sigmoid(x)
        return(x)
    
nvar = [200]*50
node_num = [50]*50
layer_num = [1]*50
input_dim = 10000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_DeepHisCoM = DeepHisCoM(nvar, node_num, layer_num, 0, device)
# Dummy input for batch size 1
input_tensor = torch.randn(64, input_dim)

# Profile the model
flops = torchprofile.profile_macs(model_DeepHisCoM, input_tensor)

# 1000 permutation tests + 1 main test
gflops = flops* 1001 / 1e9  # Convert to GFLOPs 
print(f"GFLOPs: {gflops:.6f}")
# %%
from utls.dataloader import load_data
from utls.models import *
from utls.train import *
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
from sklearn.metrics import confusion_matrix
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import argparse
from torch.distributed import all_reduce, ReduceOp
from utls.build_mask import get_mask
import pickle
from utls.cfg import cfg
import numpy as np
import json
from datetime import datetime

# data_dir = f"/isilon/datalake/cialab/original/cialab/image_database/d00154/Tumor_gene_counts/All_countings/training_data_17_tumors_31_classes.csv"
# data_dir = f"/isilon/datalake/cialab/original/cialab/image_database/d00154/Tumor_gene_counts/training_data_6_tumors.csv"
data_dir = '/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project1_GM/codes/covid_dengue/combined_gene_data_after_tmm_log1p.csv'
file_label_path = '/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project1_GM/codes/covid_dengue/combined_labels.pkl'
pathway_file = '/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project1_GM/codes/covid_dengue/GPNet/immune.gmt'
outf = "/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project1_GM/codes/covid_dengue/GPNet_rebuild/model_saves"
outrf = "/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project1_GM/codes/covid_dengue/GPNet_rebuild/results"
desired_labels = cfg['desired_labels']
batch_size = cfg['batch_size']
max_epoch = cfg['max_epoch']
snet_flag = cfg['snet_flag']
tnet_flag = cfg['tnet_flag']
feature_transform = cfg['feature_transform']
eval_interval = cfg['eval_interval']
atention_pooling_flag = cfg['atention_pooling_flag']
encoder_flag = cfg['encoder_flag']

gene_space_dim = cfg['gene_space_dim']
LOSS_SELECT = cfg['LOSS_SELECT'] 
WEIGHT_LOSS_FLAG = cfg['WEIGHT_LOSS_FLAG']
MULTI_GPU_FLAG = cfg['Multi_gpu_flag']
pre_trained = cfg['pre_trained']
lr=cfg['lr']
K = cfg['K_FOLD']
gene_number_name_mapping,feature_num, train_valid_list, test_loader = load_data(cfg, file_path=data_dir,file_label_path=file_label_path)
masks = get_mask(pathway_file, gene_number_name_mapping)
masks = masks[:cfg['mask_len']]

print("input dim:", len(gene_number_name_mapping.keys()))
input_gene_num = 10000
class_num = len(desired_labels)
print("class_num:", class_num)
samples_per_class = [feature_num[i] for i in range(len(feature_num))]
# Calculate class weights
total_samples = sum(samples_per_class)
class_weights = [total_samples / samples_per_class[i] for i in range(len(samples_per_class))]

# Normalize weights so that their sum equals the number of classes
weight_sum = sum(class_weights)
normalized_weights = torch.tensor([w / weight_sum * len(feature_num) for w in class_weights])
print("Normalized Class Weights:", normalized_weights)
model = PointNetCls(gene_idx_dim = 2, 
                            gene_space_num = gene_space_dim, 
                            class_num=class_num, 
                            input_gene_num = input_dim,
                            snet_flag = snet_flag,
                            tnet_flag = tnet_flag, 
                            feature_transform=feature_transform, 
                            atention_pooling_flag = atention_pooling_flag,
                            encoder_flag = encoder_flag)

features1_count = torch.randn(64,1, input_dim)
features2_gene_idx = torch.randn(64,2, input_dim)
# Profile the model
flops = torchprofile.profile_macs(model, (features1_count, features2_gene_idx))

gflops = flops / 1e9  # Convert to GFLOPs 
print(f"GFLOPs: {gflops:.6f}")
# %%
output_dim = 2
class PASNet(nn.Module):
	def __init__(self, In_Nodes, Out_Nodes, Pathway_Nodes=5535, Hidden_Nodes=5000):
		super(PASNet, self).__init__()
		self.sigmoid = nn.Sigmoid()
		self.softmax = nn.Softmax(dim = 1)

		###gene layer --> pathway layer
		self.sc1 = nn.Linear(In_Nodes, Pathway_Nodes)
		###pathway layer --> hidden layer
		self.sc2 = nn.Linear(Pathway_Nodes, Hidden_Nodes)
		###hidden layer --> Output layer
		self.sc3 = nn.Linear(Hidden_Nodes, Out_Nodes)
		###randomly select a small sub-network
		self.do_m1 = torch.ones(Pathway_Nodes)
		self.do_m2 = torch.ones(Hidden_Nodes)
		###if gpu is being used
		# if torch.cuda.is_available():
		# 	self.do_m1 = self.do_m1.cuda()
		# 	self.do_m2 = self.do_m2.cuda()
		###

	def forward(self, x):
		###force the connections between gene layer and pathway layer w.r.t. 'pathway_mask'
		self.sc1.weight.data = self.sc1.weight.data
		x = self.sigmoid(self.sc1(x))
		if self.training == True: ###construct a small sub-network for training only
			x = x.mul(self.do_m1)
		x = self.sigmoid(self.sc2(x))
		if self.training == True: ###construct a small sub-network for training only
			x = x.mul(self.do_m2)
		x = self.softmax(self.sc3(x)) # all rows add up to 1

		return x
     
model_DeepHisCoM = PASNet(input_dim, output_dim)
# model = model.cuda()
# Dummy input for batch size 1
input_tensor = torch.randn(64, input_dim)

# Profile the model
flops = torchprofile.profile_macs(model_DeepHisCoM, input_tensor)

# 1000 permutation tests + 1 main test
gflops = flops / 1e9  # Convert to GFLOPs 
print(f"GFLOPs: {gflops:.6f}")
# %%
