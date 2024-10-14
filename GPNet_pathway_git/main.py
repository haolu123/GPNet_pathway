#%%
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
#%%
# def parse_args():
#     parser = argparse.ArgumentParser(description='Model Training')
#     # parser.add_argument("--batch_size", default=2, type=int, help="number of batch size")
#     # parser.add_argument("--cache_dataset", action="store_true", help="use monai cache Dataset")
#     # parser.add_argument("--distributed", action="store_true", help="use monai distributed training")
#     # parser.add_argument("--epochs", default=100, type=int, help="number of total epochs to run")
#     # parser.add_argument("--model_save_dir", default="./model_save", type=str, help="model save directory")
#     # parser.add_argument("--result_dir", default="./result", type=str, help="result save directory")
#     # parser.add_argument("--val_interval", default=5, type=int, help="validation interval")
#     # parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
#     parser.add_argument('--local_rank', default=0, type=int, help='Local rank for distributed training')
#     # parser.add_argument('--rank_id', required=False, default=0, type=int, help='Needed to identify the node and save separate weights.')
#     # parser.add_argument("--smartcache_dataset", action="store_true", help="use monai smartcache Dataset")
    
#     argv = parser.parse_args()
#     return argv

# args = parse_args()


#%% fuctions for ddp

def distributed_init():
    torch.distributed.init_process_group(
                                        backend='nccl',
                                        init_method="env://",
                                        world_size=int(os.environ['WORLD_SIZE']),
                                        rank=int(os.environ["RANK"])
                                        )
    
    torch.distributed.barrier()


def distributed_params():
    return int(os.environ['LOCAL_RANK'])


def set_device(local_rank_param, multi_gpu = True):
    """Returns the device

    Args:
        local_rank_param: Give the local_rank parameter output of distributed_params()
        multi_gpu: Defaults to True.

    Returns:
        Device: Name the output device value
    """
    
    if multi_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda:{}".format(local_rank_param))
            print('CUDA is available. Setting device to CUDA.')
        else:
            device = torch.device('cpu')
            print('CUDA is not available. Setting device to CPU.')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print('CUDA is available. Setting device to CUDA.')
        else:
            device = torch.device('cpu')
            print('CUDA is not available. Setting device to CPU.')
        
    return device

def main():
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
    if MULTI_GPU_FLAG:
        ## initializing multi-node setting
        distributed_init()
        local_rank = distributed_params() # local_rank = gpu in some cases
        ## Setting device
        device = set_device(local_rank_param = local_rank, multi_gpu = True)
        torch.cuda.set_device(device) # set the cuda device, this line doesn't included in Usman's code. But appears in MONAI tutorial
        print(
                "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
                % (torch.distributed.get_rank(), torch.distributed.get_world_size())
            )
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    gene_number_name_mapping,feature_num, train_valid_list, test_loader = load_data(cfg, file_path=data_dir,file_label_path=file_label_path)
    masks = get_mask(pathway_file, gene_number_name_mapping)
    masks = masks[:cfg['mask_len']]

    print("input dim:", len(gene_number_name_mapping.keys()))
    input_gene_num = len(gene_number_name_mapping.keys())
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
    #%%
        # train PGNET first
    for fold_idx in range(K):
        model = PointNetCls(gene_idx_dim = 2, 
                            gene_space_num = gene_space_dim, 
                            class_num=class_num, 
                            input_gene_num = input_gene_num,
                            snet_flag = snet_flag,
                            tnet_flag = tnet_flag, 
                            feature_transform=feature_transform, 
                            atention_pooling_flag = atention_pooling_flag,
                            encoder_flag = encoder_flag)
        if pre_trained:
            # model_state_dict = torch.load("./model_saves"+f"/cls_model_geneSpaceD_3_transfeat_False_attenpool_False_best.pth")
            # # Load the state dict of the pretrained model into a temporary variable
            # pretrained_dict_temp = model_state_dict.copy()
            # # Remove the weights of the last layer from the pretrained state dict
            # pretrained_dict_temp.pop('fc3.weight', None)
            # pretrained_dict_temp.pop('fc3.bias', None)

            # model.load_state_dict(pretrained_dict_temp, strict=False)
            model_save_path = os.path.join(outf, "cls_model_geneSpaceD_3_transfeat_False_attenpool_False_best.pth")
            model_state_dict = torch.load(model_save_path)
            model.load_state_dict(model_state_dict)

        bias_parameters = [p for name, p in model.named_parameters() if 'bias' in name]
        model.to(device)
        if MULTI_GPU_FLAG:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        # Configure an optimizer for the bias parameters only
        optimizer = torch.optim.Adam(bias_parameters, lr=lr)
        # optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
        normalized_weights = normalized_weights.to(device)
        criterion = get_loss_criterion(LOSS_SELECT, WEIGHT_LOSS_FLAG, normalized_weights)

        train_loader, val_loader = train_valid_list[fold_idx]
        best_acc = 0
        for epoch in range(max_epoch):
            epoch_loss, epoch_acc = train_GPN(model, optimizer, scheduler, criterion, train_loader, device, cfg)
            if epoch % eval_interval == 0:
                val_loss, val_acc = val_GPN(model, val_loader, device, criterion, cfg)

                print(f"Epoch: {epoch}, Train Loss: {epoch_loss}, Train Acc: {epoch_acc}, Val Loss: {val_loss}, Val Acc: {val_acc}")
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), os.path.join(outf, f"cls_model_geneSpaceD_best_fold_{fold_idx}.pth"))
                    # save cfg in the same folder as a json file
                    with open(os.path.join(outf, f"cls_model_geneSpaceD_best_fold_{fold_idx}.json"), 'w') as f:
                        json.dump(cfg, f)
        # test the model
        model.load_state_dict(torch.load(os.path.join(outf, f"cls_model_geneSpaceD_best_fold_{fold_idx}.pth")))
        test_acc, test_cm, test_recall, test_precision, test_fpr_tpr, test_roc_auc = test_GPN(model, test_loader, device, cfg)
        print(f"Test Acc: {test_acc}, Test CM: {test_cm}, Test Recall: {test_recall}, Test Precision: {test_precision}, Test ROC_AUC: {test_roc_auc}")

    # mask the input features based on masks
    results = []
    train_loader, val_loader = train_valid_list[0]
    for idx_mask in range(len(masks)):
        model.load_state_dict(torch.load(os.path.join(outf, f"cls_model_geneSpaceD_best_fold_{0}.pth")))
        mask = masks[idx_mask].to(device)
        bias_parameters = [p for name, p in model.named_parameters() if 'bias' in name]
        # Configure an optimizer for the bias parameters only
        optimizer = torch.optim.Adam(bias_parameters, lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
        for epoch in range(5):
            epoch_loss, epoch_acc = train_with_mask(model, optimizer, scheduler, criterion, train_loader,mask, device, cfg)
        
        test_acc, test_cm , p_value_acc, p_value_0, p_value_1  = test_with_mask(model, test_loader, mask, device, cfg)
        test_acc = test_acc.item()
        test_cm = test_cm.cpu().numpy().tolist()
        print(f"Test Acc: {test_acc}, Test CM: {test_cm}, Test P_value_Acc: {p_value_acc}, Test P_value_0: {p_value_0}, Test P_value_1: {p_value_1}")
        results.append({"mask_id": idx_mask, "test_acc": test_acc, "test_cm": test_cm, "p_value_acc": p_value_acc, "p_value_0": p_value_0, "p_value_1": p_value_1})
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(os.path.join(outrf, f"results_{timestamp}.json"), 'w') as f:
        json.dump(results, f)
    with open(os.path.join(outrf, f"cfg_{timestamp}.json"), 'w') as f:
        json.dump(cfg, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Model Training')

    parser.add_argument('--local-rank',
                        required=False,
                        default=0,
                        type=int,
                        help='Needed to identify the node and save separate weights.')


    argv = parser.parse_args()
    main()