import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import enum
import re
from symbol import testlist_star_expr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms
import torch.nn.functional as F
import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support,classification_report
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict
from torch.utils.data import Dataset 
#import redis
import pickle
import time 
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score, recall_score, roc_auc_score, roc_curve
import random 
import torch.backends.cudnn as cudnn
import json
torch.multiprocessing.set_sharing_strategy('file_system')
import os
from Opt.lookahead import Lookahead
from Opt.radam import RAdam


class BagDataset(Dataset):
    def __init__(self, train_path, args, split) -> None:
        super(BagDataset).__init__()
        self.train_path = train_path
        self.args = args
        self.split = split

    def get_bag_feats(self,csv_file_df, args):
        if args.dataset.startswith('TCGA'):
            feats_csv_path = os.path.join(args.feat_type+'_datasets_new_384_tcga','data_tcga_lung_tree' ,csv_file_df.iloc[0].split('/')[-1] + '.csv')
        else:
            feats_csv_path = csv_file_df.iloc[0]
        
        df = pd.read_csv(feats_csv_path, low_memory=False)
        feats = shuffle(df).reset_index(drop=True)
        feats = feats.to_numpy()
        
        
        # zjw revise
        label = csv_file_df.iloc[1]
        label = torch.tensor(np.array(label))
        # print(type(feats), feats.shape)
        # print(feats_csv_path)
        feats_ = torch.tensor(feats.astype(float)).float()
        return label, feats_

    def dropout_patches(self,feats, p):
        idx = np.random.choice(np.arange(feats.shape[0]), int(feats.shape[0]*(1-p)), replace=False)
        sampled_feats = np.take(feats, idx, axis=0)
        return sampled_feats
    
    def __getitem__(self, idx):

        label, feats = self.get_bag_feats(self.train_path.iloc[idx], self.args)

        return  label, feats

    def __len__(self):
        return len(self.train_path)


def train(train_df, milnet, criterion, optimizer, args, log_path):
    milnet.train()
    total_loss = 0
    atten_max = 0
    atten_min = 0
    atten_mean = 0
    
    if args.model == 'moe' or args.model == 'transmil':
        criterion = torch.nn.CrossEntropyLoss()


    for i,(bag_label,bag_feats) in enumerate(train_df):
        bag_label = bag_label.cuda()
        bag_feats = bag_feats.cuda()
        #print("input shape", bag_feats.shape)
        bag_feats = bag_feats.view(-1, args.feats_size)  # n x feat_dim
        #print(bag_feats.shape)
        optimizer.zero_grad()
        if args.model == 'dsmil':
            ins_prediction, bag_prediction, attention, atten_B = milnet(bag_feats)
            max_prediction, _ = torch.max(ins_prediction, 0)  
            # print(bag_prediction, max_prediction,bag_label.long())      
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
            loss = 0.5*bag_loss + 0.5*max_loss

        elif args.model == 'abmil':
            bag_prediction, _, attention = milnet(bag_feats)
            loss =  criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        elif args.model == 'transmil':
            output = milnet(bag_feats)
            bag_prediction, bag_feature ,attention = output['logits'], output["Bag_feature"], output["A"]  
            loss = criterion(bag_prediction, bag_label)
        
        elif args.model == 'moe':
            
            bag_feats = bag_feats[None,:,:]
            output, t_loss = milnet(bag_feats)
            bag_prediction = output['Y_logits']
            bag_pred = output["Y_hat"]

            e_bag_prediction = output['e_Y_logits']
            e_bag_pred = output["e_Y_hat"]

            loss2 = criterion(e_bag_prediction, bag_label)
            loss = loss2+t_loss
            
            #break
        
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        
        if args.c_path:
            attention = output['A']
            atten_max = atten_max+ attention.max().item()
            atten_min = atten_min+attention.min().item()
            atten_mean = atten_mean+ attention.mean().item()
            sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f attention max:%.4f, min:%.4f, mean:%.4f' 
            % (i, len(train_df), loss.item(), attention.max().item(), attention.min().item(), attention.mean().item()))
           
        else:
            sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f ' % (i, len(train_df), loss.item()))

        #break

    if args.c_path:
        atten_max = atten_max / len(train_df)
        atten_min =  atten_min / len(train_df)
        atten_mean = atten_mean / len(train_df)
        with open(log_path,'a+') as log_txt:
                log_txt.write('\n atten_max'+str(atten_max))
                log_txt.write('\n atten_min'+str(atten_min))
                log_txt.write('\n atten_mean'+str(atten_mean))
    return total_loss / len(train_df)


def test(test_df, milnet, criterion, optimizer, args, log_path, epoch):
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions = []
    test_prob = []
    test_preds = []
    Tensor = torch.cuda.FloatTensor
    if args.model == 'moe' or args.model == 'transmil':
        criterion = torch.nn.CrossEntropyLoss()
    f = open('./TCGA_DIS.txt', 'w')
    with torch.no_grad():
        for i,(bag_label,bag_feats) in enumerate(test_df):
            label = bag_label.numpy()
            #print("label,", label)
            bag_label = bag_label.cuda()
            bag_feats = bag_feats.cuda()
            bag_feats = bag_feats.view(-1, args.feats_size)
            if args.model == 'dsmil':
                ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
                max_prediction, _ = torch.max(ins_prediction, 0)  
                bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
                max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
                # bag_loss = criterion(bag_prediction, bag_label.long())
                # max_loss = criterion(max_prediction.view(1, -1), bag_label.long())
                loss = 0.5*bag_loss + 0.5*max_loss
            elif args.model == 'abmil':
                bag_prediction, _, _ =  milnet(bag_feats)
                max_prediction = bag_prediction
                loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            elif args.model == 'transmil':
                output = milnet(bag_feats)
                bag_prediction, bag_feature, bag_pred =  output['logits'], output["Bag_feature"], output["Y_hat"]
                max_prediction = bag_prediction
                e_bag_pred = output["Y_hat"]
                e_bag_prediction = bag_prediction
                #print(bag_prediction, bag_label)
                #exit()
                loss =  criterion(bag_prediction.view(1, -1), bag_label)
            elif args.model == 'moe':
                bag_feats = bag_feats[None,:,:]
                output, t_loss, distribute = milnet(bag_feats)
                bag_pred = output["Y_hat"]
                bag_prediction = output['Y_logits']
                
                e_bag_prediction = output['e_Y_logits']
                e_bag_pred = output["e_Y_hat"]
                max_prediction = e_bag_prediction
                ####
                loss1 = criterion(bag_prediction, bag_label)
                loss2 = criterion(e_bag_prediction, bag_label)
                loss = loss1+loss2+t_loss

                #print('select_weights', output['select_weights'])
               
            total_loss = total_loss + loss.item()
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f ' % (i, len(test_df), loss.item()))
            #print(label, test_pred)
            test_labels.extend(label)
            test_preds.extend(e_bag_pred.cpu().numpy()) #####
            if args.average:   # notice args.average here
                test_prob.extend([(0.5*torch.sigmoid(max_prediction)+0.5*torch.sigmoid(e_bag_prediction)).squeeze().cpu().numpy()])
                
            else: test_prob.extend([(0.0*torch.sigmoid(max_prediction)+1.0*torch.sigmoid(e_bag_prediction)).squeeze().cpu().numpy()])
        f.close()
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_preds)
    
    labels = []
    for i in test_labels:
        label = np.zeros(args.num_classes)
        label[i] = 1
        labels.append(label)

    #auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    labels = np.array(labels)
    test_prob = np.array(test_prob)
    auc_value, _, thresholds_optimal = multi_label_roc(labels, test_prob, args.num_classes, pos_label=1)
    

    with open(log_path,'a+') as log_txt:
        log_txt.write('\n *****************Threshold by optimal*****************')
    
    test_labels = np.squeeze(test_labels)
    print(confusion_matrix(test_labels,test_predictions))
    info = confusion_matrix(test_labels,test_predictions)
    with open(log_path,'a+') as log_txt:
            log_txt.write('\n'+str(info))
    
    
    bag_score = 0
    # average acc of all labels
    for i in range(0, len(test_df)):
        #print(test_labels[i], test_predictions[i])
        if np.array_equal(test_labels[i], test_predictions[i]) == 0:
            print(i)
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score         
    avg_score = bag_score / len(test_df)  #ACC
    #print(test_labels)
    #exit()
    cls_report = classification_report(test_labels, test_predictions, digits=4)

    # print(confusion_matrix(test_labels,test_predictions))
    print('\n multi-label Accuracy:{:.2f}, AUC:{:.2f}'.format(avg_score*100, sum(auc_value)/len(auc_value)*100))
    print('\n', cls_report)
    with open(log_path,'a+') as log_txt:
        log_txt.write('\n  multi-label Accuracy:{:.2f}, AUC:{:.2f}'.format(avg_score*100, sum(auc_value)/len(auc_value)*100))
        log_txt.write('\n' + cls_report)
    
    return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal

def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        if sum(label)==0:
            continue
        prediction = predictions[:, c]
        # print(label, prediction,label.shape, prediction.shape, labels.shape, predictions.shape)
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        #print("threshold", threshold)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def main():
    parser = argparse.ArgumentParser(description='Train IBMIL for TransMIL')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--gpu', type=str, default= '0')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--weight_decay_conf', default=1e-4, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--dataset', default='TCGA-lung-default', type=str, help='Dataset folder name')
    parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--model', default='dsmil', type=str, help='MIL model [dsmil]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=0, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--average', type=bool, default=True, help='Average the score of max-pooling and bag aggregating')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
    parser.add_argument('--agg', type=str,help='which agg')
    parser.add_argument('--trained_model', type=str,help='pre-trained_model')
    parser.add_argument('--c_path', nargs='+', default=None, type=str,help='directory to confounders')
    parser.add_argument('--feat_type', type=str, default='vit', help='ferature type')
    parser.add_argument('-base_global_trained_model', action='store_true', help='if baseline trained_model')
    parser.add_argument('-test_only', action='store_true', help='only test')
    
    args = parser.parse_args()

    if args.feat_type == 'vit':
        args.feats_size = 384
    elif args.feat_type == 'ctrans':
        args.feats_size = 768


    # logger
    arg_dict = vars(args)
    dict_json = json.dumps(arg_dict)
    if args.c_path:
        save_path = os.path.join('deconf', args.feat_type, datetime.date.today().strftime("%m%d%Y"), str(args.dataset)+'_'+str(args.model)+'_'+str(args.agg )+'_c_path')
    else:
        save_path = os.path.join('baseline_soft', args.feat_type, datetime.date.today().strftime("%m%d%Y"), str(args.dataset)+'_'+str(args.model)+'_'+str(args.agg )+'_fulltune')
    run = len(glob.glob(os.path.join(save_path, '*')))
    save_path = os.path.join(save_path, str(run))
    os.makedirs(save_path, exist_ok=True)
    save_file = save_path + '/config.json'
    with open(save_file,'w+') as f:
        f.write(dict_json)
    log_path = save_path + '/log.txt'

    # seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    '''
    model 
    1. set require_grad    
    2. choose model and set the trainable params 
    3. load init
    '''
    
    if args.model == 'transmil':
        #import Models.TransMIL.net as mil
        import Models.TransMIL.net_frequency as mil
        milnet = mil.TransMIL(input_size=args.feats_size, n_classes=args.num_classes, confounder_path=args.c_path).cuda()
    
    if args.model == 'moe':
        from MOE_v3_2 import MixtralSparseMoeBlock # C16
        # from MOE_v3_3 import MixtralSparseMoeBlock # TCGA
        # from MOE_v3_4 import MixtralSparseMoeBlock # BRACS
        


        
        milnet = MixtralSparseMoeBlock(args).cuda()
        # from PMM_net import PMM_network
        # milnet = PMM_network(args).cuda()
    
    if args.trained_model:
        with open(log_path,'a+') as log_txt:
            log_txt.write('\n Load pretrained path {}'.format(args.trained_model))

    # for name, _ in milnet.named_parameters():
    #     print('Training {}'.format(name))
    #     with open(log_path,'a+') as log_txt:
    #         log_txt.write('\n Training {}'.format(name))
    

    if args.dataset.startswith("TCGA"):
                                                            # args.dataset,
        bags_csv = os.path.join(args.feat_type+'_datasets_new_384_tcga',  args.dataset+'.csv')
        bags_path = pd.read_csv(bags_csv)
        train_path = bags_path.iloc[0:int(len(bags_path)*0.8), :]
        test_path = bags_path.iloc[int(len(bags_path)*0.8):, :]

    elif args.dataset.startswith('Camelyon16'):
        # bags_csv = os.path.join('datasets', args.dataset, args.dataset+'_off.csv') #offical train test
                                                           # args.dataset,
        bags_csv = os.path.join(args.feat_type+'_datasets_new_384',  args.dataset+'.csv')
        bags_path = pd.read_csv(bags_csv)
        train_path = bags_path.iloc[0:268, :]
        test_path = bags_path.iloc[268:, :]
        # bags_csv = os.path.join('datasets', args.dataset)
        # training = os.path.join(bags_csv, 'training.csv')
        # testing = os.path.join(bags_csv, 'testing.csv')
        # train_bags_path = pd.read_csv(training)
        # test_bags_path = pd.read_csv(testing)
        # train_path = train_bags_path.iloc[0:] ###
        # test_path = test_bags_path.iloc[0:]
        
    trainset =  BagDataset(train_path, args, split='train')
    train_loader = DataLoader(trainset,1, shuffle=True, num_workers=16)
    testset =  BagDataset(test_path, args, split='test')
    test_loader = DataLoader(testset,1, shuffle=False, num_workers=16)

    # sanity check begins here
    print('*******sanity check *********')
    for k,v in milnet.named_parameters():
        #print("##", k)
        if 'trans1' in k:#or 'gate' in k or 'classifier' in k:                # 2-3
            v.requires_grad = False
        # else:
        #     v.requires_grad = False
        if v.requires_grad == True:
            #print(k)
            print('Training {}'.format(k))
            with open(log_path,'a+') as log_txt:
                log_txt.write('\n Training {}'.format(k))
    if args.trained_model:
        gate_model = torch.load('./baseline/vit/04022024/Camelyon16_moe_no_fulltune/13/14.pth')
        
        model_static_dicts = torch.load(args.trained_model)
        new_model_static_dicts = {}
       
        if args.base_global_trained_model:
            for k, v in model_static_dicts.items():
                new_k = 'trans1.'+k
                new_model_static_dicts[new_k] = v
            # for k, v in gate_model.items():       ## 3-3
            #     if 'gate' in k:
            #         new_model_static_dicts[k] = v
            msg = milnet.load_state_dict(new_model_static_dicts, strict=False)
        else:
            # for k, v in model_static_dicts.items():
            #     if 'gate' not in k:
            #         new_model_static_dicts[k] = v
            msg = milnet.load_state_dict(model_static_dicts, strict=False)
        print('Missing these:', msg.missing_keys)
    
    # loss, optim, schduler
    criterion = nn.BCEWithLogitsLoss() 
    original_params = []
    confounder_parms = []
    for pname, p in milnet.named_parameters():
        if ('confounder' in pname):
            confounder_parms += [p]
            print('confounders:',pname )
        else:
            original_params += [p]
    
    print('lood ahead optimizer in transmil....')
    base_optimizer = RAdam([
                            {'params':original_params},
                            {'params':confounder_parms, ' weight_decay':0.0001},
                            ], 
                            lr=args.lr,#0.0002, 
                            weight_decay=0.00001)
    optimizer = Lookahead(base_optimizer)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)

    best_score = 0
    best_acc = 0

    # only test
    if args.test_only:
        model_static_dicts = torch.load(args.trained_model)
        milnet.load_state_dict(model_static_dicts, strict=True)
        test_loss_bag, avg_score, aucs, thresholds_optimal = test(test_loader, milnet, criterion, optimizer, args, log_path, epoch=0)
        exit()
    
    for epoch in range(1, args.num_epochs):
        start_time = time.time()
        train_loss_bag = train(train_loader, milnet, criterion, optimizer, args, log_path) # iterate all bags
        print('epoch time:{}'.format(time.time()- start_time))
        test_loss_bag, avg_score, aucs, thresholds_optimal = test(test_loader, milnet, criterion, optimizer, args, log_path, epoch)
        
        info = 'Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: '%(epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))+'\n'
        with open(log_path,'a+') as log_txt:
            log_txt.write(info)
        print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' % 
                (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))) 
        # if args.model != 'transmil': #############
        #     scheduler.step()
        current_score = (sum(aucs)/2 + avg_score)/2 #### 
        current_acc = avg_score
        #if current_score >= best_score or current_acc >= best_acc :
        if current_acc >= best_acc:
            if current_acc == best_acc and current_score >= best_score:
                best_score = current_score
                best_acc = current_acc
                save_name = os.path.join(save_path, str(run+1)+'.pth')
                torch.save(milnet.state_dict(), save_name)
                with open(log_path,'a+') as log_txt:
                    info = 'Best model saved at: ' + save_name +'\n'
                    log_txt.write(info)
                    info = 'Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal))+'\n'
                    log_txt.write(info)
                print('Best model saved at: ' + save_name)
                print('Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))
            elif current_acc > best_acc:
                best_acc = current_acc
                if current_score > best_score:
                    best_score = current_score
                save_name = os.path.join(save_path, str(run+1)+'.pth')
                torch.save(milnet.state_dict(), save_name)
                with open(log_path,'a+') as log_txt:
                    info = 'Best model saved at: ' + save_name +'\n'
                    log_txt.write(info)
                    info = 'Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal))+'\n'
                    log_txt.write(info)
                print('Best model saved at: ' + save_name)
                print('Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))


        if epoch == args.num_epochs-1:
            save_name = os.path.join(save_path, 'last.pth')
            torch.save(milnet.state_dict(), save_name)
    log_txt.close()

if __name__ == '__main__':
    main()