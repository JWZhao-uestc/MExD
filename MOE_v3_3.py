import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import Models.TransMIL.net as mil
from loss import DiscriminativeLoss
import numpy as np
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x

class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x
    

class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN


        return A  ### K x N

class MixtralBlockSparseTop2MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.ffn_dim = 256#config.intermediate_size
        self.hidden_dim = input_dim

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = nn.ReLU()#ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.w1(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states

class SpectralGatingNetwork(nn.Module):
    def __init__(self, dim=512, h=60, w=32):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)
        #print('x', x.shape)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        #print('x', x.shape)
        f_a, f_b = x.shape[1], x.shape[2]
        weight = F.interpolate(self.complex_weight.permute(2,3,0,1), (f_a, f_b))
        weight = weight.permute(2,3,0,1)
        #print('weight', weight.shape, weight.permute(2,3,0,1).shape)
        weight = torch.view_as_complex(weight.contiguous())
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        #print('x', x.shape)
        x = x.reshape(B, N, C)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = SpectralGatingNetwork(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x

class MixtralSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, args):
        super().__init__()
        self.hidden_dim = args.feats_size  ### original feature size
        self.patch_dim_2 = 512
        self.D_inner = 128
        self.ffn_dim = 256 #config.intermediate_size      ### 
        self.num_experts = 3 #config.num_local_experts
        self.top_k = 1 #config.num_experts_per_tok
        self.droprate = 0.0
        #self.experts = nn.ModuleList([MixtralBlockSparseTop2MLP(self.patch_dim_2) for _ in range(self.num_experts)])
        self.trans1 = mil.TransMIL(input_size=args.feats_size, n_classes=args.num_classes, confounder_path=args.c_path).cuda()
        #self.FFT = Block(dim=512)
        #self.FFT2 = Block(dim=512)
          # gating
        self.gate = nn.Linear(self.patch_dim_2, 2, bias=False)
        self.gate1 = nn.Linear(self.patch_dim_2, 2, bias=False)
        self.gate2 = nn.Linear(self.patch_dim_2, 2, bias=False)
        #self.trans2 = mil.TransMIL(input_size=args.feats_size, n_classes=args.num_classes, confounder_path=args.c_path).cuda()
        # self.milnet = nn.ModuleList([mil.TransMIL(input_size=args.feats_size, n_classes=args.num_classes, confounder_path=args.c_path).cuda()
        #                               for _ in range(self.num_experts)])
        self.agg1 = mil.TransMIL(input_size=self.patch_dim_2, n_classes=args.num_classes, confounder_path=args.c_path).cuda()
        #self.agg2 = mil.TransMIL(input_size=self.patch_dim_2, n_classes=args.num_classes, confounder_path=args.c_path).cuda()
        # self.dimreduction = DimReduction(self.patch_dim_2, self.D_inner)
        # self.attention = Attention_Gated(self.D_inner, D=128, K=1)
        # self.Slide_classifier = Classifier_1fc(self.D_inner, args.num_classes, self.droprate)
        

        self.loss = DiscriminativeLoss()
        self.classifier = nn.ModuleList([nn.Linear(self.patch_dim_2, self.num_experts) for _ in range(self.num_experts)])
        
        joint_space_dim = 128
        #dropout_v = 0.1
        self.confounder_W_q = nn.Linear(512, joint_space_dim)
        self.confounder_W_k = nn.Linear(512, joint_space_dim)
        self._fc2 = nn.Linear(1024, args.num_classes) #1024
        self.norm2 = nn.LayerNorm(1024)             #1024
        self.criterion = nn.CrossEntropyLoss()
        #self.classifier = nn.Linear()



    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        
        #stage1
        hidden_states1 = hidden_states.view(-1, hidden_dim)
        output1 = self.trans1(hidden_states1)                                            
        Y_logits = output1["logits"]
        Y_prob = output1["Y_prob"]
        Y_hat = output1["Y_hat"]

        global_bag_feature = output1["Bag_feature"]
        patches_feature = output1["patch_feature"] #[N,c]
        #print(patches_feature.shape)
        #print('global_bag_feature', global_bag_feature.shape)

        patches_feature = patches_feature.view(-1, self.patch_dim_2)
        patches_feature_frequency = patches_feature
        # patches_feature_frequency = self.FFT(patches_feature.unsqueeze(0))
        # #patches_feature_frequency = self.FFT2(patches_feature_frequency)
        # patches_feature_frequency = patches_feature_frequency.squeeze(0)
        # patches_feature = patches_feature_frequency
        
        all_expert_mask = []
        # all_routing_weights = []
        ##### for gate 0
        router_logits = self.gate(patches_feature_frequency)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        #routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        
        # we cast back to the input dtype
        routing_weights = routing_weights.to(patches_feature.dtype)
        # all_routing_weights.append(routing_weights)
        # balance 
        tmp = torch.sum(selected_experts)
        if torch.sum(selected_experts) == 0 or torch.sum(selected_experts) == selected_experts.shape[0]:
            select_weights = routing_weights#.squeeze() #selected_idx = torch.sort(routing_weights.squeeze(),descending=True)
            average = torch.mean(select_weights)#+0.2
            #print('average', average, torch.max(select_weights), torch.min(select_weights))
            #print('select_weights', select_weights.shape)
            #print('SUM', torch.sum(selected_experts1), tmp)
            if tmp == 0:
                selected_experts[torch.where(select_weights < average)] = 1
            else:
                selected_experts[torch.where(select_weights <= average)] = 0    
        
        assert torch.sum(selected_experts) != 0
        assert torch.sum(selected_experts) != selected_experts.shape[0]

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        all_expert_mask.append(expert_mask)
        
        ##### for gate 1
        router_logits1 = self.gate1(patches_feature_frequency)
        routing_weights1 = F.softmax(router_logits1, dim=1, dtype=torch.float)
        routing_weights1, selected_experts1 = torch.topk(routing_weights1, self.top_k, dim=-1)
        #routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights1 = routing_weights1.to(patches_feature.dtype)
        # all_routing_weights.append(routing_weights1)
        
        # balance 
        tmp = torch.sum(selected_experts1)
        if torch.sum(selected_experts1) == 0 or torch.sum(selected_experts1) == selected_experts1.shape[0]:
            select_weights1 = routing_weights1#.squeeze() #selected_idx = torch.sort(routing_weights.squeeze(),descending=True)
            average = torch.mean(select_weights1)#+0.2
            #print('average', average, torch.max(select_weights), torch.min(select_weights))
            #print('select_weights', select_weights.shape)
            #print('SUM', torch.sum(selected_experts1), tmp)
            if tmp == 0:
                selected_experts1[torch.where(select_weights1 < average)] = 1
            else:
                selected_experts1[torch.where(select_weights1 < average)] = 0    
        
        assert torch.sum(selected_experts1) != 0
        assert torch.sum(selected_experts1) != selected_experts1.shape[0]
        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask1 = torch.nn.functional.one_hot(selected_experts1, num_classes=self.num_experts).permute(2, 1, 0)
        all_expert_mask.append(expert_mask1)


        # for gate 2
        router_logits2 = self.gate2(patches_feature_frequency)
        routing_weights2 = F.softmax(router_logits2, dim=1, dtype=torch.float)
        routing_weights2, selected_experts2 = torch.topk(routing_weights2, self.top_k, dim=-1)
        #routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights2 = routing_weights2.to(patches_feature.dtype)
        # all_routing_weights.append(routing_weights2)
        
        #balance
        tmp = torch.sum(selected_experts2)
        if torch.sum(selected_experts2) == selected_experts2.shape[0]:
            select_weights2 = routing_weights2#.squeeze() #selected_idx = torch.sort(routing_weights.squeeze(),descending=True)
            average = torch.mean(select_weights2)#+0.2
            #print('average', average, torch.max(select_weights), torch.min(select_weights))
            #print('select_weights', select_weights.shape)
            #print('SUM', torch.sum(selected_experts1), tmp)
            if tmp == 0:
                selected_experts2[torch.where(select_weights2 < average)] = 1
            else:
                selected_experts2[torch.where(select_weights2 < average)] = 0    
        
        assert torch.sum(selected_experts2) != selected_experts2.shape[0]

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask2 = torch.nn.functional.one_hot(selected_experts2, num_classes=self.num_experts).permute(2, 1, 0)
        all_expert_mask.append(expert_mask2)

        #logits = []
        local_bag_feature = []
        experts_predict = []
        experts_prob = []
        joint_losses = 0
        p = 0.0
        represent_patchs = []
        select_idxs = []

        #lamb = 1.0
        #print('total patch', patches_feature.shape[0])
        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts): # the 0-th expert predict 0, the 1-th expert predict 1. 
            #expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(all_expert_mask[expert_idx][1])

            if top_x.shape[0] == 0:
                logits = torch.zeros((1,self.num_experts)).cuda()
                logits[:,expert_idx] = 0.01
                logits[:,1-expert_idx] = 0.99
                experts_predict.append(logits.squeeze())
                experts_prob.append(logits.squeeze())
                p=10.0
                represent_patchs.append(torch.zeros([1,self.patch_dim_2]).to(patches_feature.device))
                #lamb = 100
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)

            current_state = patches_feature[None, top_x_list].reshape(-1, self.patch_dim_2)            
            #current_hidden = expert_layer(current_state)# * routing_weights[top_x_list, idx_list, None]
            current_state_router = routing_weights[top_x_list, idx_list, None]
            #current_state_router = all_routing_weights[expert_idx][top_x_list, idx_list, None]
            #print(current_state.shape)#, current_state_router.shape, int(current_state.shape[0]*0.5))
            select_weights, selected_patches = torch.sort(current_state_router.squeeze(),descending=True)
            selected_patches_list = selected_patches.tolist()
          
            if expert_idx == 0:
                nums = int(current_state.shape[0]*0.25)  # 0.8, 0.25, 0.25
                #print("***", int(current_state.shape[0]*0.5))
                if  nums != 0:
                    selected_patches = current_state[None, selected_patches_list[0:nums]].reshape(-1, self.patch_dim_2)
                    expert_select_idxs = selected_patches_list[0:nums]
                else:
                    selected_patches = current_state #[None, [selected_patches_list]].reshape(-1, self.patch_dim_2)
                    expert_select_idxs = selected_patches_list
                #print("expert 0", selected_patches.shape)

            if expert_idx == 1:
                #selected_patches = current_state

                nums = int(current_state.shape[0]*0.5)  # 0.8, 0.25, 0.5
                if  nums != 0:
                    selected_patches = current_state[None, selected_patches_list[0:nums]].reshape(-1, self.patch_dim_2)
                    expert_select_idxs = selected_patches_list[0:nums]
                else:
                    selected_patches = current_state #[None, [selected_patches_list]].reshape(-1, self.patch_dim_2)
                    expert_select_idxs = selected_patches_list

                #print("expert 0", selected_patches.shape)

            if expert_idx == 2:
                #selected_patches = current_state

                nums = int(current_state.shape[0]*0.25) #0.15, 0.1, 0.25
                if  nums != 0:
                    selected_patches = current_state[None, selected_patches_list[0:nums]].reshape(-1, self.patch_dim_2)
                    expert_select_idxs = selected_patches_list[0:nums]
                else:
                    selected_patches = current_state   
                    expert_select_idxs = selected_patches_list
                #print('expert 1', selected_patches.shape)
            select_idxs.append(expert_select_idxs)
            represent_patchs.append(selected_patches)

            class_specify_feas = torch.mean(selected_patches, 0) #current_state
            #bag_logits = self.Slide_classifier(bag_feat)

            logits = self.classifier[expert_idx](class_specify_feas)
            experts_predict.append(logits)
            expert_idx = torch.tensor(np.array([expert_idx])).cuda()
           
            if expert_idx < 2:
                loss = self.criterion(logits.view(1,-1), expert_idx)
                joint_losses += loss
            
            logits_prob = F.softmax(logits, -1)
            experts_prob.append(logits_prob)
        joint_losses += p
        
        assert len(represent_patchs) == self.num_experts

        #print('****', patches_feature.shape, represent_patchs[0].shape, represent_patchs[1].shape, represent_patchs[2].shape)
        # print('****', patches_feature.shape[0], represent_patchs[0].shape[0], represent_patchs[1].shape[0], represent_patchs[2].shape[0])
        distribute = [patches_feature.shape[0], represent_patchs[0].shape[0], represent_patchs[1].shape[0], represent_patchs[2].shape[0]]

        represent_patchs = torch.cat(represent_patchs, 0)
        output = self.agg1(represent_patchs)
        e_Y_logits = output['logits']
        e_Y_hat = output['Y_hat']

        # tmp_set = []
        # for i, idx_s in enumerate(select_idxs):
        #     #if selected_patches.shape[0] == 1:
        #     if not type(idx_s) == list:
        #         idx_s = [idx_s]
        #     tmp_set = set(tmp_set).union(set(idx_s))
        # select_idxs_union = list(tmp_set)
        # # print(select_idxs_unoin)
        # #exit()
        # represent_patchs = patches_feature[None, select_idxs_union].reshape(-1, self.patch_dim_2)

        # patches_feature_s = output["patch_feature"].squeeze(0) #[N,c]
        # patches_feature_s = represent_patchs
        # patches_feature_s = self.dimreduction(represent_patchs)
        # A = self.attention(patches_feature_s)   ## K x N
        # bag_A = F.softmax(A, dim=1).mean(0, keepdim=True)
        
        # bag_feat = torch.mm(bag_A, patches_feature_s)
        # bag_logits = self.Slide_classifier(bag_feat)

        # e_Y_logits = bag_logits
        # e_Y_hat = torch.argmax(bag_logits, 1)
        
        results_dict = {'e_Y_logits': e_Y_logits, 'e_Y_hat': e_Y_hat, 'Y_logits': Y_logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'router_logits': router_logits, 'select_weights':select_weights}
        
         

        return results_dict, joint_losses, distribute