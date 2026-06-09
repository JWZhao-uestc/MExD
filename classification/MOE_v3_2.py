import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import Models.TransMIL.net as mil
#from loss import DiscriminativeLoss
import numpy as np
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

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.model.ori_data_dim  ### original feature size
        self.classes = config.data.num_classes
        self.patch_dim_2 = 512
        self.ffn_dim = 256 #config.intermediate_size      ### 
        self.num_experts = 2 #config.num_local_experts
        self.top_k = 1 #config.num_experts_per_tok

        #self.experts = nn.ModuleList([MixtralBlockSparseTop2MLP(self.patch_dim_2) for _ in range(self.num_experts)])
        self.trans1 = mil.TransMIL(input_size=self.hidden_dim, n_classes=self.classes, confounder_path=None)#.cuda()
          # gating
        self.gate = nn.Linear(self.patch_dim_2, self.num_experts, bias=False)
        
        self.agg1 = mil.TransMIL(input_size=self.patch_dim_2, n_classes=self.classes, confounder_path=None)#.cuda()
        #self.agg2 = mil.TransMIL(input_size=self.patch_dim_2, n_classes=args.num_classes, confounder_path=args.c_path).cuda()


        #self.loss = DiscriminativeLoss()
        self.classifier = nn.ModuleList([nn.Linear(self.patch_dim_2, self.classes) for _ in range(self.num_experts)])
        
        joint_space_dim = 128
        self.confounder_W_q = nn.Linear(512, joint_space_dim)
        self.confounder_W_k = nn.Linear(512, joint_space_dim)
        self._fc2 = nn.Linear(1024, self.classes) #1024
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
        
        patches_feature = patches_feature.view(-1, self.patch_dim_2)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(patches_feature)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        #routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(patches_feature.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, self.patch_dim_2), dtype=patches_feature.dtype, device=patches_feature.device
        )
        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        local_bag_feature = []
        experts_predict = []
        experts_prob = []
        joint_losses = 0
        p = 0.0
        represent_patchs = []
        class_specify_feas_sets = []

        
        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts): # the 0-th expert predict 0, the 1-th expert predict 1. 
            #expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            
            if top_x.shape[0] == 0:
                logits = torch.zeros((1,2)).to(hidden_states1.device) 
                logits[:,expert_idx] = 0.01
                logits[:,1-expert_idx] = 0.99
                experts_predict.append(logits.squeeze())
                experts_prob.append(logits.squeeze())
                p=10.0
                represent_patchs.append(torch.zeros([1,self.patch_dim_2]).to(patches_feature.device))
                class_specify_feas_sets.append(torch.zeros([1,self.patch_dim_2]).to(patches_feature.device))
                
                #lamb = 100
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)

            current_state = patches_feature[None, top_x_list].reshape(-1, self.patch_dim_2)            
            
            current_state_router = routing_weights[top_x_list, idx_list, None]

            select_weights, selected_patches = torch.sort(current_state_router.squeeze(),descending=True)
            selected_patches_list = selected_patches.tolist()
            
            if expert_idx == 0:
                nums = int(current_state.shape[0]*0.25)  # 0.15, 0.25, 0.2
                
                if  nums != 0:
                    selected_patches = current_state[None, selected_patches_list[0:nums]].reshape(-1, self.patch_dim_2)
                else:
                    selected_patches = current_state #[None, [selected_patches_list]].reshape(-1, self.patch_dim_2)
            
            if expert_idx == 1:
               
                nums = int(current_state.shape[0]*0.5)  # 0.3, 0.5, 0.5
                if  nums != 0:
                    selected_patches = current_state[None, selected_patches_list[0:nums]].reshape(-1, self.patch_dim_2)
                else:
                    selected_patches = current_state #[None, [selected_patches_list]].reshape(-1, self.patch_dim_2)
                
            
            represent_patchs.append(selected_patches)
            
            class_specify_feas = torch.mean(selected_patches, 0) #current_state
            class_specify_feas_sets.append(class_specify_feas)

            logits = self.classifier[expert_idx](class_specify_feas)
        
            experts_predict.append(logits)
            
            expert_idx = torch.tensor(np.array([expert_idx])).to(hidden_states1.device) 
            
            loss = self.criterion(logits.view(1,-1), expert_idx)
            logits_prob = F.softmax(logits, -1)
            experts_prob.append(logits_prob)
        

        expert_feature = class_specify_feas_sets[0]*experts_prob[0][0] + class_specify_feas_sets[1]*experts_prob[1][1]

        assert len(represent_patchs) == self.num_experts

        represent_patchs = torch.cat(represent_patchs, 0)
    
        output = self.agg1(represent_patchs) 
        e_Y_logits = output['logits']
        e_Y_prob = output['Y_prob']
        e_Y_hat = output['Y_hat']
        
        results_dict = {'e_Y_logits': e_Y_logits, 'e_Y_prob': e_Y_prob, 'e_Y_hat': e_Y_hat, 
                        'Y_logits': Y_logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'router_logits': router_logits, 
                        'select_weights': select_weights, 'expert_feature': expert_feature}
        
        return results_dict, joint_losses