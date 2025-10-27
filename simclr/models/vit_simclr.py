import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import sys
sys.path.append('../')
from functools import partial

import moco.builder_infence
import moco.builder
import moco.loader
import vits



class VitSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(VitSimCLR, self).__init__()
        self.vit_dict = {"vit_small": moco.builder_infence.MoCo_ViT(partial(vits.__dict__['vit_small'], stop_grad_conv1=True))}

        self.vit = self._get_basemodel(base_model)
        #print(vit)
        #exit()
        num_ftrs = 256#vit.fc.in_features

        #self.features = #nn.Sequential(*list(vit.children())[:-1])

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

    def _get_basemodel(self, model_name):
        try:
            model = self.vit_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        #print('x',x.shape)
        h = self.vit(x)
        h = h.squeeze()
        #print("h:", h.shape) #[N, C]
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x



        """
        vit_model = moco.builder_infence.MoCo_ViT(partial(vits.__dict__['vit_small'], stop_grad_conv1=True))#
        #vit_model.head = nn.Identity()
        weight_path = args.weights
        state_dict_weights = torch.load(weight_path)['state_dict']  
        new_state_dict_weights = {}
        for (k, v) in state_dict_weights.items():
            k_new = k[7:]
            new_state_dict_weights[k_new] = v
            
        vit_model.load_state_dict(new_state_dict_weights, strict=True)
        i_classifier = vit_model.cuda()
        """
