import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# official code of transmil

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

class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x

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

class FFT_block(nn.Module):

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

# class FFT_block(nn.Module):

#     def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.filter = SpectralGatingNetwork(dim, h=h, w=w)
#         # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         # self.norm2 = norm_layer(dim)
#         # mlp_hidden_dim = int(dim * mlp_ratio)
#         # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#     def forward(self, x):
#         #x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
#         x = x + self.drop_path(self.filter(self.norm1(x)))
#         return x

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, n_classes, input_size, confounder_path=None):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self.patch_dim_2 = 512
        self.D_inner = 128
        self.droprate = 0.0

        self._fc1 = nn.Sequential(nn.Linear(input_size, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.layer1 = FFT_block(dim=512)
        self.layer2 = FFT_block(dim=512)
        self.norm = nn.LayerNorm(512)

        self.dimreduction = DimReduction(self.patch_dim_2, self.D_inner)
        self.attention = Attention_Gated(self.D_inner, D=128, K=1)
        self.Slide_classifier = Classifier_1fc(self.D_inner, self.n_classes, self.droprate)
        

        self._fc2 = nn.Linear(512, self.n_classes)
        self.confounder_path = confounder_path
        if confounder_path:
            self.confounder_path = confounder_path
            conf_list = []
            for i in confounder_path:
                conf_list.append(torch.from_numpy(np.load(i)).float())
            conf_tensor = torch.cat(conf_list, 0)  #[ k, C, K] k-means, c classes , K-dimension, should concatenate at centers k
            #print("conf_tensot", conf_tensor.size())
            #exit()
            self.register_buffer("confounder_feat",conf_tensor)
            joint_space_dim = 128
            dropout_v = 0.1
            self.confounder_W_q = nn.Linear(512, joint_space_dim)
            self.confounder_W_k = nn.Linear(512, joint_space_dim)
            self._fc2 = nn.Linear(1024, self.n_classes) #1024
            self.norm2 = nn.LayerNorm(1024)             #1024


    def forward(self, feats):

        ## h = kwargs['data'].float() #[B, n, 1024]
        
        h = feats.unsqueeze(0)
        #print(h.shape)
        
        h = self._fc1(h) #[B, n, 512]
        #print('h', h.shape)
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        #cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        #h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        #---->PPEG
        #h = self.pos_layer(h, _H, _W) #[B, N, 512]
        
        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]
        h_p = h

        #---->cls_token
        h_not_norm = h[:,0]
        A = None
        if self.confounder_path:
            norm =False
            if 'nn' not in self.confounder_path[0]: # un normalized 
                h = self.norm(h)[:,0]
                device = h.device
                #print("h", h.size())
                #exit()
                bag_q = self.confounder_W_q(h)
                conf_k = self.confounder_W_k(self.confounder_feat)
                A = torch.mm(conf_k, bag_q.transpose(0, 1))
                A = F.softmax( A / torch.sqrt(torch.tensor(conf_k.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
                conf_feats = torch.mm(A.transpose(0, 1), self.confounder_feat) # compute bag representation, B in shape C x V
                h = torch.cat((h,conf_feats),dim=1)
                # print('not norm')
            
            else:
                device = h_not_norm.device
                bag_q = self.confounder_W_q(h_not_norm)
                conf_k = self.confounder_W_k(self.confounder_feat)
                A = torch.mm(conf_k, bag_q.transpose(0, 1))
                A = F.softmax( A / torch.sqrt(torch.tensor(conf_k.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
                conf_feats = torch.mm(A.transpose(0, 1), self.confounder_feat) # compute bag representation, B in shape C x V
                h = torch.cat((h, conf_feats.unsqueeze(1).repeat(1,h.shape[1],1)),dim=-1)
                h = self.norm2(h)[:,0]
                # print(' norm')
        else:
            h = self.norm(h)#[:,0]

       
        #---->predict
        # print("h", h.size())
        # exit()
        h = self.dimreduction(h)
        h_s = h.squeeze(0)
        A_ = self.attention(h_s)   ## K x N
        bag_A = F.softmax(A_, dim=1).mean(0, keepdim=True)
        #print(A_.shape, h_s.shape)
        bag_feat = torch.mm(bag_A, h_s)
        logits = self.Slide_classifier(bag_feat)
        
        #logits = self._fc2(h) #[B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        patch_feature = self.norm(h_p)#[:,1:]
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, "Bag_feature":h, "A":A, 'h_not_norm':h_not_norm, 'patch_feature': patch_feature}
        
        return results_dict

if __name__ == "__main__":
    data = torch.randn((1, 6000, 1024)).cuda()
    model = TransMIL(n_classes=2).cuda()
    print(model.eval())
    results_dict = model(data = data)
    print(results_dict)