import timm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange, repeat


####################################################
out_channel = {'alexnet': 256, 'vgg16': 512, 'vgg19': 512, 'vgg16_bn': 512, 'vgg19_bn': 512,
               'resnet18': 512, 'resnet34': 512, 'resnet50': 2048, 'resnext50_32x4d': 2048,
               'resnext101_32x8d': 2048, 'mobilenet_v2': 1280, 'mobilenet_v3_small': 576,
               'mobilenet_v3_large': 960 ,'mnasnet1_3': 1280, 'shufflenet_v2_x1_5': 1024,
               'squeezenet1_1': 512, 'efficientnet-b0': 1280, 'efficientnet-l2': 5504,
               'efficientnet-b1': 1280, 'efficientnet-b2': 1408, 'efficientnet-b3': 1536,
               'efficientnet-b4': 1792, 'efficientnet-b5': 2048, 'efficientnet-b6': 2304,
               'efficientnet-b7': 2560, 'efficientnet-b8': 2816, 'vit_deit_small_patch16_224': 384}

feature_map = {'alexnet': -2, 'vgg16': -2,  'vgg19': -2, 'vgg16_bn': -2,  'vgg19_bn': -2,
               'resnet18': -2, 'resnet34': -2, 'resnet50': -2, 'resnext50_32x4d': -2,
               'resnext101_32x8d': -2, 'mobilenet_v2': 0, 'mobilenet_v3_large': -2,
               'mobilenet_v3_small': -2, 'mnasnet1_3': 0, 'shufflenet_v2_x1_5': -1,
               'squeezenet1_1': 0, 'vit_deit_small_patch16_224': 'inf'}

diff_fc_layer = ['mobilenet_v2', 'mnasnet1_3', 'shufflenet_v2_x1_5']
####################################################

class VanillaModel(nn.Module):
    def __init__(self, backbone):
        super(VanillaModel, self).__init__()

        self.backbone  = backbone
        # Vision Transformer
        if 'vit' in self.backbone:
            model = timm.create_model(self.backbone, pretrained=False, num_classes=0)
            self.feature_extract = model
        else:
            model = getattr(models, self.backbone)
            model = model(pretrained=False)
            # Seperate feature and classifier layers
            self.feature_extract = nn.Sequential(*list(model.children())[0]) if feature_map[self.backbone]==0 \
                                   else nn.Sequential(*list(model.children())[:feature_map[self.backbone]])

    def forward(self, x):
        feature = self.feature_extract(x)
        feature = F.adaptive_avg_pool2d(feature, 1)
        out     = torch.flatten(feature, 1)
        return out


class DINO(nn.Module):
    """
    Parameters
    ----------
    backbone : Either ViTs or conventional deep models
        if ViT: timm.models.vision_transformer.VisionTransformer
    """
    def __init__(self, backbone):
        super().__init__()
        # Vision Transformer
        if 'vit' in backbone:
            model = timm.create_model(backbone, pretrained=False, num_classes=0)
            self.backbone = model
        else:
            model  = getattr(models, backbone)
            model  = model(pretrained=False)
            layers = list(model.children())[0] if feature_map[backbone]==0 \
                else list(model.children())[:feature_map[backbone]]
            layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
            self.backbone = nn.Sequential(*layers)

    def forward(self, x):
        repr = self.backbone(x)  # (n_samples * n_crops, in_dim)
        return repr


class KimiaNet(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.densenet121(pretrained=False)

        model.features = nn.Sequential(model.features,
                                       nn.AdaptiveAvgPool2d((1,1)))
        self.model = model.features

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        return x


class SimCLR(nn.Module):
    def __init__(self, backbone):
        super(SimCLR, self).__init__()

        # Vision Transformer
        if 'vit' in backbone:
            model = timm.create_model(backbone, pretrained=False, num_classes=0)
            self.feature_extract = model

        else:
            model = getattr(models, backbone)
            model = model(pretrained=False)

            self.feature_extract = nn.Sequential(*list(model.children())[0]) if feature_map[backbone]==0 \
                                   else nn.Sequential(*list(model.children())[:feature_map[backbone]])

    def forward(self, x):
        x = self.feature_extract(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        return x


class DeepMIL(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = 128
        torch.autograd.set_detect_anomaly(True)
        self.attention  = nn.Sequential(nn.Linear(out_channel[cfg['backbone']], dim),
                                       nn.Tanh(),
                                       nn.Linear(dim, 1))
        self.classifier = nn.Sequential(nn.Linear(out_channel[cfg['backbone']], dim),
                                       nn.ReLU(),
                                       nn.Linear(dim, cfg['num_classes']))

    def forward(self, x):
        """
        x   (input)            : B (batch size) x K (nb_patch) x out_channel
        A   (attention weights): B (batch size) x K (nb_patch) x 1
        M   (weighted mean)    : B (batch size) x out_channel
        out (final output)     : B (batch size) x num_classes
        """
        A = self.attention(x)
        A = A.masked_fill((x == 0).all(dim=2).reshape(A.shape), -9e15) # filter padded rows
        A = F.softmax(A, dim=1)   # softmax over K
        M = torch.einsum('b k d, b k o -> b o', A, x) # d is 1 here
        out = self.classifier(M)
        return A, out

class VarMIL(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        dim = 128
        torch.autograd.set_detect_anomaly(True)
        self.device     = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.attention  = nn.Sequential(nn.Linear(out_channel[backbone], dim),
                                       nn.Tanh(),
                                       nn.Linear(dim, 1))
        self.classifier = nn.Sequential(nn.Linear(2*out_channel[backbone], dim),
                                       nn.ReLU(),
                                       nn.Linear(dim, num_classes))

    def forward(self, x):
        """
        x   (input)            : B (batch size) x K (nb_patch) x out_channel
        A   (attention weights): B (batch size) x K (nb_patch) x 1
        M   (weighted mean)    : B (batch size) x out_channel
        S   (std)              : B (batch size) x K (nb_patch) x out_channel
        V   (weighted variance): B (batch size) x out_channel
        nb_patch (nb of patch) : B (batch size)
        M_V (concate M and V)  : B (batch size) x 2*out_channel
        out (final output)     : B (batch size) x num_classes
        """
        b, k, c = x.shape
        A = self.attention(x)
        A = A.masked_fill((x == 0).all(dim=2).reshape(A.shape), -9e15) # filter padded rows
        A = F.softmax(A, dim=1)                                        # softmax over K
        M = torch.einsum('b k d, b k o -> b o', A, x)                  # d is 1 here
        S = torch.pow(x-M.reshape(b,1,c), 2)
        V = torch.einsum('b k d, b k o -> b o', A, S)
        nb_patch = (torch.tensor(k).expand(b)).to(self.device)
        nb_patch = nb_patch - torch.sum((x == 0).all(dim=2), dim=1)    # filter padded rows
        nb_patch = nb_patch / (nb_patch - 1)                           # I / I-1
        nb_patch = torch.nan_to_num(nb_patch, posinf=1)                # for cases, when we have only 1 patch (inf)
        V = V * nb_patch[:, None]                                      # broadcasting
        M_V = torch.cat((M, V), dim=1)
        out = self.classifier(M_V)
        return A, out


class SelfAttention(nn.Module):
    '''
    Modified From: https://theaisummer.com/einsum-attention
    '''
    def __init__(self, cfg):
        super().__init__()
        dim = out_channel[cfg['backbone']]
        print(dim)
        torch.autograd.set_detect_anomaly(True)
        self.to_qvk = nn.Linear(dim, dim * 3, bias=False)
        self.scale_factor = dim ** -0.5
        # self.classifier = nn.Sequential(nn.Linear(out_channel[cfg['backbone']], dim),
        #                                nn.ReLU(),
        #                                nn.Linear(dim, cfg['num_classes']))

    def forward(self, x):
        """
        x   (input)            : B (batch size) x K (nb_patch) x out_channel
        A   (attention weights): B (batch size) x K (nb_patch) x 1
        M   (weighted mean)    : B (batch size) x out_channel
        out (final output)     : B (batch size) x num_classes
        """
        b, k_, o = x.shape
        print(x)
        print(torch.sum(x, dim=-1))
        print(k_, type(k_))
        print('x, qkv')
        qkv = self.to_qvk(x)
        print(x.shape, qkv.shape)
        q, k, v = tuple(rearrange(qkv, 'b t (d k) -> k b t d ', k=3))
        print('q, k, v')
        print(q.shape, k.shape, v.shape)
        scaled_dot_prod = torch.einsum('b i d , b j d -> b i j', q, k) * self.scale_factor
        print('scaled_dot_prod')
        print(scaled_dot_prod.shape)
        mask = (x == 0).all(dim=2)
        # mask = torch.logical_or(repeat(mask, 'm n -> m n k', k=k_),
        #                         repeat(mask, 'm n -> m k n', k=k_))
        mask = repeat(mask, 'm n -> m k n', k=k_)
        print('mask')
        print(mask)
        print(mask.shape)
        scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -9e15) # filter padded rows
        print(scaled_dot_prod)
        A = torch.softmax(scaled_dot_prod, dim=-1)
        print(A)
        print(A.shape)
        A = torch.einsum('b i j , b j d -> b i d', A, v)
        print('A')
        print(A)
        print(torch.sum(A, dim=-1))
        outputs = A.sum(dim=1)
        print(outputs)
        print(outputs.shape)
        raise ValueError


        return

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg['model'] == 'DeepMIL':
            self.model = DeepMIL(cfg)
        elif cfg['model'] == 'VarMIL':
            self.model = VarMIL(cfg)
        elif cfg['model'] == 'SelfAttention':
            self.model = SelfAttention(cfg)
        else:
            raise NotImplementedError()

    def forward(self, x):
        """
        x (input) : B (batch size) x K (nb_patch) x out_channel
        """
        attention, out = self.model(x)
        return attention, out

if __name__ == "__main__":
    model = Model({'backbone': "vit_deit_small_patch16_224",
                   'num_classes': 2})
    print(model)
    print(model(torch.rand(4,3,224,224)).shape)
