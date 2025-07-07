import timm
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, PatchEmbed
from utils.inc_net import BaseNet
import math
from torch.nn import functional as F
from .zoo import *

def build_promptmodel(modelname='vit_base_patch16_224', Prompt_Token_num=10, VPT_type="Deep", args=None):
    basic_model = timm.create_model(modelname, pretrained=True)#pretrained_cfg_overlay=dict(file=args['model_path'])
    if modelname in ['vit_base_patch16_224']:
        model = VPT_ViT(Prompt_Token_num=Prompt_Token_num, VPT_type=VPT_type, args=args)
    else:
        raise NotImplementedError("Unknown type {}".format(modelname))

    # drop head.weight and head.bias
    basicmodeldict = basic_model.state_dict()
    basicmodeldict.pop('head.weight')
    basicmodeldict.pop('head.bias')

    model.load_state_dict(basicmodeldict, False)

    model.head = torch.nn.Identity()

    model.Freeze()

    return model


class VPT_ViT(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=200, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 embed_layer=PatchEmbed, norm_layer=None, act_layer=None, Prompt_Token_num=1,
                 VPT_type="Shallow", basic_state_dict=None, args=None):

        # Recreate ViT
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                         embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate, embed_layer=embed_layer,
                         norm_layer=norm_layer, act_layer=act_layer)

        print('Using VPT model')
        self.args = args
        # load basic state_dict
        if basic_state_dict is not None:
            self.load_state_dict(basic_state_dict, False)

        self.VPT_type = VPT_type
        if VPT_type == "Deep":
            print("Using Deep Prompt")
            self.TSP = DPrompt(args, 768, args["nb_tasks"] - 1, Prompt_Token_num / 2)
            self.RSP = NDPrompt(args, 768, 1, round(args["init_cls"] * self.args["prompt_pool_num"]), Prompt_Token_num / 2)

        else:  # "Shallow"
            print("Using Shallow Prompt")
            self.TSP = DPrompt(768, num_classes / 2, Prompt_Token_num / 2)
            self.RSP = NDPrompt(768, 1, 20, Prompt_Token_num / 2)

        self.Prompt_Token_num = Prompt_Token_num

    def New_CLS_head(self, new_classes=15):
        self.head = nn.Linear(self.embed_dim, new_classes)

    def Freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        # self.TIP.requires_grad = True
        try:
            for name,param in self.TSP.named_parameters():
                param.requires_grad = True
            for param in self.RSP.parameters():
                param.requires_grad = True
        except:
            pass

    def Freeze_new(self):
        for param in self.parameters():
            param.requires_grad = False

        # self.TIP.requires_grad = True
        try:
            for name,param in self.TSP.named_parameters():
                param.requires_grad = True
        except:
            print("TSP cant grad")
            pass

    def obtain_prompt(self):
        return 0

    def load_prompt(self, prompt_state_dict):
        pass

    def forward_features(self, x,targets=None,train=False,proto=False):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        num_tokens = x.shape[1]
        alphalist=[]
        loss_match=0
        if self.VPT_type == "Deep":
            for i in range(len(self.blocks)):
                x_query = x[:, 0, :]
                RSP = self.RSP.forward(x_query, i, train=train, proto=proto)
                TSP,loss = self.TSP.forward(x_query, i, targets=targets)
                loss_match+=loss
                if TSP is not None:
                    # print(TSP.shape)
                    x = torch.cat([x, TSP], dim=1)
                if RSP is not None:
                    # print(TSP.shape)
                    if x.shape[0]!=RSP.shape[0]:
                        if train or proto:
                            x=x.repeat(int(RSP.shape[0]/x.shape[0]),1,1)
                    x = torch.cat([x, RSP], dim=1)
                x = self.blocks[i](x)[:, :num_tokens]
        else:  # self.VPT_type == "Shallow"
            TIP = self.TIP.expand(x.shape[0], -1, -1)
            x_query = x[:, 0, :]
            TSP, alpha = self.TSP.forward(x_query, 0)
            RSP = self.RSP.forward(x_query, 0)
            Prompt_Tokens = TIP
            if TSP is not None:
                alphalist.append(alpha)
                # print(TSP.shape)
                Prompt_Tokens = torch.cat([Prompt_Tokens, TSP], dim=1)
            if RSP is not None:
                alphalist += alpha
                # print(TSP.shape)
                Prompt_Tokens = torch.cat([Prompt_Tokens, RSP], dim=1)
            x = torch.cat((x, Prompt_Tokens), dim=1)
            x = self.blocks(x)[:, :num_tokens ]

        x = self.norm(x)
        return x,loss_match/len(self.TSP.e_layers)

    def forward(self, x, targets=None,train=False,proto=False):
        x,loss_match = self.forward_features(x, targets=targets,train=train, proto=proto)
        x = x[:, 0, :]
        return x,loss_match

    def forward_midfeature(self, x, perturb_var=0, train=False, proto=False):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        num_tokens = x.shape[1]
        query_list = []
        for i in range(len(self.TSP.e_layers) + self.TSP.e_layers[0]):
            x_query = x[:, 0, :]
            RSP = self.RSP.forward(x_query, i)
            TSP, a = self.TSP.forward(x_query, i)
            if TSP is not None:
                # print(TSP.shape)
                x = torch.cat([x, TSP], dim=1)
                query_list.append(x_query.unsqueeze(0))
            if RSP is not None:
                # print(TSP.shape)
                if x.shape[0] != RSP.shape[0]:
                    if train or proto:
                        x = x.repeat(int(RSP.shape[0] / x.shape[0]), 1, 1)
                x = torch.cat([x, RSP], dim=1)
            x = self.blocks[i](x)[:, :num_tokens]
        query_list = torch.cat(query_list, dim=0)
        return query_list
class SimpleVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            fc.old_out=nb_output
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc


    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        x,loss=self.backbone(x)
        return x

    def forward(self, x, targets=None,train=False, proto=False):
        x,loss_match = self.backbone(x, targets=targets,train=train, proto=proto)
        out = self.fc(x)
        out.update({"features": x})
        out.update({"loss_match": loss_match})
        return out

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.old_out=0
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        if self.old_out==0:
            weight=self.weight
            # out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        else:
            weight=torch.cat((self.weight[:self.old_out].detach().clone(),self.weight[self.old_out:,]))
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(weight, p=2, dim=1))
        if self.to_reduce:
            # Reduce_proxy
            out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}

def reduce_proxies(out, nb_proxy):
    if nb_proxy == 1:
        return out
    bs = out.shape[0]
    nb_classes = out.shape[1] / nb_proxy
    assert nb_classes.is_integer(), 'Shape error'
    nb_classes = int(nb_classes)

    simi_per_class = out.view(bs, nb_classes, nb_proxy)
    attentions = F.softmax(simi_per_class, dim=-1)

    return (attentions * simi_per_class).sum(-1)