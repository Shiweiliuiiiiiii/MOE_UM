from functools import partial

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import os 
import numpy as np
import timm

# from timm.models.vision_transformer import PatchEmbed, Block

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import VisionTransformer
from parallel_experts import MoE

from moe import MoE as MMoE
# from mixture_of_experts import MoE as newMoE

from parallel_experts import RandomMoE

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, head_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        if head_dim is None:
            head_dim = dim // num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))

        # For rare cases, the attention weights are inf due to the mix-precision training.
        # We clamp the tensor to the max values of the current data type
        # This is different from MAE training as we don't observe such cases on image-only MAE.
        if torch.isinf(attn).any():
            clamp_value = torch.finfo(attn.dtype).max-1000
            attn = torch.clamp(attn, min=-clamp_value, max=clamp_value)

        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# MLP hidden/4 topk=4
class MoEAttention(nn.Module):
    def __init__(self, dim, num_experts=24, num_heads=8, head_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
        sample_topk=2, cvloss=0, switchloss=0.01 * 10, zloss=0.001 * 1, moe_type='normal'):
        super().__init__()
        self.num_experts = num_experts
        self.sample_topk = sample_topk

        self.num_heads = num_heads
        if head_dim is None:
            head_dim = dim // num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.moe_type = moe_type

        if moe_type == 'random':
            self.q_proj = RandomMoE(dim, head_dim, num_experts, num_heads, cvloss=cvloss, switchloss=switchloss, zloss=zloss)
        elif moe_type == 'FLOP': # use this to evaluate FLOPs
            self.att_experts = [
                nn.Sequential(
                    nn.Linear(dim, head_dim),
                )
                for _ in range(num_experts)
            ]
            self.q_proj = MMoE(dim, self.att_experts, num_heads, dropout=0., concat=True)
            self.out_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(head_dim, dim),
                    nn.Dropout(0.)
                )
                for _ in range(num_experts)
            ])
        else:
            self.q_proj = MoE(dim, head_dim, num_experts, num_heads, cvloss=cvloss, switchloss=switchloss, zloss=zloss)

        self.kv_proj = nn.Sequential(
            nn.Linear(dim, head_dim * 2),
        )

        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):

        B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        if self.moe_type == 'FLOP':
            q, aux_loss = self.q_proj(x, multiply_by_gates=False, sample_topk=self.sample_topk)
        else:
            q, aux_loss = self.q_proj.map(x, sample_topk=self.sample_topk)
        k, v = self.kv_proj(x).chunk(2, dim=-1)

        q = q.reshape(B, N, self.num_heads, self.head_dim)
        k = k.reshape(B, N, self.head_dim)
        v = v.reshape(B, N, self.head_dim)

        attn = torch.einsum('bihd,bjd->bhij', q, k) * self.scale
        # attn = attn.premute(0,3,1,2) # b, h, i, j

        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))

        # For rare cases, the attention weights are inf due to the mix-precision training.
        # We clamp the tensor to the max values of the current data type
        # This is different from MAE training as we don't observe such cases on image-only MAE.
        if torch.isinf(attn).any():
            clamp_value = torch.finfo(attn.dtype).max-1000
            attn = torch.clamp(attn, min=-clamp_value, max=clamp_value)

        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        attn = torch.einsum('bhij,bjd->bihd', attn, v)

        if self.moe_type == 'FLOP':
            x = self.q_proj.dispatch(
                    attn.reshape(B, N, self.num_heads, self.head_dim).contiguous(), 
                    self.out_proj
                )
        else:
            x = self.q_proj.reduce(attn)
        x = self.proj_drop(x)
        return x, aux_loss

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

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, head_dim=None, init_values=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, head_dim=head_dim, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MoEnhanceBlock(nn.Module):

    def __init__(self, dim, num_heads, num_attn_experts=24, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 num_ffd_experts=16, ffd_heads=2, ffd_noise=True,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, head_dim=None, init_values=None, z_weight=0.000,
                 post_layer_norm=False,
                 cvloss=0, switchloss=0.01 * 1, zloss=0.001 * 1, sample_topk=0, moe_type='normal'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if num_attn_experts == 0:
            self.attn = Attention(
                dim, num_heads=num_heads, head_dim=head_dim, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        else:
            self.attn = MoEAttention(
                dim, num_heads=num_heads, num_experts=num_attn_experts, head_dim=head_dim, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                cvloss=cvloss, switchloss=switchloss, zloss=zloss, sample_topk=sample_topk, moe_type=moe_type)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.moe_type = moe_type

        if moe_type == 'FLOP':
            ffd_exports = [
                    nn.Sequential(
                    # nn.LayerNorm(dim),
                    nn.Linear(dim, mlp_hidden_dim // ffd_heads),
                    nn.GELU(),
                    # nn.Dropout(dropout),
                    nn.Linear(mlp_hidden_dim // ffd_heads, dim),
                    # nn.Dropout(dropout)
                    # nn.LayerNorm(dim),
                    )
                    for _ in range(num_ffd_experts)
                ]
            self.mlp = MMoE(dim, ffd_exports, ffd_heads, 0.)
        else:
            self.mlp = MoE(dim,
                    mlp_hidden_dim // ffd_heads, num_ffd_experts, ffd_heads,
                    bias=True,
                    cvloss=cvloss,
                    switchloss=switchloss,
                    zloss=zloss,
                    activation=nn.Sequential(
                        nn.GELU(),
                        # self.dropout_module Remove dropout for now
                    ),
                    noisy_gating=ffd_noise
                )
        self.post_layer_norm = post_layer_norm
        assert z_weight == 0

    def forward(self, x, mask=None):
        if self.post_layer_norm:
            if hasattr(self.attn, 'moe_type'):
                y, router_loss = self.attn(x, mask=mask)
            else:
                y = self.attn(x, mask=mask)
                router_loss = 0

            x = x + self.drop_path(y)
            x = self.norm1(x)

            if self.moe_type == 'TB':
                y, *aux = self.mlp(x)
                x = x + self.drop_path(y)
                x = self.norm2(x)
                return x, aux
                
            y, aux_loss = self.mlp(x)
            x = x + self.drop_path(y)
            x = self.norm2(x)
            return x, router_loss + aux_loss
        else:
            x = self.norm1(x)
            if hasattr(self.attn, 'moe_type'):
                y, router_loss = self.attn(x, mask=mask)
            else:
                y = self.attn(x, mask=mask)
                router_loss = 0
            x = x + self.drop_path(y)

            if self.moe_type == 'TB':
                y, *aux = self.mlp(self.norm2(x))
                x = x + self.drop_path(y)
                return x, aux

            y, aux_loss = self.mlp(self.norm2(x))
            x = x + self.drop_path(y)
            return x, router_loss + aux_loss

class VisionTransformerMoE(VisionTransformer):
    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None, 
                 num_attn_experts=48, head_dim=None, att_w_topk_loss=0.0, att_limit_k=0, 
                 num_ffd_experts=16, ffd_heads=2, ffd_noise=True,
                 moe_type='normal',
                 switchloss=0.01 * 1, zloss=0.001 * 1, w_topk_loss= 0.0, limit_k=0, 
                 noisy_gating=True,
                 post_layer_norm=False,
                 twice_mlp=False,
                 twice_attn=False,
                 acc_aux_loss=False,
                 **kwargs):
        super(VisionTransformerMoE, self).__init__(
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,  
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer, 
            **kwargs)

        self.moe_type = moe_type
        self.depth = depth

        self.acc_aux_loss = acc_aux_loss
        self.w_topk_loss = w_topk_loss
        self.switchloss = switchloss
        self.zloss = zloss

        self.R = {
                'depth': depth,
                'head_dim': head_dim,
                'noisy_gating': noisy_gating,
                'ffd_heads': ffd_heads, 'ffd_noise': ffd_noise,
                'dim': embed_dim, 'num_heads': num_heads, 'mlp_ratio': mlp_ratio, 'qkv_bias': qkv_bias,
                'drop': drop_rate, 'attn_drop': attn_drop_rate, 'drop_path_rate': drop_path_rate, 'norm_layer': norm_layer,
                'moe_type': moe_type, 'switchloss': switchloss, 'zloss': zloss, 'w_topk_loss': w_topk_loss, 'limit_k': limit_k,
                'post_layer_norm': post_layer_norm,
                'twice_mlp': twice_mlp,
                'twice_attn': twice_attn,
                }

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.twice_mlp = twice_mlp

        self.blocks = nn.Sequential(*[
            MoEnhanceBlock(
                num_attn_experts=num_attn_experts, head_dim=head_dim,
                num_ffd_experts=num_ffd_experts, ffd_heads=ffd_heads, ffd_noise=ffd_noise,
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                moe_type=moe_type,switchloss=switchloss, zloss=zloss,
                )
            for i in range(depth)])

        self.apply(self._init_weights)

    def frozen(self):
        self.patch_embed.requires_grad = False
        self.pos_embed.requires_grad = False
        self.cls_token.requires_grad = False
        for blk in self.blocks:
            blk.attn.kv_proj.requires_grad = False
            blk.attn.q_proj.experts.requires_grad = False
            blk.attn.q_proj.output_experts.requires_grad = False

            blk.mlp.experts.requires_grad = False
            blk.mlp.output_experts.requires_grad = False

    def moa_init_weight(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.fill_(0.00)
  
    def get_router_loss(self):
        router_loss = 0
        for blk in self.blocks:
            if hasattr(blk.attn, 'num_experts'):
                aux_loss = blk.attn.q_proj.get_aux_loss_and_clear()
                router_loss = router_loss + aux_loss

            if hasattr(blk.mlp, 'num_experts'):
                aux_loss = blk.mlp.get_aux_loss_and_clear()
                router_loss = router_loss + aux_loss
        return router_loss

    def all_clear(self):
        for blk in self.blocks:
            aux_loss = blk.attn.q_proj.init_aux_statistics()
            if hasattr(blk.mlp, 'num_experts'):
                aux_loss = blk.mlp.init_aux_statistics()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # apply Transformer blocks
        router_loss = 0
        for i, blk in enumerate(self.blocks):
            x, aux_loss = blk(x)
            router_loss = router_loss + aux_loss

            # if i == len(self.blocks) - 1:
            #     x = x[:, 1:, :].mean(dim=1)
            #     x = self.fc_norm(x)
        
        if self.acc_aux_loss:
            router_loss = self.get_router_loss()
        
        return x, router_loss

    def forward(self, x):

        output, router_loss = self.forward_features(x)
        output = self.forward_head(output)

        return output, router_loss


def vit_tiny(**kwargs): # 6.43M 
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(**kwargs): # 23.48M 4.6G
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# this one works for upcycle function
def vit_moe_mlp16E4_small(**kwargs): # 67.37M 5.21G
    model = VisionTransformerMoE(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=0, head_dim=384//6,
        num_ffd_experts=16, ffd_heads=1, ffd_noise=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_topk_moe_mlp16E4_small(**kwargs): # 67.37M 5.21G
    model = VisionTransformerMoE(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        num_attn_experts=6, head_dim=384//6 * 2, w_topk_loss=0.1, limit_k=4, 
        num_ffd_experts=16, ffd_heads=4, ffd_noise=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def load_from_dense_pretrained_to_moe(moe_model, pretrained_weights, num_experts, **kwargs):
    if not type(pretrained_weights) == str:
        dense_state_dict = pretrained_weights
    elif "https" in pretrained_weights:
        checkpoint = torch.hub.load_state_dict_from_url(url=pretrained_weights,
                                                              map_location="cpu",
                                                              check_hash=True)
        dense_state_dict = checkpoint["model"]
    else:
        dense_state_dict = torch.load(pretrained_weights)

    moe_state_dict = moe_model.state_dict()

    for k, v in dense_state_dict.items():
        if k in moe_state_dict:
            moe_state_dict[k].data.copy_(v)
        elif "mlp" in k:
            mapped_key, mapped_value = map_dense2moe(k, v, num_experts)
            # print(v.shape, mapped_value.shape, moe_state_dict[mapped_key].shape)
            moe_state_dict[mapped_key].data.copy_(mapped_value)
        else:
            print(f"key {k} not in moe state dict.. are you sure keys are mapped correctly?")

    moe_model.load_state_dict(moe_state_dict)

    return moe_model


def map_dense2moe(key, value, num_experts):
    # fc1 maps to experts
    # fc2 maps to output_experts
    mapped_key = ''
    mapped_value = None
    if 'fc1.weight' in key:
        mapped_key = key.replace('fc1.weight', 'experts.w')
        mapped_value = torch.unsqueeze(value.T, 0).repeat(num_experts, 1, 1)
    elif 'fc1.bias' in key:
        mapped_key = key.replace('fc1.bias', 'experts.b')
        mapped_value = torch.unsqueeze(value, 0).repeat(num_experts, 1)
    elif 'fc2.weight' in key:
        mapped_key = key.replace('fc2.weight', 'output_experts.w')
        mapped_value = torch.unsqueeze(value.T, 0).repeat(num_experts, 1, 1)
    elif 'fc2.bias' in key:
        mapped_key = key.replace('fc2.bias', 'output_experts.b')
        mapped_value = torch.unsqueeze(value, 0).repeat(num_experts, 1)
    else:
        print("K,V not mapped...")
    return mapped_key, mapped_value


if __name__ == '__main__':
    device = 'cuda'
    model = vit_small()
    print("=========== Dense Model State =============")
    # print(list(model.state_dict().keys()))
    print("=========== MoE Model State =============")
    moe_model = vit_moe_mlp16E4_small()
    print(moe_model)
    print("size moe: ", sum(p.numel() for n,p in moe_model.named_parameters()))
    x = torch.randn(16, 3, 224, 224).to(device)
    t = torch.randint(0, 1000, (16,)).to(device)
    print("loading state_dict")
    moe_model = load_from_dense_pretrained_to_moe(moe_model, model.state_dict(), num_experts=16).to(device)

    # print(list(model.state_dict().keys()))
    print("Successfully loaded state_dict")
    print("size dense: ", sum(p.numel() for n,p in model.named_parameters()))
    print("size moe: ", sum(p.numel() for n,p in moe_model.named_parameters()))

    # Test forward
    print("testing forward")
    y, aux_loss = moe_model(x)
    ce_loss = nn.CrossEntropyLoss()(y, t)
    print(y.shape, aux_loss, ce_loss)
    
    # Test backward
    print("testing backward")
    (ce_loss + aux_loss).backward()
    print("Successfully backward")