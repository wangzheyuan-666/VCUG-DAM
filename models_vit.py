# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from timm.models.layers import trunc_normal_

from timm.models.vision_transformer import PatchEmbed, Block


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

class DecoderLinear(nn.Module):
    def __init__(self,
            embed_dim=1024, 
            decoder_embed_dim=1024,
            num_seg = 4,
            num_cls_heads = 1,
            norm_layer=nn.LayerNorm
        ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_seg = num_seg
        self.num_cls_heads = num_cls_heads
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, num_seg*16*16, bias=True) # decoder to patch

        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x):
        x = x[:,self.num_cls_heads:,:]
        x = self.decoder_pred(
            self.decoder_norm(
                self.decoder_embed(x)
            )
        )
        n,l,c = x.shape
        # print(x.shape)
        h = w = int(l**0.5)
        pred = torch.einsum('nlc->ncl',x).contiguous().view(n,c,h,w)
        pred = F.pixel_shuffle(pred, upscale_factor=16)
        return pred


class MaskTransformer(nn.Module):
    def __init__(
            self,
            embed_dim=1024, 
            decoder_embed_dim=512, 
            decoder_depth=1, 
            decoder_num_heads=16,
            mlp_ratio=4., 
            num_seg = 4,
            num_cls_heads = 1,
            norm_layer=nn.LayerNorm 
        ):
        super().__init__()
        self.num_seg = num_seg
        self.num_cls_heads = num_cls_heads
        self.scale = decoder_embed_dim ** -0.5

        # --------------------------------------------------------------------------
        # decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, num_seg*16*16, bias=True) # decoder to patch
        # --------------------------------------------------------------------------


        self.proj_patch = nn.Parameter(self.scale * torch.randn(decoder_embed_dim, decoder_embed_dim))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(decoder_embed_dim, decoder_embed_dim))

        # self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(num_seg)


        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, x):
        # embed tokens
        x = self.decoder_embed(x)

        # add pos embed
        # x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)
        patches, cls_seg_feat = x[:, self.num_cls_heads:], x[:, :self.num_cls_heads]
        pred = self.decoder_pred(patches)

        # n,l,c = patches.shape
        # h = w = int(l**0.5)
        # patches = torch.einsum('nlc->ncl',patches).contiguous().view(n,c,h,w)
        # patches = self.head(patches)
        # patches = torch.einsum('ncl->nlc',patches.view(n,c,-1))

        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        # print(cls_seg_feat.shape, patches.shape)
        masks = patches @ cls_seg_feat.transpose(1, 2)
        # masks = self.mask_norm(masks)

        n,l,c = masks.shape
        h = w = int(l**0.5)
        masks = torch.einsum('nlc->ncl',masks).contiguous().view(n,c,h,w)
        masks = F.interpolate(masks, scale_factor=16, mode="bilinear")
        masks = torch.sigmoid(masks)

        pred = torch.einsum('nlc->ncl',pred).contiguous().view(n,-1,h,w)
        pred = F.pixel_shuffle(pred, upscale_factor=16)

        # print(masks.shape)
        # masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))

        return pred * masks

    def get_attention_map(self, x, layer_id):
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.n_layers}."
            )
        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x)
            else:
                return blk(x, return_attention=True)


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        self.num_seg = 4
        ##########多
        self.num_cls_heads = 1
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + self.num_cls_heads, self.embed_dim))
        self.head = nn.Linear(self.num_features, 16)
        # self.head2 = nn.Linear(self.num_features, 2)
        # self.head3 = nn.Linear(self.num_features, 12)

        self.decoder = MaskTransformer(embed_dim=self.embed_dim)
        # self.decoder = DecoderLinear(embed_dim=self.embed_dim)

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    
    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], self.num_cls_heads, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x
        # if self.dist_token is None:
        #     return self.pre_logits(x[:, 0])
        # else:
        #     return x[:, 0], x[:, 1]

    def forward(self, x, return_classes=False):
        x = self.forward_features(x)

        masks = self.decoder(x)
        # x_seg = x[:,self.num_heads:,:]
        # n,l,c = x_seg.shape
        # # print(x.shape)
        # h = w = int(l**0.5)
        # pred = torch.einsum('nlc->ncl',x_seg).contiguous().view(n,c,h,w)

        # pred = self.up(pred)
        x = self.head(x[:,0,:])
        if return_classes:
            return x
        # x1,x2,x3 = self.head1(x[:,1,:]), self.head2(x[:,2,:]), self.head3(x[:,3,:])
        # x = torch.cat([x1,x2,x3],dim=1)
        return masks, x

        
        # if self.head_dist is not None:
        #     x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
        #     if self.training and not torch.jit.is_scripting():
        #         # during inference, return the average of both classifier predictions
        #         return x, x_dist
        #     else:
        #         return (x + x_dist) / 2
        # else:
        #     x = self.head(x)
        # return x



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



if __name__ == '__main__':
    model = vit_base_patch16()
    a = torch.rand(1,3,224,224)
    b,_ = model(a)
    print(b.shape)