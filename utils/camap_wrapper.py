import sys
import gc
import types
import math
from functools import partial
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import StableDiffusionPipeline, StableDiffusion3Pipeline, StableDiffusionXLPipeline

from utils.wrapper_utils import wrap, unwrap, obtain_origin

# --- obtain/set atten map wrapper ---
def camap_wrapper(pipeline, camap_w=1.0, mode="bilinear", name="camap_wrapper"):
    if isinstance(pipeline, StableDiffusion3Pipeline):
        network = pipeline.transformer
        is_sd3 = True
    elif isinstance(pipeline, (StableDiffusionPipeline, StableDiffusionXLPipeline)):
        network = pipeline.unet
        is_sd3 = False

    # ===== some wrapper funcs =====
    def network_forward(self, hidden_states, timestep, encoder_hidden_states, **kwargs):
        hs, t, ehs = hidden_states, timestep, encoder_hidden_states
        if len(t.shape) > 0:
            _t = t[0]
        else:
            _t = t

        # --- set collect mode ---
        if pipeline.scheduler.is_last_timestep(_t) and not pipeline.scheduler.is_last_stage():
            self._camap = {} 
            self._collect_mode = True
            # logger.trace("turn on the collect mode")
        else:
            self._collect_mode = False
            # logger.trace("turn off the collect mode")

        # --- set utilizer mode ---
        if not pipeline.scheduler.is_first_stage():
            self._utilizer_mode = True
            # logger.trace("turn on the utilizer mode")
        else:
            self._utilizer_mode = False
            # logger.trace("turn off the utilizer mode")

        # --- forward ---
        ret = obtain_origin(self, name, "forward")(hs,timestep=timestep,encoder_hidden_states=ehs, **kwargs)

        # --- merge all the attention map ---
        if self._collect_mode is True:
            self._camap_fr = None
            for k, _data  in self._camap.items():
                v = _data["values"] / _data["count"]
                if self._camap_fr is None: 
                    self._camap_fr = v
                elif self._camap_fr.shape[-2] > v.shape[-2]:
                    self._camap_fr += F.interpolate(v, size=self._camap_fr.shape[-2:], mode=mode)
                else:
                    self._camap_fr = F.interpolate(frv, size=v.shape[-2:], mode=mode)
                    self._camap_fr += v
            self._camap_fr = self._camap_fr / len(self._camap.keys())
            # logger.trace(f'collect all cross-attention map to {self._camap_fr.shape}')
            del self._camap

        return ret

    wrap(network, name, 'forward', network_forward)
    # ===== ends =====

    # ===== some wrapper funcs for attention layers =====
    def attnlayer_forward(self, *args, layername, **kwargs):

        origin_func = F.scaled_dot_product_attention
        def fake_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):

            out = origin_func(query, key, value, attn_mask, dropout_p, is_causal)
            if network._collect_mode:
                if is_sd3:
                    q_v, k_t = query[:,:,:-154,:], key[:,:,-154:,:]
                    eyematrix = torch.eye(k_t.shape[-2], dtype=k_t.dtype, device=k_t.device)
                    camap = torch.mean(origin_func(q_v, k_t, eyematrix), dim=1, keepdims=True)
                else:
                    eyematrix = torch.eye(key.shape[-2], dtype=key.dtype, device=key.device)
                    camap = torch.mean(origin_func(query, key, eyematrix), dim=1, keepdims=True)

                _, _, s1, s2 = camap.shape
                if (s1,s2) not in network._camap:
                    network._camap[(s1,s2)] = {"values": 0.0, "count": 0}
                network._camap[(s1,s2)]["values"] += camap
                network._camap[(s1,s2)]["count"] += 1

            if network._utilizer_mode is True:
                _, _, s1, _ = out.shape
                _, _, _, s2 = network._camap_fr.shape
                if is_sd3:
                    camap = F.interpolate(network._camap_fr, size=(s1-154, s2), mode=mode)
                    out_o = attnmap @ value[:,:,-154:,:] * 154/4096
                    out[:,:,:-154,:] += out_o*camap_w
                else:
                    camap = F.interpolate(network._camap_fr, size=(s1, s2), mode=mode)
                    out_o = camap @ value
                    out = (1-camap_w)*out + camap_w*out_o

            return out

        F.scaled_dot_product_attention = fake_scaled_dot_product_attention
        out = obtain_origin(self, name, "forward")(*args, **kwargs)
        F.scaled_dot_product_attention = origin_func

        return out

    for layername, module in network.named_modules():
        if layername.endswith(".attn2"): # sd21 or sdxl
            wrap(module, name, 'forward', partial(attnlayer_forward, layername=layername))
        if layername.endswith(".attn"): # sd3
            wrap(module, name, 'forward', partial(attnlayer_forward, layername=layername))

    # ===== ends =====

    return pipeline, name

def camap_unwrapper(pipeline, name="camap_wrapper"):
    if isinstance(pipeline, StableDiffusion3Pipeline):
        network = pipeline.transformer
        is_sd3 = True
    elif isinstance(pipeline, (StableDiffusionPipeline, StableDiffusionXLPipeline)):
        network = pipeline.unet
        is_sd3 = False
    else:
        raise Exception(f"unsupported pipeline {type(pipeline)}")

    unwrap(network, name, "forward")

    for layername, module in network.named_modules():
        if layername.endswith(".attn2"):
            unwrap(module, name, 'forward')
        if layername.endswith(".attn"):
            unwrap(module, name, 'forward')

    return pipeline
# ---- ends ----
