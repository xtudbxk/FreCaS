import sys
import math
import types
from functools import partial
from loguru import logger
logger = logger.opt(lazy=True)

import torch
import torch.nn as nn

import diffusers
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, StableDiffusion3Pipeline, StableDiffusionXLPipeline

from utils.wrapper_utils import wrap, unwrap, obtain_origin
from utils.sd_utils import  vae_encode, vae_decode

def freq_decompose(latent, tsize=(128,128), mode="bilinear"):
    outs = []
    current_latent = latent
    _, _, h, w = current_latent.shape
    cl_low = torch.nn.functional.interpolate(current_latent, size=tsize, mode=mode)
    cl_low = torch.nn.functional.interpolate(cl_low, size=(h,w), mode=mode)
    cl_high = current_latent - cl_low

    outs.append(cl_high)
    outs.append(cl_low)
    return outs

# ---- freqgs wrapper ----
def facfg_wrapper(pipeline, name="freqgs_wrapper", gs=1.0, sizes=((128,128),(256,256)), gsw=None, mode="bilinear"):
    if isinstance(pipeline, StableDiffusion3Pipeline):
        network = pipeline.transformer
        is_sd3 = True
    elif isinstance(pipeline, (StableDiffusionPipeline, StableDiffusionXLPipeline)):
        network = pipeline.unet
        is_sd3 = False
    else:
        raise Exception(f"unsupported pipeline {type(pipeline)}")
    sche = pipeline.scheduler

    def facfg_forward(self, hidden_states, timestep, encoder_hidden_states, **kwargs):
        hs, t, ehs = hidden_states, timestep, encoder_hidden_states
        if not pipeline.scheduler.is_first_stage():
            # --- origin result ---
            out = obtain_origin(self, name, "forward")(hs,timestep=t,encoder_hidden_states=ehs, **kwargs)[0]

            if gs > 1.0:
                sh, sw = sizes[pipeline.scheduler._msp_i-1]

                # freq gs 
                uncond_out, cond_out = torch.chunk(out, 2)
                uncond_out_decomposed = freq_decompose(uncond_out, mode=mode, tsize=(sh, sw))
                cond_out_decomposed = freq_decompose(cond_out, mode=mode, tsize=(sh,sw))

                # logger.trace(f"gs_weight: {gsw}")
                out = sum([_uncout + _gsw*(_cout-_uncout) for _uncout, _cout, _gsw in zip(uncond_out_decomposed, cond_out_decomposed, gsw)])
                out = torch.cat([0.0*out, out/gs], dim=0)
            return (out, )
        else:
            out = obtain_origin(self, name, "forward")(hs,timestep=t,encoder_hidden_states=ehs, **kwargs)
            # logger.trace(f"no further operation on cfg while gs {gs}")
            return out

    wrap(network, name, 'forward', facfg_forward)
    return pipeline, name

def facfg_unwrapper(pipeline, name="facfg_wrapper"):
    if isinstance(pipeline, StableDiffusion3Pipeline):
        network = pipeline.transformer
    elif isinstance(pipeline, (StableDiffusionPipeline, StableDiffusionXLPipeline)):
        network = pipeline.unet
    else:
        raise Exception(f"unsupported pipeline {type(pipeline)}")

    unwrap(network, name, "forward")
    return pipeline
# ---- ends ----
