import torch
from utils.wrapper_utils import wrap, unwrap, obtain_origin

from diffusers import StableDiffusionPipeline, StableDiffusion3Pipeline, StableDiffusionXLPipeline

def vae_encode(pipeline, latent):
    if isinstance(pipeline, StableDiffusionPipeline):
        return sd21_vae_encode(pipeline, latent)
    elif isinstance(pipeline, StableDiffusion3Pipeline):
        return sd3_vae_encode(pipeline, latent)
    elif isinstance(pipeline, StableDiffusionXLPipeline):
        return sdxl_vae_encode(pipeline, latent)

def vae_decode(pipeline, latent):
    if isinstance(pipeline, StableDiffusionPipeline):
        return sd21_vae_decode(pipeline, latent)
    elif isinstance(pipeline, StableDiffusion3Pipeline):
        return sd3_vae_decode(pipeline, latent)
    elif isinstance(pipeline, StableDiffusionXLPipeline):
        return sdxl_vae_decode(pipeline, latent)

def sd21_vae_encode(pipeline, latent):
    x0 = pipeline.vae.encode(latent).latent_dist.sample()
    x0 = x0 * pipeline.vae.config.scaling_factor
    return x0

def sd21_vae_decode(pipeline, latent):
    image = pipeline.vae.decode(latent/pipeline.vae.config.scaling_factor, return_dict=False)[0]
    return image

def sd3_vae_encode(pipeline, latent):
    x0 = pipeline.vae.encode(latent).latent_dist.sample()
    x0 = (x0+pipeline.vae.config.shift_factor)*pipeline.vae.config.scaling_factor
    return x0

def sd3_vae_decode(pipeline, latent):
    latent = (latent/pipeline.vae.config.scaling_factor)+pipeline.vae.config.shift_factor
    latent = pipeline.vae.decode(latent, return_dict=False)[0]
    return latent


def sdxl_vae_decode(pipeline, latent):
    origin_dtype = latent.dtype
    latent = latent.to(pipeline.vae.dtype)
    latent = latent / pipeline.vae.config.scaling_factor
    image = pipeline.vae.decode(latent, return_dict=False)[0]
    return image.to(origin_dtype)

def sdxl_vae_encode(pipeline, image):
    origin_dtype = image.dtype
    image = image.to(pipeline.vae.dtype)
    latent = pipeline.vae.encode(image).latent_dist.sample()
    latent = latent * pipeline.vae.config.scaling_factor
    return latent.to(origin_dtype)

def sdxl_vae_wrapper(pipeline, name="_sdxl_vae_wrapper"):

    def sdxl_vae_decode_tf32(self, z, return_dict=True, generator=None):
        origin_dtype = z.dtype
        z = z.to(torch.float32)
        ret = obtain_origin(self, name, "decode")(z, return_dict, generator)
        return (ret[0].to(origin_dtype),)

    wrap(pipeline.vae, name, 'decode', sdxl_vae_decode_tf32)
    pipeline.vae = pipeline.vae.to(torch.float32)

    return pipeline, name

def sdxl_vae_unwrapper(pipeline, name="_sdxl_vae_unwrapper"):
    unwrap(pipeline.vae, name, 'decode')
