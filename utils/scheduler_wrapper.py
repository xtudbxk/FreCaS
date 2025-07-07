import torch
from loguru import logger

from diffusers import DDIMScheduler, StableDiffusion3Pipeline

from utils.sd_utils import vae_encode, vae_decode
from utils.wrapper_utils import wrap, unwrap, obtain_origin

# for sd21/sdxl
def get_next_starttimestep(scheduler, ot, os=1024, ts=2048, factor=2):
    s = (os/ts)**factor
    last_alphas = scheduler.alphas_cumprod[ot]
    alphas_cumprod = scheduler.alphas_cumprod
    total_len = alphas_cumprod.shape[0]
    target_alphas = s*last_alphas/(1+(s-1)*last_alphas)
    timestep = total_len - torch.searchsorted(torch.flip(alphas_cumprod, dims=(0,)), target_alphas, right=True)
    return int(timestep)

def get_next_starttimestep_rf(scheduler, ot, os=1024, ts=2048):
    toshift = (ts / os)**0.5
    newt = 1000*toshift * ot / (1000 + (toshift-1)*ot )
    return int(newt)

def scheduler_wrapper(pipeline, schedulers, name="_scheduler_wrapper", gamma=2.0):
    pipeline.scheduler._msp_schedulers = schedulers

    # --- some assist funcs ---
    def is_first_stage(self):
        return self._msp_i == 0
    def is_last_stage(self):
        return self._msp_i == len(schedulers)-1
    def is_first_timestep(self, t):
        return t == schedulers[self._msp_i]["timesteps"][0]
    def is_last_timestep(self, t):
        return t == schedulers[self._msp_i]["timesteps"][-1]

    wrap(pipeline.scheduler, name, 'is_first_stage', is_first_stage)
    wrap(pipeline.scheduler, name, 'is_last_stage', is_last_stage)
    wrap(pipeline.scheduler, name, 'is_first_timestep', is_first_timestep)
    wrap(pipeline.scheduler, name, 'is_last_timestep', is_last_timestep)

    # ---- a function to init all the timesteps at the begining ----
    def set_timesteps(self, num_inference_steps, **kwargs):
        if isinstance(self, DDIMScheduler):
            self.config.timestep_spacing = "linspace"
        obtain_origin(self, name, "set_timesteps")(1000, device=pipeline.vae.device)
        self._msp_i = 0

        # --- obtain the timesteps for each stage ---
        timesteps = None
        st = 999

        for si, sche in enumerate(schedulers):
            et, ts = sche['endtime'], sche['steps']
            indexs = 999-torch.linspace(st, et, steps=ts).to(torch.int32).to(pipeline.vae.device)
            if timesteps is None:
                timesteps = self.timesteps[indexs]
            else:
                timesteps = torch.cat([timesteps,self.timesteps[indexs]])
            # for locating the previous timestep in sd2/sdxl
            sche["num_inference_steps"] = int(1000/(st-et)*(ts-1))
            # for locating the delta t in sd3
            sche["indexs"] = torch.cat([indexs, indexs[-2:-1]+1])

            sche["timesteps"] = self.timesteps[indexs]

            # obtain st for next pipeline
            if si < len(schedulers)-1: # except the last one
                osh, osw = schedulers[si]["size"]
                tsh, tsw = schedulers[si+1]["size"]
                if tsh/osh > tsw/osw:
                    os, ts = osh, tsh
                else:
                    os, ts = osw, tsw

                if isinstance(pipeline, StableDiffusion3Pipeline):
                    st = get_next_starttimestep_rf(self, et, os, ts)
                else:
                    st = get_next_starttimestep(self, et, os, ts, gamma)
            # --- ends ---

        self.timesteps = timesteps
        # logger.info(f'all timesteps: {self.timesteps}')

    wrap(pipeline.scheduler, name, 'set_timesteps', set_timesteps)
    # ---- ends ----

    # ---- for step func ----
    def step(self, model_output, timestep, sample, **kwargs):

        if self.is_first_stage() and self.is_first_timestep(timestep):
            obtain_origin(self, name, "set_timesteps")(1000, device=pipeline.vae.device)
            self.num_inference_steps = schedulers[self._msp_i]["num_inference_steps"]
            self._indexs = schedulers[self._msp_i]["indexs"]
            self._indexs_i = 0

        # --- origin step ---
        _return_dict = kwargs.get("return_dict", True)
        kwargs["return_dict"] = True
        out = obtain_origin(self, name, "step")(model_output, timestep, sample, **kwargs)
        # --- ends ---

        # --- set _msp_i ---
        if self.is_last_timestep(timestep) and not self.is_last_stage():
            self._msp_i += 1
            self._msp_i %= len(schedulers)

            # set timesteps for a new stage
            self.num_inference_steps = schedulers[self._msp_i]["num_inference_steps"]
            self._indexs = schedulers[self._msp_i]["indexs"]
            self._indexs_i = 0

            factor = schedulers[self._msp_i]['size'][0] / schedulers[self._msp_i-1]['size'][0]

            # denoise
            latents = out.pred_original_sample

            # decode
            images = vae_decode(pipeline, latents)
            images = torch.clip(images, -1, 1)

            # interpolate
            images = torch.nn.functional.interpolate(images, scale_factor=factor, mode="nearest")

            # encode
            latents = vae_encode(pipeline, images)

            # diffuse
            noise = torch.randn_like(latents)
            out.prev_sample = self.add_noise(latents, timesteps=schedulers[self._msp_i]["timesteps"][0], noise=noise)

        # --- ends ---

        if _return_dict: return out
        else: return (out.prev_sample, )

    wrap(pipeline.scheduler, name, 'step', step)
    # ---- ends ----

    return pipeline, name

def scheduler_unwrapper(pipeline, name="scheduler_unwrapper"):
    unwrap(pipeline.scheduler, name, "step")
    unwrap(pipeline.scheduler, name, 'set_timesteps')
