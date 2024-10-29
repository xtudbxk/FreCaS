from loguru import logger

import torch
from diffusers.utils.torch_utils import randn_tensor
from diffusers import FlowMatchEulerDiscreteScheduler

class FlowMatchEulerDiscreteModifiedSchedulerOutput():
    prev_sample: torch.FloatTensor
    pred_original_sample: torch.FloatTensor
    def __init__(self, prev_sample, pred_original_sample):
        FlowMatchEulerDiscreteModifiedSchedulerOutput.prev_sample = prev_sample
        FlowMatchEulerDiscreteModifiedSchedulerOutput.pred_original_sample = pred_original_sample

class FlowMatchEulerDiscreteModifiedScheduler(FlowMatchEulerDiscreteScheduler):
    def add_noise(self, origin_samples, timesteps, noise):
        return self.scale_noise(origin_samples, timesteps, noise=noise)

    def step(self, model_output, timestep, sample, 
             s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"),
             s_noise=1.0, generator=None, return_dict=True,
             ):

        # if self.step_index is None:
            # self._init_step_index(timestep)

        # self._init_step_index(timestep)
        _index = self._indexs[self._indexs_i]
        logger.trace(f"timestep: {timestep}, step_index: {_index} _indexs_i: {self._indexs_i}")        

        # Upcast to avoid precision issues when computing prev_sample
        origin_type = sample.dtype
        sample = sample.to(torch.float32)

        sigma = self.sigmas[_index]

        gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0

        noise = randn_tensor(
            model_output.shape, dtype=model_output.dtype, device=model_output.device, generator=generator
        )

        eps = noise * s_noise
        sigma_hat = sigma * (gamma + 1)

        if gamma > 0:
            sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        # NOTE: "original_sample" should not be an expected prediction_type but is left in for
        # backwards compatibility

        # if self.config.prediction_type == "vector_field":

        denoised = sample - model_output * sigma
        # 2. Convert to an ODE derivative
        derivative = (sample - denoised) / sigma_hat

        dt = self.sigmas[self._indexs[self._indexs_i+1]] - sigma_hat

        prev_sample = sample + derivative * dt
        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        self._indexs_i += 1

        # cast back to original type
        prev_sample = prev_sample.to(origin_type)
        denoised = denoised.to(origin_type)

        if not return_dict:
            return (prev_sample,)

        return FlowMatchEulerDiscreteModifiedSchedulerOutput(prev_sample=prev_sample,pred_original_sample=denoised)
