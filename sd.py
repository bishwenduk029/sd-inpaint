import random

import PIL.Image
import os
import cv2
import numpy as np
import torch
from diffusers import (
    PNDMScheduler,
    DDIMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

from base import DiffusionInpaintModel
from schema import Config, SDSampler


class SD(DiffusionInpaintModel):
    pad_mod = 8
    min_size = 512

    def init_model(self, device: torch.device, **kwargs):
        from diffusers.pipelines.stable_diffusion import StableDiffusionInpaintPipeline

        HF_AUTH_TOKEN = os.getenv('HF_AUTH_TOKEN')

        model_kwargs = {
            "local_files_only": kwargs.get("local_files_only", kwargs["sd_run_local"])
        }
        model_kwargs.update(
            dict(
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            )
        )

        torch_dtype = torch.float16
        self.model = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_id_or_path,
            torch_dtype=torch_dtype,
            use_auth_token=HF_AUTH_TOKEN,
            **model_kwargs
        )

        self.callback = kwargs.pop("callback", None)

    def forward(self, image, mask, config: Config):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W, 1] 255 means area to repaint
        return: BGR IMAGE
        """

        scheduler_config = self.model.scheduler.config

        if config.sd_sampler == SDSampler.ddim:
            scheduler = DDIMScheduler.from_config(scheduler_config)
        elif config.sd_sampler == SDSampler.pndm:
            scheduler = PNDMScheduler.from_config(scheduler_config)
        elif config.sd_sampler == SDSampler.k_lms:
            scheduler = LMSDiscreteScheduler.from_config(scheduler_config)
        elif config.sd_sampler == SDSampler.k_euler:
            scheduler = EulerDiscreteScheduler.from_config(scheduler_config)
        elif config.sd_sampler == SDSampler.k_euler_a:
            scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler_config)
        elif config.sd_sampler == SDSampler.dpm_plus_plus:
            scheduler = DPMSolverMultistepScheduler.from_config(scheduler_config)
        else:
            raise ValueError(config.sd_sampler)

        self.model.scheduler = scheduler

        if config.sd_mask_blur != 0:
            k = 2 * config.sd_mask_blur + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)[:, :, np.newaxis]

        img_h, img_w = image.shape[:2]

        output = self.model(
            image=PIL.Image.fromarray(image),
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
            mask_image=PIL.Image.fromarray(mask[:, :, -1], mode="L"),
            num_inference_steps=config.sd_steps,
            guidance_scale=config.sd_guidance_scale,
            output_type="np.array",
            callback=self.callback,
            height=img_h,
            width=img_w,
            generator=torch.manual_seed(config.sd_seed),
        ).images[0]

        output = (output * 255).round().astype("uint8")
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output

    def forward_post_process(self, result, image, mask, config):
        if config.sd_match_histograms:
            result = self._match_histograms(result, image[:, :, ::-1], mask)

        if config.sd_mask_blur != 0:
            k = 2 * config.sd_mask_blur + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)
        return result, image, mask

    @staticmethod
    def is_downloaded() -> bool:
        # model will be downloaded when app start, and can't switch in frontend settings
        return True


class SD15(SD):
    name = "sd1.5"
    model_id_or_path = "runwayml/stable-diffusion-inpainting"


class Anything4(SD):
    name = "anything4"
    model_id_or_path = "Sanster/anything-4.0-inpainting"


class RealisticVision14(SD):
    name = "realisticVision1.4"
    model_id_or_path = "Sanster/Realistic_Vision_V1.4-inpainting"


class SD2(SD):
    name = "sd2"
    model_id_or_path = "stabilityai/stable-diffusion-2-inpainting"
