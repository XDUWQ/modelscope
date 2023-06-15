# Copyright © Alibaba, Inc. and its affiliates.

import os
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
from diffusers import \
    StableDiffusionPipeline as DiffuserStableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers import (StableDiffusionPipeline, AutoencoderKL, 
                       DDPMScheduler, UNet2DConditionModel)
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from PIL import Image

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.multi_modal.diffusers_wrapped.diffusers_pipeline import \
    DiffusersPipeline
from modelscope.utils.constant import Tasks


@PIPELINES.register_module(
    Tasks.text_to_image_synthesis,
    module_name=Pipelines.diffusers_stable_diffusion)
class StableDiffusionPipeline(DiffusersPipeline):

    def __init__(self, model: str, lora_dir: str = None, 
                 unet_dir: str = None, text_encoder_dir: str = None, **kwargs):
        """
        use `model` to create a stable diffusion pipeline
        Args:
            model: model id on modelscope hub or local model dir.
            lora_dir: lora directory for lora tune.
            unet_dir: unet directory for fine-tuned model.
            text_encoder_dr: text encoder for fine-tuned model.
        """

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
        # build complete the diffuser stable diffusion pipeline
        if unet_dir is None and text_encoder_dir is None:
            self.pipeline = DiffuserStableDiffusionPipeline.from_pretrained(
                model, torch_dtype=torch.float16)        
        # build respectively diffuser stable diffusion pipeline
        else:
            if unet_dir is not None:
                unet = UNet2DConditionModel.from_pretrained(unet_dir)
            else:
                unet = UNet2DConditionModel.from_pretrained(model, subfolder='unet')
            if text_encoder_dir is not None:
                text_encoder = CLIPTextModel.from_pretrained(text_encoder_dir)
            else:
                text_encoder = CLIPTextModel.from_pretrained(model, subfolder="text_encoder")
            
            vae = AutoencoderKL.from_pretrained(model, subfolder='vae')
            tokenizer = CLIPTokenizer.from_pretrained(model, subfolder='tokenizer')
            scheduler = DDPMScheduler.from_pretrained(model, subfolder='scheduler')
            self.pipeline = StableDiffusionPipeline(
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                tokenizer=tokenizer,
                scheduler=scheduler,
                safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
                feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
            )

        # load lora moudle to unet
        if lora_dir is not None:
            assert os.path.exists(lora_dir), f"{lora_dir} isn't exist"
            self.pipeline.unet.load_attn_procs(lora_dir)
        
        self.pipeline = self.pipeline.to(self.device)

    def preprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return inputs

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        if not isinstance(inputs, dict):
            raise ValueError(
                f'Expected the input to be a dictionary, but got {type(input)}'
            )
        if 'prompt' not in inputs:
            raise ValueError('input should contain "prompt", but not found')

        images = self.pipeline(
            inputs['prompt'], num_inference_steps=30, guidance_scale=7.5)

        return images

    def postprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        images = []
        for img in inputs.images:
            if isinstance(img, Image.Image):
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                images.append(img)
        return {OutputKeys.OUTPUT_IMGS: images}
