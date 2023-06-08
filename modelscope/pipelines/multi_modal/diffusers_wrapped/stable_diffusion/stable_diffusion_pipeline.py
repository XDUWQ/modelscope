# Copyright © Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Union

import cv2
import os
import numpy as np
import torch
from diffusers import StableDiffusionPipeline as DiffuserStableDiffusionPipeline
from PIL import Image

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.builder import PIPELINES
from modelscope.models.base import Model
from modelscope.pipelines.multi_modal.diffusers_wrapped.diffusers_pipeline import \
    DiffusersPipeline
from modelscope.utils.constant import Tasks
from modelscope.tuners.lora import LoRATuner
                                    

# Wrap around the diffusers stable diffusion pipeline implementation
# for a unified ModelScope pipeline experience. Native stable diffusion
# pipelines will be implemented in later releases.
@PIPELINES.register_module(
    Tasks.text_to_image_synthesis,
    module_name=Pipelines.diffusers_stable_diffusion)
class StableDiffusionPipeline(DiffusersPipeline):

    def __init__(self, model: Union[Model, str], device: str = 'gpu', 
                 lora_model: str = None, **kwargs):
        """
        use `model` to create a stable diffusion pipeline
        Args:
            model: model id on modelscope hub.
            lora_model: lora model. 
            device: str = 'gpu'
        """
        super().__init__(model, device, **kwargs)
        if isinstance(model, str):
            torch_dtype = kwargs.get('torch_dtype', torch.float32)
            # build upon the diffuser stable diffusion pipeline
            self.pipeline = DiffuserStableDiffusionPipeline.from_pretrained(
                model, torch_dtype=torch_dtype)
            self.pipeline.to(self.device)
        if lora_model is not None:
            # model = Model.from_pretrained(model)
            LoRATuner.tune(self.model, replace_modules=['to_q', 'to_k', 'to_v'])
            if not os.path.isdir(lora_model):
                raise ValueError('lora_model path is not exits')
            self.model.load_state_dict(torch.load(os.path.join(lora_model, 'pytorch_model.bin')))
            self.model.eval()

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        if not isinstance(inputs, dict):
            raise ValueError(
                f'Expected the input to be a dictionary, but got {type(input)}'
            )
        if 'text' not in inputs:
            raise ValueError('input should contain "text", but not found')
        if isinstance(self.model, str):
            return self.pipeline(
                prompt=inputs.get('text'),
                height=inputs.get('height'),
                width=inputs.get('width'),
                num_inference_steps=inputs.get('num_inference_steps', 50),
                guidance_scale=inputs.get('guidance_scale', 7.5),
                negative_prompt=inputs.get('negative_prompt'),
                num_images_per_prompt=inputs.get('num_images_per_prompt', 1),
                eta=inputs.get('eta', 0.0),
                generator=inputs.get('generator'),
                latents=inputs.get('latents'),
                output_type=inputs.get('output_type', 'pil'),
                return_dict=inputs.get('return_dict', True),
                callback=inputs.get('callback'),
                callback_steps=inputs.get('callback_steps', 1))
        else:
            with torch.no_grad():
                return super().forward(inputs, **forward_params)

    def postprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        images = []
        for img in inputs.images:
            if isinstance(img, Image.Image):
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                images.append(img)
        return {OutputKeys.OUTPUT_IMGS: images}
