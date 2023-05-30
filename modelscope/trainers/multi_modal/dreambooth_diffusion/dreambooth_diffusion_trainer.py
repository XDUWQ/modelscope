# Copyright 2022-2023 The Alibaba Fundamental Vision Team Authors. All rights reserved.
from typing import Union
from collections.abc import Mapping

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributed as dist

from diffusers import (AutoencoderKL, DDPMScheduler,
                       UNet2DConditionModel)
from transformers import AutoTokenizer, PretrainedConfig
from modelscope.metainfo import Trainers
from modelscope.outputs import OutputKeys
from modelscope.models.base import TorchModel
from modelscope.trainers.builder import TRAINERS
from modelscope.utils.data_utils import to_device
from modelscope.utils.torch_utils import is_dist
from modelscope.trainers.trainer import EpochBasedTrainer
from modelscope.utils.constant import ModeKeys, TrainerStages

class UnetModel(TorchModel):
    def __init__(self, unet, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = unet
    
    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)


@TRAINERS.register_module(module_name=Trainers.dreambooth_diffusion)
class DreamboothDiffusionTrainer(EpochBasedTrainer):

    def __init__(self, *args, **kwargs):
        self.prior_loss_weight = kwargs.pop('prior_loss_weight', 1.0)
        self.class_prompt = kwargs.pop('class_prompt', "a photo of dog")
        self.with_prior_preservation = kwargs.pop('with_prior_preservation', False)
        self.instance_prompt = kwargs.pop('instance_prompt', "a photo of sks dog")
        self.pretrained_model_name_or_path = kwargs.pop('pretrained_model_name_or_path', 
                                                        "runwayml/stable-diffusion-v1-5")
        super().__init__(*args, **kwargs)

    def build_model(self) -> Union[nn.Module, TorchModel]:
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.pretrained_model_name_or_path, subfolder='scheduler')
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder='tokenizer',
            revision=None,
            use_fast=False)
        self.vae = AutoencoderKL.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder='vae',
            revision=None)
        self.unet = UNet2DConditionModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder='unet',
            revision=None)
        # import correct text encoder class
        text_encoder_cls = self.import_model_class_from_model_name_or_path(self.pretrained_model_name_or_path)
        self.text_encoder = text_encoder_cls.from_pretrained(
            self.pretrained_model_name_or_path, 
            subfolder="text_encoder", 
            revision=None)
        if self.vae is not None:
            self.vae.requires_grad_(False)
        if self.text_encoder is not None:
            self.text_encoder.requires_grad_(False)
        
        return UnetModel(self.unet)

    def train(self,
              checkpoint_path=None,
              load_all_state=True,
              *args,
              **kwargs):
        """Start training.
        Args:
            checkpoint_path(`str`, `optional`): The previous saving checkpoint to read,
                usually it's a `some-file-name.pth` file generated by this trainer.
            load_all_state(`bool`: `optional`): Load all state out of the `checkpoint_path` file, including the
                state dict of model, optimizer, lr_scheduler, the random state and epoch/iter number. If False, only
                the model's state dict will be read, and model will be trained again.
            kwargs:
                strict(`boolean`): If strict, any unmatched keys will cause an error.
        """

        self._mode = ModeKeys.TRAIN
        self.train_dataloader = self.get_train_dataloader()
        self.data_loader = self.train_dataloader
        self.register_optimizers_hook()
        self.register_processors()
        self.print_hook_info()
        self.set_checkpoint_file_to_hook(checkpoint_path, load_all_state,
                                         kwargs.get('strict', False))
        self.model.train()
        self.model = self.model.to(self.device)
        self.vae = self.vae.to(self.device)
        self.text_encoder = self.text_encoder.to(self.device)

        self.train_loop(self.train_dataloader)

    def train_step(self, model, inputs):
        """ Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`TorchModel`): The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        self._mode = ModeKeys.TRAIN
        # inputs
        batch = self.data_assemble(inputs)
        pixel_values = batch["pixel_values"].to(dtype=torch.float32)
        if self.vae is not None:
            # Convert images to latent space
            model_input = self.vae.encode(batch["pixel_values"]).latent_dist.sample()
            model_input = model_input * self.vae.config.scaling_factor
        else:
            model_input = pixel_values
        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device)
        timesteps = timesteps.long()
        # Add noise to the model input according to the noise magnitude at each timestep
        noisy_model_input = self.noise_scheduler.add_noise(model_input, noise, timesteps)
        # Get the text embedding for conditioning
        encoder_hidden_states = self.encode_prompt(
            self.text_encoder,
            batch["input_ids"],
            batch["attention_mask"],
            text_encoder_use_attention_mask=False,
        )
        # Predict the noise residual by unet model
        model_pred = model(noisy_model_input, timesteps, encoder_hidden_states).sample
        if model_pred.shape[1] == 6:
            model_pred, _ = torch.chunk(model_pred, 2, dim=1)
        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(model_input, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
        if self.with_prior_preservation:
            # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
            target, target_prior = torch.chunk(target, 2, dim=0)
            # Compute instance loss
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            # Compute prior loss
            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
            # Add the prior loss to the instance loss.
            loss = loss + self.prior_loss_weight * prior_loss
        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        train_outputs = {OutputKeys.LOSS: loss}
        print("loss: ", loss)

        # add model output info to log
        if 'log_vars' not in train_outputs:
            default_keys_pattern = ['loss']
            match_keys = set([])
            for key_p in default_keys_pattern:
                match_keys.update(
                    [key for key in train_outputs.keys() if key_p in key])

            log_vars = {}
            for key in match_keys:
                value = train_outputs.get(key, None)
                if value is not None:
                    if is_dist():
                        value = value.data.clone().to('cuda')
                        dist.all_reduce(value.div_(dist.get_world_size()))
                    log_vars.update({key: value.item()})
            self.log_buffer.update(log_vars)
        else:
            self.log_buffer.update(train_outputs['log_vars'])

        self.train_outputs = train_outputs

    def evaluation_step(self, data):
        self.model.eval()

        with torch.no_grad():
            batch = self.data_assemble(data)
            model_input = self.vae.encode(batch["pixel_values"]).latent_dist.sample()
            model_input = model_input * self.vae.config.scaling_factor
            noise = torch.randn_like(model_input)
            bsz = model_input.shape[0]
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device)
            timesteps = timesteps.long()
            noisy_model_input = self.noise_scheduler.add_noise(model_input, noise, timesteps)
            encoder_hidden_states = self.encode_prompt(
                self.text_encoder,
                batch["input_ids"],
                batch["attention_mask"],
                text_encoder_use_attention_mask=False,
            )
            # Predict reslut by unet model
            model_pred = self.model(noisy_model_input, timesteps, encoder_hidden_states).sample
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(model_input, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            result = {OutputKeys.LOSS: loss}
    
        return result

    def import_model_class_from_model_name_or_path(self, pretrained_model_name_or_path: str):
        text_encoder_config = PretrainedConfig.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=None
        )
        model_class = text_encoder_config.architectures[0]

        if model_class == "CLIPTextModel":
            from transformers import CLIPTextModel

            return CLIPTextModel
        elif model_class == "RobertaSeriesModelWithTransformation":
            from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

            return RobertaSeriesModelWithTransformation
        elif model_class == "T5EncoderModel":
            from transformers import T5EncoderModel

            return T5EncoderModel
        else:
            raise ValueError(f"{model_class} is not supported.")

    def encode_prompt(self, text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
        text_input_ids = input_ids.to(text_encoder.device)

        if text_encoder_use_attention_mask:
            attention_mask = attention_mask.to(text_encoder.device)
        else:
            attention_mask = None

        prompt_embeds = text_encoder(
            text_input_ids,
            attention_mask=attention_mask,
        )
        prompt_embeds = prompt_embeds[0]

        return prompt_embeds
    
    def data_assemble(self, inputs, with_prior_preservation=False):
        batch = {}
        batch["pixel_values"] = inputs["target"]
        text_inputs = self.tokenize_prompt(self.tokenizer, self.instance_prompt)
        batch["input_ids"] = text_inputs.input_ids
        batch["attention_mask"] = text_inputs.attention_mask

        return batch

    def tokenize_prompt(self, tokenizer, prompt, tokenizer_max_length=None):
        if tokenizer_max_length is not None:
            max_length = tokenizer_max_length
        else:
            max_length = tokenizer.model_max_length

        text_inputs = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        return text_inputs