#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import copy
import itertools
import logging
import math
import os
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path

from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs

import diffusers
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
    FluxFillPipeline
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm

from material_coating_dataset import (
    get_train_dataset,
    prepare_train_dataset,
    create_train_dataloader
)
from material_trait_embeddings import create_material_trait_embeddings, save_material_trait_embeddings, \
    load_material_trait_embeddings

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.32.0.dev0")

logger = get_logger(__name__)


def get_dtype(mixed_precision):
    weight_dtype = torch.float32
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    return weight_dtype


def split_dataset_for_validation(dataset, validation_count, seed):
    """
    Split the dataset into train and validation sets.
    Args:
        dataset: The full dataset
        validation_count: Number of samples to use for validation
        seed: Random seed for reproducible splits

    Returns:
        train_dataset, validation_samples
    """
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()

    validation_count = min(validation_count, len(dataset))
    validation_indices = indices[:validation_count]
    train_indices = indices[validation_count:]

    validation_samples = []
    for idx in validation_indices:
        sample = dataset[idx]
        validation_samples.append(sample)

    if hasattr(dataset, 'select'):
        train_dataset = dataset.select(train_indices)
    else:
        train_dataset = torch.utils.data.Subset(dataset, train_indices)

    logger.info(f"Split dataset: {len(train_dataset)} training samples, {len(validation_samples)} validation samples")

    return train_dataset, validation_samples


def prepare_dataset(args, accelerator):
    full_dataset = get_train_dataset(args, accelerator)
    full_dataset = prepare_train_dataset(full_dataset, accelerator, args)

    train_dataset, validation_samples = split_dataset_for_validation(
        full_dataset,
        args.num_validation_samples,
        args.validation_seed
    )

    train_dataloader = create_train_dataloader(train_dataset, args)
    return train_dataloader, train_dataset, validation_samples


def calculate_validation_loss(validation_sample, transformer, vae, noise_scheduler_copy, args, accelerator,
                              weight_dtype, material_traits_embeddings):
    """
    Calculate validation loss for a single sample using the same logic as training.
    """
    with torch.no_grad():
        # Get data from validation sample
        pixel_values = validation_sample['pixel_values'].unsqueeze(0).to(accelerator.device)  # Add batch dim
        source_images = validation_sample['source_images'].unsqueeze(0).to(accelerator.device)
        masks = validation_sample['masks'].unsqueeze(0).to(accelerator.device)

        height = validation_sample['pixel_values'].shape[1]
        width = validation_sample['pixel_values'].shape[2]

        # Encode images to latents (same as training)
        model_input = vae.encode(pixel_values.to(vae.dtype)).latent_dist.sample()
        model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
        model_input = model_input.to(weight_dtype)

        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

        # Sample random timestep
        bsz = 1
        noise = torch.randn_like(model_input, device=accelerator.device, dtype=weight_dtype)
        u = compute_density_for_timestep_sampling(
            weighting_scheme=args.weighting_scheme,
            batch_size=bsz,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            mode_scale=args.mode_scale,
        )
        indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

        # Add noise according to flow matching (same as training)
        def get_sigmas_single(timesteps, n_dim=4, dtype=torch.float32):
            sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
            schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
            timesteps = timesteps.to(accelerator.device)
            step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
            sigma = sigmas[step_indices].flatten()
            while len(sigma.shape) < n_dim:
                sigma = sigma.unsqueeze(-1)
            return sigma

        sigmas = get_sigmas_single(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

        # Pack latents (same as training)
        packed_noisy_model_input = FluxFillPipeline._pack_latents(
            noisy_model_input,
            batch_size=model_input.shape[0],
            num_channels_latents=model_input.shape[1],
            height=model_input.shape[2],
            width=model_input.shape[3],
        )

        # Process source images (same as training)
        # source_images is already triptych with albedo in middle panel
        source_image_latents = vae.encode(
            source_images.reshape(pixel_values.shape).to(dtype=weight_dtype)
        ).latent_dist.sample()

        mask, source_image_latents = prepare_mask_latents(
            mask=masks,
            source_image_latents=source_image_latents,
            batch_size=bsz,
            num_channels_latents=model_input.shape[1],
            num_images_per_prompt=1,
            height=height,
            width=width,
            dtype=weight_dtype,
            device=accelerator.device,
            vae_scale_factor=vae_scale_factor,
            vae_shift_factor=vae.config.shift_factor,
            vae_scaling_factor=vae.config.scaling_factor
        )
        source_image_latents = torch.cat((source_image_latents, mask), dim=-1)

        # Create transformer input by concatenating noisy input with source image latents (same as training)
        transformer_input = torch.cat((packed_noisy_model_input, source_image_latents), dim=2)

        # Prepare latent image ids (same as training)
        latent_image_ids = FluxFillPipeline._prepare_latent_image_ids(
            model_input.shape[0],
            model_input.shape[2] // 2,
            model_input.shape[3] // 2,
            accelerator.device,
            weight_dtype,
        )

        # Handle guidance (same as training)
        unwrapped_transformer = accelerator.unwrap_model(transformer)
        unwrapped_transformer = unwrapped_transformer._orig_mod if is_compiled_module(
            unwrapped_transformer) else unwrapped_transformer

        if unwrapped_transformer.config.guidance_embeds:
            guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
            guidance = guidance.expand(model_input.shape[0])
        else:
            guidance = None

        # Extract material traits from validation sample (same as training)
        validation_material_traits = {
            "thickness": validation_sample["thickness"].unsqueeze(0),
            "metallic": validation_sample["metallic"].unsqueeze(0),
            "roughness": validation_sample["roughness"].unsqueeze(0),
            "transmission_weight": validation_sample["transmission_weight"].unsqueeze(0),
            "apply_texture_task": validation_sample["apply_texture_task"].unsqueeze(0),
            "replace_task": validation_sample["replace_task"].unsqueeze(0),
            "remove_task": validation_sample["remove_task"].unsqueeze(0),
            "uv_mapping_spherical": validation_sample["uv_mapping_spherical"].unsqueeze(0),
            "uv_mapping_cubic": validation_sample["uv_mapping_cubic"].unsqueeze(0),
            "uv_mapping_original": validation_sample["uv_mapping_original"].unsqueeze(0),
        }
        prompt_embeds, pooled_prompt_embeds, text_ids = \
            accelerator.unwrap_model(material_traits_embeddings).make_prompt_embeddings(
                batch_size=bsz,
                material_traits_dict=validation_material_traits,
                device=accelerator.device
            )

        # Forward pass (same as training)
        model_pred = transformer(
            hidden_states=transformer_input,
            timestep=timesteps / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]

        # Unpack latents (same as training)
        model_pred = FluxFillPipeline._unpack_latents(
            model_pred,
            height=model_input.shape[2] * vae_scale_factor,
            width=model_input.shape[3] * vae_scale_factor,
            vae_scale_factor=vae_scale_factor,
        )

        # Calculate loss (same as training)
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
        target = noise - model_input
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss = loss.mean()
        return loss.item()


def save_model_card(
        repo_id: str,
        images=None,
        base_model: str = None,
        instance_prompt=None,
        validation_prompt=None,
        repo_folder=None,
):
    widget_dict = []
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            widget_dict.append(
                {"text": validation_prompt if validation_prompt else " ", "output": {"url": f"image_{i}.png"}}
            )

    model_description = f"""
# Flux-Fill DreamBooth LoRA - {repo_id}

<Gallery />

## Model description

These are {repo_id} DreamBooth LoRA weights for {base_model}.

The weights were trained using [DreamBooth](https://dreambooth.github.io/) with a custom [Flux diffusers trainer](https://github.com/Sebastian-Zok/FLUX-Fill-LoRa-Training).

## Trigger words

You should use `{instance_prompt}` to trigger the image generation.

## Download model

[Download the *.safetensors LoRA]({repo_id}/tree/main) in the Files & versions tab.

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
from diffusers import AutoPipelineForText2Image
import torch
pipeline = AutoPipelineForText2Image.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to('cuda')
pipeline.load_lora_weights('{repo_id}', weight_name='pytorch_lora_weights.safetensors')
image = pipeline('{validation_prompt if validation_prompt else instance_prompt}').images[0]
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)

## License

Please adhere to the licensing terms as described [here](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        prompt=instance_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-image",
        "diffusers-training",
        "diffusers",
        "lora",
        "flux",
        "flux-diffusers",
        "template:sd-lora",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(
        args,
        accelerator,
        validation_samples,
        material_traits_embeddings,
        vae,
        vae_scale_factor,
        transformer,
        noise_scheduler_copy,
        torch_dtype,
        is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images"
    )

    weight_dtype = get_dtype(args.mixed_precision)

    if not is_final_validation:
        pipeline = FluxFillPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            transformer=accelerator.unwrap_model(transformer),
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
    else:
        transformer = FluxTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=weight_dtype
        )
        pipeline = FluxFillPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            transformer=transformer,
            torch_dtype=weight_dtype,
        )
        pipeline.load_lora_weights(args.output_dir)

    pipeline = pipeline.to(accelerator.device, dtype=torch_dtype)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
    autocast_ctx = torch.autocast(accelerator.device.type) if not is_final_validation else nullcontext()

    image_logs = []
    validation_losses = []

    for validation_sample in validation_samples:
        # Calculate validation loss for this sample
        loss = calculate_validation_loss(
            validation_sample, transformer, vae, noise_scheduler_copy, args, accelerator,
            weight_dtype, material_traits_embeddings
        )
        validation_losses.append(loss)

        validation_image = validation_sample["source_images"]  # Masked triptych
        validation_coating_mask = validation_sample["masks"]  # Triptych mask

        bsz = 1
        validation_material_traits = {
            "thickness": validation_sample["thickness"].unsqueeze(0),
            "metallic": validation_sample["metallic"].unsqueeze(0),
            "roughness": validation_sample["roughness"].unsqueeze(0),
            "transmission_weight": validation_sample["transmission_weight"].unsqueeze(0),
            "apply_texture_task": validation_sample["apply_texture_task"].unsqueeze(0),
            "replace_task": validation_sample["replace_task"].unsqueeze(0),
            "remove_task": validation_sample["remove_task"].unsqueeze(0),
            "uv_mapping_spherical": validation_sample["uv_mapping_spherical"].unsqueeze(0),
            "uv_mapping_cubic": validation_sample["uv_mapping_cubic"].unsqueeze(0),
            "uv_mapping_original": validation_sample["uv_mapping_original"].unsqueeze(0),
        }
        prompt_embeds, pooled_prompt_embeds, text_ids = \
            accelerator.unwrap_model(material_traits_embeddings).make_prompt_embeddings(batch_size=bsz,
                                                                                        material_traits_dict=validation_material_traits,
                                                                                        device=accelerator.device)

        images = []
        for _ in range(args.num_validation_images):
            with autocast_ctx:
                image = pipeline(
                    image=validation_image.unsqueeze(0),
                    mask_image=validation_coating_mask.unsqueeze(0),
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    generator=generator,
                    width=args.resolution * 3,
                    height=args.resolution,
                ).images[0]
                images.append(image)

        # Extract material traits for logging
        material_traits_log = {}
        for trait_name in accelerator.unwrap_model(material_traits_embeddings).TRAIT_NAMES:
            material_traits_log[trait_name] = validation_sample[trait_name].item()

        image_logs.append(
            {"validation_image": validation_image,
             "validation_coating_mask": validation_coating_mask,
             "images": images,
             "material_traits": material_traits_log,
             "original_dataset_path": validation_sample['original_dataset_path']}
        )

    # Calculate and log average validation loss
    avg_validation_loss = sum(validation_losses) / len(validation_losses) if validation_losses else 0.0
    logger.info(f"Average validation loss: {avg_validation_loss:.6f}")

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            formatted_images = []
            for log in image_logs:
                images = log["images"]
                validation_image = log["validation_image"]
                validation_coating_mask = log["validation_coating_mask"]
                formatted_images.append(wandb.Image(validation_image, caption="Triptych Input"))
                formatted_images.append(wandb.Image(validation_coating_mask, caption="Triptych Mask"))
                traits_str = ", ".join([f"{k}={v:.3f}" for k, v in log["material_traits"].items()])
                dataset_idx = log["original_dataset_path"]
                caption_with_idx = f"{dataset_idx}: {traits_str}"

                for image in images:
                    image = wandb.Image(image, caption=caption_with_idx)
                    formatted_images.append(image)

            tracker.log({tracker_key: formatted_images})
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    # Log validation loss
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            tracker.log({"validation_loss": avg_validation_loss})
        else:
            logger.warning(f"validation loss logging not implemented for {tracker.name}")

    del pipeline
    free_memory()

    return image_logs, avg_validation_loss


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
             "default, the standard Image Dataset maps out 'file_name' "
             "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")

    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=False,
        help="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        help="A file path to a local or remote (https) image file that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--validation_mask",
        type=str,
        default=None,
        help="A file path to a local or remote (https) image file that is used as mask for the inpainting task during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-dreambooth-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="the FLUX.1 dev variant is a guidance distilled model",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
             "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")

    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help=(
            'The transformer modules to apply LoRA training on. Please specify the layers in a comma seperated. E.g. - "to_k,to_q,to_v,to_out.0" will result in lora training of attention layers only'
        ),
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
             "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        default=False,
        help="Cache the VAE latents",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--upcast_before_saving",
        action="store_true",
        default=False,
        help=(
            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
            "Defaults to precision dtype used for training to save memory"
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--jsonl_for_train",
        type=str,
        default=None,
        help="Path to the jsonl file containing the training data.",
    )
    parser.add_argument(
        "--enable_data_augmentation",
        action="store_true",
        help="Enable data augmentation (horizontal flip, rotation, color jitter)",
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        help="Use center crop instead of resize when processing images",
    )
    parser.add_argument(
        "--normal_drop",
        type=float,
        default=0.5,
        help="Probability for dropping normal image",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--num_validation_samples",
        type=int,
        default=10,
        help="Number of samples from the training dataset to use for validation",
    )
    parser.add_argument(
        "--validation_seed",
        type=int,
        default=21,
        help="Seed for validation dataset split. If None, uses the main training seed.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1000,
        help=(
            "Run validation every X steps. Validation consists of running inference on a subset of the training data."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="dreambooth-flux-kontext-lora",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--transmissive_filter",
        type=str,
        default=None,
        choices=["transmissive_only", "non_transmissive_only"],
        help="Filter dataset to only transmissive or non-transmissive samples based on transmission_weight",
    )
    parser.add_argument(
        "--use_best_uv_mapping",
        action="store_true",
        help="Filter dataset to only use items where uv_mapping matches best_mapping_method.",
    )
    parser.add_argument(
        "--filter_out_planar_mapping",
        action="store_true",
        help="Filter dataset to not use items where uv_mapping == PLANAR",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.instance_data_dir is None and args.jsonl_for_train is None:
        raise ValueError("Specify either `--dataset_name`, `--instance_data_dir`, or `--jsonl_for_train`")

    if sum([args.dataset_name is not None, args.instance_data_dir is not None, args.jsonl_for_train is not None]) > 1:
        raise ValueError("Specify only one of `--dataset_name`, `--instance_data_dir`, or `--jsonl_for_train`")

    # if args.mask_data_dir is None:
    #     raise ValueError("Specify a --mask_data_dir`")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    return args


def prepare_mask_latents(
        mask,
        source_image_latents,
        batch_size,
        num_channels_latents,
        num_images_per_prompt,
        height,
        width,
        dtype,
        device,
        vae_scale_factor,
        vae_shift_factor,
        vae_scaling_factor
):
    """ Prepare mask latents """
    # 1. calculate the height and width of the latents
    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    source_image_latents = (source_image_latents - vae_shift_factor) * vae_scaling_factor
    source_image_latents = source_image_latents.to(device=device, dtype=dtype)

    # 2. duplicate mask and source_image_latents for each generation per prompt, using mps friendly method
    batch_size = batch_size * num_images_per_prompt
    if mask.shape[0] < batch_size:
        if not batch_size % mask.shape[0] == 0:
            raise ValueError(
                "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                " of masks that you pass is divisible by the total requested batch size."
            )
        mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
    if source_image_latents.shape[0] < batch_size:
        if not batch_size % source_image_latents.shape[0] == 0:
            raise ValueError(
                "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                f" to a total batch size of {batch_size}, but {source_image_latents.shape[0]} images were passed."
                " Make sure the number of images that you pass is divisible by the total requested batch size."
            )
        source_image_latents = source_image_latents.repeat(batch_size // source_image_latents.shape[0], 1, 1, 1)

    # 3. pack the source_image_latents
    # batch_size, num_channels_latents, height, width -> batch_size, height//2 * width//2 , num_channels_latents*4

    source_image_latents = FluxFillPipeline._pack_latents(
        source_image_latents,
        batch_size,
        num_channels_latents,
        height,
        width,
    )

    # 4. resize mask to latents shape we concatenate the mask to the latents
    mask = mask[:, 0, :, :]  # batch_size, 8 * height, 8 * width (mask has not been 8x compressed)
    mask = mask.view(
        batch_size, height, vae_scale_factor, width, vae_scale_factor
    )  # batch_size, height, 8, width, 8
    mask = mask.permute(0, 2, 4, 1, 3)  # batch_size, 8, 8, height, width
    mask = mask.reshape(
        batch_size, vae_scale_factor * vae_scale_factor, height, width
    )  # batch_size, 8*8, height, width

    # 5. pack the mask:
    # batch_size, 64, height, width -> batch_size, height//2 * width//2 , 64*2*2
    mask = FluxFillPipeline._pack_latents(
        mask,
        batch_size,
        vae_scale_factor * vae_scale_factor,
        height,
        width,
    )
    mask = mask.to(device=device, dtype=dtype)

    return mask, source_image_latents


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200)) # 2 hours

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs, process_group_kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )

    # We only train the additional adapter LoRA layers
    transformer.requires_grad_(False)
    vae.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)

    material_traits_embeddings = create_material_trait_embeddings()

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]

    # now we will add new LoRA weights the transformer layers
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            material_traits_embeddings_to_save = None
            modules_to_save = {}

            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    modules_to_save["transformer"] = model
                elif isinstance(model, type(unwrap_model(material_traits_embeddings))):
                    material_traits_embeddings_to_save = unwrap_model(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            FluxFillPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
            )

            save_material_trait_embeddings(material_traits_embeddings_to_save,
                                           os.path.join(output_dir, "material_trait_embeddings.pt"))

    def load_model_hook(models, input_dir):
        transformer_ = None
        material_traits_embeddings_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            elif isinstance(model, type(unwrap_model(material_traits_embeddings))):
                material_traits_embeddings_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict = FluxFillPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        # Load material traits embeddings and conditioning projector for material coating training
        material_trait_embeddings_path = os.path.join(input_dir, "material_trait_embeddings.pt")
        if os.path.exists(material_trait_embeddings_path):
            loaded_material_trait_embeddings = load_material_trait_embeddings(
                material_trait_embeddings_path, accelerator.device)

            unwrapped_material_traits = unwrap_model(material_traits_embeddings_)
            unwrapped_material_traits.load_state_dict(loaded_material_trait_embeddings.state_dict())

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_, material_traits_embeddings_]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer, material_traits_embeddings]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    total_trainable_parameters = itertools.chain(material_traits_embeddings.parameters(), transformer_lora_parameters)

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            total_trainable_parameters,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            total_trainable_parameters,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    train_dataloader, train_dataset, validation_samples = prepare_dataset(args, accelerator)

    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    if args.cache_latents:
        latents_cache = []
        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                batch["pixel_values"] = batch["pixel_values"].to(
                    accelerator.device, non_blocking=True, dtype=weight_dtype
                )
                latents_cache.append(vae.encode(batch["pixel_values"]).latent_dist)

        if args.validation_prompt is None:
            del vae
            free_memory()

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    transformer, material_traits_embeddings, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, material_traits_embeddings, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)
    mask_processor = VaeImageProcessor(
        vae_scale_factor=vae_scale_factor * 2,
        vae_latent_channels=vae.config.latent_channels,
        do_normalize=False,
        do_binarize=True,
        do_convert_grayscale=True,
    )
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        material_traits_embeddings.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer, material_traits_embeddings]

            with accelerator.accumulate(models_to_accumulate):
                # 1. Base Ground Truth Latent (x_0)
                if args.cache_latents:
                    model_input = latents_cache[step].sample()
                else:
                    pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                    model_input = vae.encode(pixel_values).latent_dist.sample()

                model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

                bsz = model_input.shape[0]
                h, w_trip = pixel_values.shape[2], pixel_values.shape[3]
                w = w_trip // 3

                coat_mask = batch["masks"]

                # 2. Dynamically construct Masks for Track 1 (Compose) and Track 2 (Decompose)

                # First, extract the true shape mask regardless of where the dataset put it. It can be on the left or the right. It's a XOR.
                true_shape_mask = coat_mask[:, :, :, 0:w] + coat_mask[:, :, :, w * 2:w * 3]
                # Clamp to 1.0 just in case of any floating point anomalies
                true_shape_mask = torch.clamp(true_shape_mask, 0.0, 1.0)

                # Compose Mask (M_c): Predict the 3rd panel (Coated Result) [cite: 1210, 1211]
                mask_c_pixel = torch.zeros((bsz, 1, h, w_trip), device=accelerator.device, dtype=weight_dtype)
                # Explicitly paste the true shape into the 3rd panel
                mask_c_pixel[:, :, :, w * 2:w * 3] = true_shape_mask

                # Decompose Mask (M_d): Predict 1st and 2nd panels (Uncoated + Albedo) [cite: 1210, 1211]
                mask_d_pixel = torch.zeros((bsz, 1, h, w_trip), device=accelerator.device, dtype=weight_dtype)
                # Copy mask to left (generated clean) panel
                mask_d_pixel[:, :, :, 0:w] = true_shape_mask
                # Full white for middle panel (generate albedo)
                mask_d_pixel[:, :, :, w:w * 2] = 1.0

                # 3. Create Masked Conditional Images in Pixel Space
                masked_pixel_c = pixel_values * (1.0 - mask_c_pixel) + 0.5 * mask_c_pixel
                masked_pixel_d = pixel_values * (1.0 - mask_d_pixel) + 0.5 * mask_d_pixel

                # Encode Conditional Images
                with torch.no_grad():
                    source_latents_c_raw = vae.encode(masked_pixel_c).latent_dist.sample()
                    source_latents_d_raw = vae.encode(masked_pixel_d).latent_dist.sample()

                # Process conditional latents using existing helper
                packed_mask_c, cond_latents_c = prepare_mask_latents(
                    mask=mask_processor.preprocess(mask_c_pixel, height=h, width=w_trip),
                    source_image_latents=source_latents_c_raw,
                    batch_size=bsz, num_channels_latents=model_input.shape[1],
                    num_images_per_prompt=1, height=h, width=w_trip,
                    dtype=weight_dtype, device=accelerator.device,
                    vae_scale_factor=vae_scale_factor, vae_shift_factor=vae_config_shift_factor,
                    vae_scaling_factor=vae_config_scaling_factor
                )
                transformer_cond_c = torch.cat((cond_latents_c, packed_mask_c), dim=-1)

                packed_mask_d, cond_latents_d = prepare_mask_latents(
                    mask=mask_processor.preprocess(mask_d_pixel, height=h, width=w_trip),
                    source_image_latents=source_latents_d_raw,
                    batch_size=bsz, num_channels_latents=model_input.shape[1],
                    num_images_per_prompt=1, height=h, width=w_trip,
                    dtype=weight_dtype, device=accelerator.device,
                    vae_scale_factor=vae_scale_factor, vae_shift_factor=vae_config_shift_factor,
                    vae_scaling_factor=vae_config_scaling_factor
                )
                transformer_cond_d = torch.cat((cond_latents_d, packed_mask_d), dim=-1)

                # 4. Construct Traits for Compose (\tau_c) and Decompose (\tau_d)
                # Fallback: if it was a remove task, its forward sum was 0, so we default it to apply_texture
                forward_sum = batch["apply_texture_task"] + batch["replace_task"]
                apply_texture_c = batch["apply_texture_task"] + (forward_sum == 0).float()

                traits_c = {
                    "thickness": batch["thickness"],
                    "metallic": batch["metallic"],
                    "roughness": batch["roughness"],
                    "transmission_weight": batch["transmission_weight"],
                    "apply_texture_task": apply_texture_c,
                    "replace_task": batch["replace_task"],
                    "remove_task": torch.zeros_like(batch["remove_task"]),  # Force Compose
                    "uv_mapping_spherical": batch["uv_mapping_spherical"],
                    "uv_mapping_cubic": batch["uv_mapping_cubic"],
                    "uv_mapping_original": batch["uv_mapping_original"],
                }

                traits_d = {
                    **traits_c,
                    "apply_texture_task": torch.zeros_like(batch["apply_texture_task"]),
                    "replace_task": torch.zeros_like(batch["replace_task"]),
                    "remove_task": torch.ones_like(batch["remove_task"]),  # Force Decompose
                }

                prompt_embeds_c, pooled_embeds_c, text_ids = accelerator.unwrap_model(
                    material_traits_embeddings).make_prompt_embeddings(
                    batch_size=bsz, material_traits_dict=traits_c, device=accelerator.device
                )
                prompt_embeds_d, pooled_embeds_d, _ = accelerator.unwrap_model(
                    material_traits_embeddings).make_prompt_embeddings(
                    batch_size=bsz, material_traits_dict=traits_d, device=accelerator.device
                )

                # 5. Independent Timesteps and Noise for Track 1 and Track 2 [cite: 1224]
                u_1 = compute_density_for_timestep_sampling(weighting_scheme=args.weighting_scheme, batch_size=bsz,
                                                            logit_mean=args.logit_mean, logit_std=args.logit_std,
                                                            mode_scale=args.mode_scale)
                indices_1 = (u_1 * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps_1 = noise_scheduler_copy.timesteps[indices_1].to(device=model_input.device)
                sigmas_1 = get_sigmas(timesteps_1, n_dim=model_input.ndim, dtype=model_input.dtype)
                noise_1 = torch.randn_like(model_input)

                u_2 = compute_density_for_timestep_sampling(weighting_scheme=args.weighting_scheme, batch_size=bsz,
                                                            logit_mean=args.logit_mean, logit_std=args.logit_std,
                                                            mode_scale=args.mode_scale)
                indices_2 = (u_2 * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps_2 = noise_scheduler_copy.timesteps[indices_2].to(device=model_input.device)
                sigmas_2 = get_sigmas(timesteps_2, n_dim=model_input.ndim, dtype=model_input.dtype)
                noise_2 = torch.randn_like(model_input)

                latent_image_ids = FluxFillPipeline._prepare_latent_image_ids(
                    bsz, model_input.shape[2] // 2, model_input.shape[3] // 2, accelerator.device, weight_dtype
                )

                guidance = torch.tensor([args.guidance_scale], device=accelerator.device).expand(
                    bsz) if accelerator.unwrap_model(transformer).config.guidance_embeds else None

                # 6. Helper to execute the forward pass and approximate \hat{x}_0 [cite: 1212, 1223]
                def get_velocity(x_0_base, x_1_noise, sigmas, timesteps, cond_input, p_emb, p_pool):
                    noisy_input = (1.0 - sigmas) * x_0_base + sigmas * x_1_noise
                    packed_noisy = FluxFillPipeline._pack_latents(
                        noisy_input, batch_size=bsz, num_channels_latents=model_input.shape[1],
                        height=model_input.shape[2], width=model_input.shape[3]
                    )
                    trans_input = torch.cat((packed_noisy, cond_input), dim=2)

                    pred = transformer(
                        hidden_states=trans_input, timestep=timesteps / 1000, guidance=guidance,
                        pooled_projections=p_pool, encoder_hidden_states=p_emb, txt_ids=text_ids,
                        img_ids=latent_image_ids, return_dict=False
                    )[0]

                    return FluxFillPipeline._unpack_latents(
                        pred, height=model_input.shape[2] * vae_scale_factor,
                        width=model_input.shape[3] * vae_scale_factor, vae_scale_factor=vae_scale_factor
                    )

                # --- Alternating Cycle Tracks ---
                x_0_real = model_input

                weighting_1 = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme,
                                                             sigmas=sigmas_1).to(dtype=model_input.dtype)
                weighting_2 = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme,
                                                             sigmas=sigmas_2).to(dtype=model_input.dtype)

                if global_step % 2 == 0:
                    # ==========================================
                    # TRACK 1: Compose Focus (Compose -> Decompose)
                    # ==========================================
                    # Pass A: Real Compose
                    p_c = get_velocity(x_0_real, noise_1, sigmas_1, timesteps_1, transformer_cond_c,
                                       prompt_embeds_c, pooled_embeds_c)
                    g_c = noise_1 - x_0_real
                    loss_rec = torch.mean((weighting_1.float() * (p_c.float() - g_c.float()) ** 2).reshape(bsz, -1),
                                          1).mean()

                    # Approximate clean latent from Compose (Do not detach, keep graph connected)
                    x_0_hat_c = ((1.0 - sigmas_1) * x_0_real + sigmas_1 * noise_1) - sigmas_1 * p_c

                    # Pass B: Cycle Decompose
                    # Force the decomposed prediction to match the ground truth noise
                    p_d_tilde = get_velocity(x_0_hat_c, noise_2, sigmas_2, timesteps_2, transformer_cond_d, # TODO: transformer_cond_d is made from the real coated image C, rather than the predicted x_0_hat_c. Should try using the first step prediction as conditioning instead.
                                             prompt_embeds_d, pooled_embeds_d)
                    g_d = noise_2 - x_0_real
                    loss_cyc = torch.mean(
                        (weighting_2.float() * (p_d_tilde.float() - g_d.float()) ** 2).reshape(bsz, -1), 1).mean()

                    loss = loss_rec + loss_cyc
                    accelerator.backward(loss)
                else:
                    # ==========================================
                    # TRACK 2: Decompose Focus (Decompose -> Compose)
                    # ==========================================
                    # Pass A: Real Decompose
                    p_d = get_velocity(x_0_real, noise_2, sigmas_2, timesteps_2, transformer_cond_d,
                                       prompt_embeds_d, pooled_embeds_d)
                    g_d = noise_2 - x_0_real
                    loss_rec = torch.mean((weighting_2.float() * (p_d.float() - g_d.float()) ** 2).reshape(bsz, -1),
                                          1).mean()

                    # Approximate clean latent from Decompose (Do not detach, keep graph connected)
                    x_0_hat_d = ((1.0 - sigmas_2) * x_0_real + sigmas_2 * noise_2) - sigmas_2 * p_d

                    # Pass B: Cycle Compose
                    # Force the composed prediction to match the ground truth noise
                    p_c_tilde = get_velocity(x_0_hat_d, noise_1, sigmas_1, timesteps_1, transformer_cond_c, # TODO: same as in track 1, but for decomposition prediction x_0_hat_d.
                                             prompt_embeds_c, pooled_embeds_c)
                    g_c = noise_1 - x_0_real
                    loss_cyc = torch.mean(
                        (weighting_1.float() * (p_c_tilde.float() - g_c.float()) ** 2).reshape(bsz, -1), 1).mean()

                    loss = loss_rec + loss_cyc
                    accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = itertools.chain(transformer.parameters(),
                                                     material_traits_embeddings.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 0:
                        _ = log_validation(
                            args=args,
                            accelerator=accelerator,
                            validation_samples=validation_samples,
                            material_traits_embeddings=material_traits_embeddings,
                            vae=vae,
                            vae_scale_factor=vae_scale_factor,
                            transformer=transformer,
                            noise_scheduler_copy=noise_scheduler_copy,
                            torch_dtype=weight_dtype,
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)
        if args.upcast_before_saving:
            transformer.to(torch.float32)
        else:
            transformer = transformer.to(weight_dtype)
        transformer_lora_layers = get_peft_model_state_dict(transformer)

        FluxFillPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
        )

        # run inference
        images = []
        if args.validation_prompt and args.num_validation_images > 0:
            images = log_validation(
                args=args,
                accelerator=accelerator,
                validation_samples=validation_samples,
                material_traits_embeddings=material_traits_embeddings,
                vae=vae,
                transformer=transformer,
                noise_scheduler_copy=noise_scheduler_copy,
                vae_scale_factor=vae_scale_factor,
                is_final_validation=True,
                torch_dtype=weight_dtype,
            )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                instance_prompt=args.instance_prompt,
                validation_prompt=args.validation_prompt,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

        images = None

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)