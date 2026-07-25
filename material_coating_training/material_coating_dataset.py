"""
Material Coating Dataset Module

This module contains the dataset classes and data loading utilities
for the material coating LoRA training pipeline.
"""

import logging
import os
import random
from typing import Dict, List, Union, Any, Optional

import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms
from mask_crop_helper import crop_images_to_bounding_box_by_mask

logger = logging.getLogger(__name__)


class MaterialCoatingDataset(Dataset):
    """
    Dataset class for material coating training with on-demand loading and data augmentation.

    Features:
    - On-demand image loading to reduce memory usage
    - Consistent data augmentation across all related images
    - Error handling for corrupted/missing images
    - Configurable augmentation parameters
    - Support for coating-specific properties (thickness, material properties)
    """

    def __init__(self, hf_dataset, args):
        """
        Initialize the dataset.

        Args:
            hf_dataset: HuggingFace dataset object
            args: Training arguments containing configuration
        """
        self.hf_dataset = hf_dataset
        self.args = args

        self.aug_prob = getattr(args, 'augmentation_prob', 0.5)
        self.normal_drop = getattr(args, 'normal_drop', 0.5)
        self.enable_augmentation = getattr(args, 'enable_data_augmentation', True)
        self.crop = getattr(args, 'crop', False)
        self.use_best_uv_mapping = getattr(args, 'use_best_uv_mapping', False)
        self.filter_out_planar_mapping = getattr(args, 'filter_out_planar_mapping', False)
        self.transmissive_filter = getattr(args, 'transmissive_filter', None)

        self.filtered_indices = list(range(len(self.hf_dataset)))

        if self.transmissive_filter:
            self.filtered_indices = self._filter_by_transmission_weight()

        if self.use_best_uv_mapping:
            self.filtered_indices = self._filter_by_uv_mapping()

        if self.filter_out_planar_mapping:
            self.filtered_indices = self._filter_out_planar_uv_mapping()

        # Base transforms (without augmentation)
        self.base_image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.resize_transform = transforms.Resize((args.resolution, args.resolution),
                                                  interpolation=transforms.InterpolationMode.BILINEAR,
                                                  antialias=True)
        self.base_control_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        logger.info(f"Initialized MaterialCoatingDataset with {len(self.hf_dataset)} samples")
        if self.transmissive_filter:
            logger.info(f"Applied {self.transmissive_filter} filter, reduced to {len(self.filtered_indices)} samples")
        if self.use_best_uv_mapping:
            logger.info(f"Applied use_best_uv_mapping filter, reduced to {len(self.filtered_indices)} samples")
        logger.info(f"Augmentation enabled: {self.enable_augmentation}, prob: {self.aug_prob}")

    def _filter_by_uv_mapping(self):
        """Filter dataset to only include items where uv_mapping matches best_mapping_method."""
        new_filtered_indices = []
        for idx in self.filtered_indices:
            item = self.hf_dataset[idx]
            if item["uv_mapping"] == item["best_mapping_method"]:
                new_filtered_indices.append(idx)
        return new_filtered_indices

    def _filter_out_planar_uv_mapping(self):
        """Filter dataset to only include items where uv_mapping matches best_mapping_method."""
        new_filtered_indices = []
        for idx in self.filtered_indices:
            item = self.hf_dataset[idx]
            if item["uv_mapping"] != "PLANAR":
                new_filtered_indices.append(idx)
        return new_filtered_indices

    def _filter_by_transmission_weight(self):
        """Filter dataset based on transmission weight values."""
        new_filtered_indices = []

        for idx in self.filtered_indices:
            item = self.hf_dataset[idx]
            transmission_weight = item["transmission_weight"]

            if self.transmissive_filter == "transmissive_only":
                if transmission_weight > 0.0:
                    new_filtered_indices.append(idx)
            elif self.transmissive_filter == "non_transmissive_only":
                if transmission_weight <= 0.0:
                    new_filtered_indices.append(idx)

        return new_filtered_indices

    def __len__(self) -> int:
        return len(self.filtered_indices)

    def load_image(self, image_path_or_obj: Union[str, Image.Image]) -> Image.Image:
        """
        Args:
            image_path_or_obj: Path to image file or PIL Image object

        Returns:
            PIL Image object
        """
        if isinstance(image_path_or_obj, str):
            return Image.open(image_path_or_obj).convert("RGB")
        else:
            return image_path_or_obj.convert("RGB")

    def apply_consistent_augmentation(self, idx, target_image, source_image, normal_image, coating_mask_image, albedo):
        """
        Apply the same random augmentation to all images in the list.
        This ensures spatial consistency between target, mask, and material reference images.

        Args:
            images: List of PIL Images to augment and a bool indicating if each one is a mask or not.
            coating_mask_image: PIL Image to guide crop.
        Returns:
            List of augmented PIL Images
        """
        if not self.enable_augmentation:
            return self.apply_reshaping(target_image, source_image, normal_image, coating_mask_image, albedo, idx, False)

        apply_hflip = random.random() < self.aug_prob

        if apply_hflip:
            images = [target_image, source_image, normal_image, coating_mask_image, albedo]

            for i, img in enumerate(images):
                images[i] = img.transpose(Image.FLIP_LEFT_RIGHT)

        return self.apply_reshaping(target_image, source_image, normal_image, coating_mask_image, albedo, idx, jitter=True)

    def apply_reshaping(self, target_image, source_image, normal_image, coating_mask_image, albedo, idx, jitter=False):
        if self.crop:
            crop_bounds = crop_images_to_bounding_box_by_mask(logger, coating_mask_image, self.args.resolution, idx, jitter=jitter)
            all_except_albedo = [target_image, source_image, normal_image, coating_mask_image]
            for i, image in enumerate(all_except_albedo):
                all_except_albedo[i] = image.crop(crop_bounds)

            return all_except_albedo + [self.resize_transform(albedo)]
        else:
            images = [target_image, source_image, normal_image, coating_mask_image, albedo]
            for i, image in enumerate(images):
                images[i] = self.resize_transform(image)
        return images

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single item from the dataset.

        Args:
            idx: Index of the item to retrieve

        Returns:
            Dictionary containing processed tensors and caption
        """
        actual_idx = self.filtered_indices[idx]
        item = self.hf_dataset[actual_idx]

        target_image_path = item["target_image"]
        source_image = self.load_image(item["image"])
        coating_mask_image = self.load_image(item["coating_mask"])
        normal_image = self.load_image(item["normal"])
        target_image = self.load_image(target_image_path)

        albedo_texture_exists = item["albedo_image"] is not None
        albedo = self.load_image(item["albedo_image"]) if albedo_texture_exists \
            else Image.new('RGB', (self.args.resolution, self.args.resolution), color=(0, 0, 0))

        # Apply binary threshold to coating mask (< 10 = black, >= 10 = white)
        mask_array = np.array(coating_mask_image)
        mask_array = np.where(mask_array < 10, 0, 255)
        coating_mask_image = Image.fromarray(mask_array.astype(np.uint8))

        # Apply consistent augmentation to all images. Should not crop the albedo since it is no longer projected.
        augmented_images = self.apply_consistent_augmentation(idx, target_image, source_image, normal_image,
                                                              coating_mask_image, albedo)
        target_image, source_image, normal_image, coating_mask_image, albedo = augmented_images

        # Handle uniform colors: create colored albedo when apply_uniform_task is active
        uniform_color_r = item["base_color_r"]
        uniform_color_g = item["base_color_g"]
        uniform_color_b = item["base_color_b"]
        is_uniform_task = not albedo_texture_exists
        thickness = torch.tensor(item["thickness"], dtype=torch.float32)
        metallic = torch.tensor(item["metallic"], dtype=torch.float32)
        roughness = torch.tensor(item["roughness"], dtype=torch.float32)
        transmission_weight = torch.tensor(item["transmission_weight"], dtype=torch.float32)

        uv_mapping_str = item["uv_mapping"]
        uv_mapping_spherical = torch.tensor(1.0, dtype=torch.float32) if uv_mapping_str == "SPHERICAL" else torch.tensor(0.0, dtype=torch.float32)
        uv_mapping_cubic = torch.tensor(1.0, dtype=torch.float32) if uv_mapping_str == "CUBIC" else torch.tensor(0.0, dtype=torch.float32)
        uv_mapping_original = torch.tensor(1.0, dtype=torch.float32) if uv_mapping_str == "ORIGINAL" else torch.tensor(0.0, dtype=torch.float32)

        apply_texture_task = torch.tensor(item["apply_texture_task"], dtype=torch.float32)
        replace_task = torch.tensor(item["replace_task"], dtype=torch.float32)
        remove_task = torch.tensor(item["remove_task"], dtype=torch.float32)

        # Create uniform color albedo if this is a uniform task
        if is_uniform_task:
            mask_array = np.array(coating_mask_image)
            h, w = mask_array.shape[:2]
            uniform_color_array = np.zeros((h, w, 3), dtype=np.float32)
            uniform_color_array[...] = [uniform_color_r, uniform_color_g, uniform_color_b]

            albedo = Image.fromarray((uniform_color_array * 255).astype(np.uint8))
        else:
            pass

        is_remove_task = item["remove_task"] > 0.0

        if is_remove_task:
            clean_image = target_image  # Clean image
            coated_image = source_image  # Coated image
        else:
            clean_image = source_image  # Clean or old coating
            coated_image = target_image  # Coated image

        triptych, masked_triptych, triptych_mask = create_triptych_images(
            clean_image=clean_image, albedo=albedo, coated_image= coated_image, mask=coating_mask_image,
            is_remove_task=is_remove_task,
        )

        source_image_tensor = self.base_image_transform(masked_triptych)
        target_image_tensor = self.base_image_transform(triptych)

        # Convert triptych mask to tensor
        mask_tensor = self.mask_transform(triptych_mask)

        filename = os.path.basename(target_image_path)
        parent_folder = os.path.basename(os.path.dirname(target_image_path))
        original_dataset_path = f"{parent_folder}/{filename}"

        return {
            "pixel_values": target_image_tensor,
            "thickness": thickness,
            "metallic": metallic,
            "roughness": roughness,
            "transmission_weight": transmission_weight,
            "apply_texture_task": apply_texture_task,
            "uv_mapping_spherical": uv_mapping_spherical,
            "uv_mapping_cubic": uv_mapping_cubic,
            "uv_mapping_original": uv_mapping_original,
            "replace_task": replace_task,
            "remove_task": remove_task,
            "masks": mask_tensor,
            "source_images": source_image_tensor,
            "original_dataset_path": original_dataset_path
        }


def create_triptych_images(clean_image: Optional[Image.Image], albedo: Optional[Image.Image],
                           coated_image: Optional[Image.Image], mask: Image.Image, is_remove_task: bool):
    """
    Create a three-panel horizontal concatenation.
    """
    w, h = mask.size
    triptych = Image.new('RGB', (w * 3, h), 0)

    # Paste images horizontally: [clean | albedo | coated]. For inference some panels would be missing. We build:
    # Add / replace: [clean | albedo | clean]
    # Remove: [coated | black | coated]
    # The coat mask will make a hole where the coating is to be applied or removed, leaving the clean image background.
    if clean_image is not None:
        triptych.paste(clean_image, (0, 0))

    if albedo is not None:
        triptych.paste(albedo, (w, 0))

    if coated_image is not None:
        triptych.paste(coated_image, (w * 2, 0))

    if coated_image is None and clean_image is not None and not is_remove_task:
        triptych.paste(clean_image, (w * 2, 0))
    elif clean_image is None and coated_image is not None and is_remove_task:
        triptych.paste(coated_image, (0, 0))

    triptych_mask = Image.new('L', (w * 3, h), 0)

    if is_remove_task:
        # [clean | albedo | coated]
        #            ||
        #            \/
        # [mask  |  white | black]
        white_mask = Image.new('L', (w, h), 255)
        triptych_mask.paste(mask, (0, 0))
        triptych_mask.paste(white_mask, (w, 0))
    else: # Add / replace task
        # [clean | albedo | coated]
        #            ||
        #            \/
        # [black |  black | mask]
        triptych_mask.paste(mask, (w * 2, 0))

    masked_triptych = triptych.copy()
    # Convert mask to binary array
    mask_array_3panel = np.array(triptych_mask) / 255.0  # Normalize to [0, 1]
    mask_array_3panel_rgb = np.stack([mask_array_3panel] * 3, axis=-1)

    # Apply masking: preserve unmasked regions, set masked regions to gray
    triptych_array = np.array(masked_triptych).astype(np.float32) / 255.0
    triptych_array = triptych_array * (1 - mask_array_3panel_rgb) + 0.5 * mask_array_3panel_rgb
    masked_triptych = Image.fromarray((triptych_array * 255).astype(np.uint8))

    return triptych, masked_triptych, triptych_mask


def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Enhanced collate function with error handling for on-demand loading.

    Args:
        examples: List of dataset items

    Returns:
        Batched tensors and captions
    """
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    thickness = torch.stack([example["thickness"] for example in examples])
    metallic = torch.stack([example["metallic"] for example in examples])
    roughness = torch.stack([example["roughness"] for example in examples])
    transmission_weight = torch.stack([example["transmission_weight"] for example in examples])
    uv_mapping_spherical = torch.stack([example["uv_mapping_spherical"] for example in examples])
    uv_mapping_cubic = torch.stack([example["uv_mapping_cubic"] for example in examples])
    uv_mapping_original = torch.stack([example["uv_mapping_original"] for example in examples])
    apply_texture_task = torch.stack([example["apply_texture_task"] for example in examples])
    replace_task = torch.stack([example["replace_task"] for example in examples])
    remove_task = torch.stack([example["remove_task"] for example in examples])

    masks = torch.stack([example["masks"] for example in examples])
    masks = masks.to(memory_format=torch.contiguous_format).float()
    source_images = torch.stack([example["source_images"] for example in examples])
    source_images = source_images.to(memory_format=torch.contiguous_format).float()
    original_dataset_paths = [example["original_dataset_path"] for example in examples]

    return {
        "pixel_values": pixel_values,
        "thickness": thickness,
        "metallic": metallic,
        "roughness": roughness,
        "transmission_weight": transmission_weight,
        "uv_mapping_spherical": uv_mapping_spherical,
        "uv_mapping_cubic": uv_mapping_cubic,
        "uv_mapping_original": uv_mapping_original,
        "apply_texture_task": apply_texture_task,
        "replace_task": replace_task,
        "remove_task": remove_task,
        "masks": masks,
        "source_images": source_images,
        "original_dataset_paths": original_dataset_paths
    }

def get_train_dataset(args, accelerator):
    """
    Load and prepare the training dataset from various sources.

    Args:
        args: Training arguments
        accelerator: Accelerate accelerator object

    Returns:
        HuggingFace dataset object
    """
    dataset = None

    # Load dataset from different sources
    if args.dataset_name is not None:
        logger.info(f"Loading dataset from HuggingFace Hub: {args.dataset_name}")
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )

    if args.jsonl_for_train is not None:
        logger.info(f"Loading dataset from JSONL file: {args.jsonl_for_train}")
        dataset = load_dataset("json", data_files=args.jsonl_for_train, cache_dir=args.cache_dir)
        dataset = dataset.flatten_indices()

    if dataset is None:
        raise ValueError("No dataset specified. Use either --dataset_name or --jsonl_for_train")

    # Validate dataset structure
    column_names = dataset["train"].column_names
    required_columns = ["image", "coating_mask", "normal", "target_image", "albedo_image", "thickness", "metallic",
                        "roughness", "transmission_weight", "uv_mapping", "best_mapping_method",
                        "apply_texture_task", "replace_task", "remove_task"]

    missing_columns = [col for col in required_columns if col not in column_names]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}. Available columns: {column_names}")

    logger.info(f"Dataset columns: {column_names}")
    logger.info(f"Dataset size: {len(dataset['train'])}")

    # Prepare dataset with shuffling and sampling
    with accelerator.main_process_first():
        train_dataset = dataset["train"].shuffle(seed=args.seed)
        if args.max_train_samples is not None:
            original_size = len(train_dataset)
            train_dataset = train_dataset.select(range(min(args.max_train_samples, original_size)))
            logger.info(f"Limited dataset from {original_size} to {len(train_dataset)} samples")

    return train_dataset


def prepare_train_dataset(dataset, accelerator, args):
    """
    Prepare the training dataset with custom MaterialCoatingDataset class.

    Args:
        dataset: HuggingFace dataset object
        accelerator: Accelerate accelerator object
        args: Training arguments

    Returns:
        MaterialCoatingDataset object
    """
    logger.info("Preparing training dataset with on-demand loading and augmentation")

    # Create custom dataset with on-demand loading
    custom_dataset = MaterialCoatingDataset(dataset, args)

    logger.info(f"Dataset prepared with {len(custom_dataset)} samples")
    return custom_dataset


def create_train_dataloader(dataset, args):
    """
    Create a training dataloader with optimized settings.

    Args:
        dataset: MaterialCoatingDataset object
        args: Training arguments

    Returns:
        PyTorch DataLoader
    """
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True if args.dataloader_num_workers > 0 else False,  # Keep workers alive
        drop_last=True,  # Avoid issues with different batch sizes
    )