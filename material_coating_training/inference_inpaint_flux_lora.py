import argparse
import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from diffusers import FluxTransformer2DModel, FluxFillPipeline
from diffusers.utils import load_image, make_image_grid
import logging
from material_trait_embeddings import load_material_trait_embeddings
from mask_crop_helper import crop_images_to_bounding_box_by_mask
from material_coating_dataset import create_triptych_images
from torchvision import transforms

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_with_inpainting(
        pipeline,
        image,  # Now a triptych: [input | albedo | gray_fill]
        mask_image,  # Now a triptych mask: [mask | black | mask]
        material_traits_embeddings,
        material_traits_dict,
        mixed_precision,
        guidance_scale=3.5,
        num_inference_steps=28,
        generator=None,
        width=512,
        height=512,
        crop_result_only=False,
):
    """Custom inference function for flux inpainting with material traits

    Note: With triptych approach, albedo is embedded in the middle panel of the image.
    Image structure: [input | albedo | region_to_fill]
    Mask structure: [mask | black | mask]
    """
    device = pipeline.device
    batch_size = 1
    weight_dtype = get_dtype(mixed_precision)

    with torch.no_grad():
        # Generate prompt embeddings using material traits
        prompt_embeds, pooled_prompt_embeds, text_ids = material_traits_embeddings.make_prompt_embeddings(
            batch_size=batch_size,
            material_traits_dict=material_traits_dict,
            device=device,
        )

        prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=weight_dtype)

        # Use the standard pipeline - no albedos parameter needed!
        # The albedo is already visible in the middle panel of the triptych
        result = pipeline(
            image=image,
            mask_image=mask_image,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            generator=generator,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )

    # Extract the generated result from the right panel of the triptych
    # The output is a triptych, but we only want the coated panel for 'add' / 'replace' or the left panel (clean) for 'remove'
    if not crop_result_only:
        return result.images[0]

    is_remove_task = material_traits_dict["remove_task"] > 0.0

    if is_remove_task:
        generated_image = result.images[0].crop((0, 0, height, height))
    else:
        generated_image = result.images[0].crop((height * 2, 0, height * 3, height))

    return generated_image


def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint directory"""
    if not os.path.exists(output_dir):
        raise ValueError(f"Output directory {output_dir} does not exist")

    dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    if not dirs:
        raise ValueError(f"No checkpoints found in {output_dir}")

    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    latest_checkpoint = dirs[-1]

    logger.info(f"Found latest checkpoint: {latest_checkpoint}")
    return os.path.join(output_dir, latest_checkpoint)


def parse_base_color(color_str):
    """Parse base color string 'R,G,B' to tuple of floats"""
    try:
        r, g, b = color_str.split(',')
        return float(r), float(g), float(b)
    except ValueError:
        raise ValueError(f"Invalid base color format: {color_str}. Expected 'R,G,B' format.")


def convert_task_option_to_scalar(task_option):
    """Convert task option string to scalar values"""
    if task_option == "AT":
        return 1.0, 0.0, 0.0  # apply_texture_task, replace_task, remove_task
    elif task_option == "RL":
        return 0.0, 1.0, 0.0  # apply_texture_task, replace_task, remove_task
    elif task_option == "RM":
        return 0.0, 0.0, 1.0  # apply_texture_task, replace_task, remove_task
    else:
        raise ValueError(f"Invalid task option: {task_option}. Must be 'AT', 'RL', or 'RM'.")


def convert_uv_mapping_option_to_scalar(uv_mapping_option):
    """Convert UV mapping option string to scalar values"""
    uv_mapping_option = uv_mapping_option.lower()

    if uv_mapping_option == "cubic":
        return 1.0, 0.0, 0.0 # cubic, spherical, original
    elif uv_mapping_option == "spherical":
        return 0.0, 1.0, 0.0 # cubic, spherical, original
    elif uv_mapping_option == "original":
        return 0.0, 0.0, 1.0 # cubic, spherical, original
    else:
        raise ValueError(f"Invalid UV mapping option: {uv_mapping_option}. Must be 'cubic', 'spherical', or 'original'.")


def create_material_traits_dict(thickness, metallic, roughness, transmission_weight, task_option, uv_mapping_option):
    """
    Create material traits dictionary from individual parameters.
    
    Args:
        base_color_str: Base color string in format 'R,G,B'
        thickness: Thickness value
        metallic: Metallic value
        roughness: Roughness value
        transmission_weight: Transmission weight value
        task_option: Task option string ('AT', 'RL', or 'RM')
        uv_mapping_option: UV mapping option string ('cubic', 'spherical', or 'original')
        
    Returns:
        Dictionary with material traits
    """
    apply_texture_task, replace_task, remove_task = convert_task_option_to_scalar(task_option)
    uv_mapping_cubic, uv_mapping_spherical, uv_mapping_original = convert_uv_mapping_option_to_scalar(uv_mapping_option)

    return {
        "thickness": torch.tensor([thickness]),
        "metallic": torch.tensor([metallic]),
        "roughness": torch.tensor([roughness]),
        "transmission_weight": torch.tensor([transmission_weight]),
        "apply_texture_task": torch.tensor([apply_texture_task]),
        "replace_task": torch.tensor([replace_task]),
        "remove_task": torch.tensor([remove_task]),
        "uv_mapping_spherical": torch.tensor([uv_mapping_spherical]),
        "uv_mapping_cubic": torch.tensor([uv_mapping_cubic]),
        "uv_mapping_original": torch.tensor([uv_mapping_original])
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for FLUX Inpaint LoRA")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--lora_weights_path",
        type=str,
        required=True,
        help="Path to the trained LoRA weights directory (will use latest checkpoint if directory contains checkpoints).",
    )
    parser.add_argument(
        "--source_images",
        type=str,
        nargs="+",
        required=True,
        help="List of source image paths.",
    )
    parser.add_argument(
        "--coat_masks",
        type=str,
        nargs="+",
        required=True,
        help="List of coating mask paths (must match source_images length).",
    )
    parser.add_argument(
        "--albedos",
        type=str,
        nargs="+",
        required=True,
        help="List of albedo paths (must match source_images length). Set an instance to 'None' if for a given sample there is no albedo",
    )
    parser.add_argument(
        "--base_colors",
        type=str,
        nargs="+",
        required=True,
        help="List of base colors in format 'R,G,B' (e.g., '0.5,0.3,0.8').",
    )
    parser.add_argument(
        "--thicknesses",
        type=float,
        nargs="+",
        required=True,
        help="List of thickness values.",
    )
    parser.add_argument(
        "--metallics",
        type=float,
        nargs="+",
        required=True,
        help="List of metallic values.",
    )
    parser.add_argument(
        "--roughnesses",
        type=float,
        nargs="+",
        required=True,
        help="List of roughness values.",
    )
    parser.add_argument(
        "--transmission_weights",
        type=float,
        nargs="+",
        required=True,
        help="List of transmission weight values.",
    )
    parser.add_argument(
        "--task_options",
        type=str,
        nargs="+",
        required=True,
        help="List of task options ('AT', 'RL', or 'RM').",
    )
    parser.add_argument(
        "--uv_options",
        type=str,
        nargs="+",
        required=True,
        help="List of UV mapping options ('cubic', 'spherical', or 'original').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_outputs",
        help="Directory to save generated images.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Resolution for generated images.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="Guidance scale for generation.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=28,
        help="Number of inference steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode.",
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        help="Use mask-based cropping instead of resize when processing images",
    )

    args = parser.parse_args()

    # Validate input lengths
    if len(args.source_images) != len(args.coat_masks):
        raise ValueError("source_images and coat_masks must have the same length")

    if len(args.source_images) != len(args.albedos):
        raise ValueError("source_images and albedos must have the same length")
    
    material_property_lengths = [
        len(args.base_colors), len(args.thicknesses),
        len(args.metallics), len(args.roughnesses), len(args.transmission_weights),
        len(args.task_options), len(args.uv_options)
    ]
    if len(set(material_property_lengths)) != 1:
        raise ValueError("All material property arrays must have the same length")

    return args


def get_dtype(mixed_precision):
    weight_dtype = torch.float32
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    return weight_dtype


def reshape_images(idx, source_img, mask_img, albedo_img, resolution, crop):
    resize_transform = transforms.Resize((resolution, resolution),
                                         interpolation=transforms.InterpolationMode.BILINEAR,
                                         antialias=True)

    if crop:
        crop_bounds = crop_images_to_bounding_box_by_mask(logger, mask_img, resolution, idx)
        source_img = source_img.crop(crop_bounds)
        mask_img = mask_img.crop(crop_bounds)
    else:
        source_img = resize_transform(source_img)
        mask_img = resize_transform(mask_img)

    albedo_img = resize_transform(albedo_img)

    return source_img, mask_img, albedo_img


def main(args):
    # Create InferenceConfig from args
    config = InferenceConfig(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        lora_weights_path=args.lora_weights_path,
        resolution=args.resolution,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        mixed_precision=args.mixed_precision,
        crop=args.crop,
        output_dir=args.output_dir
    )

    # Setup pipeline using helper function
    pipeline, material_traits_embeddings, device, transforms_dict = setup_inference_pipeline(config)

    # Run inference using helper function
    all_results = run_inference_batch(
        pipeline=pipeline,
        material_traits_embeddings=material_traits_embeddings,
        device=device,
        transforms_dict=transforms_dict,
        config=config,
        source_images=args.source_images,
        coating_masks=args.coat_masks,
        albedos=args.albedos,
        base_colors=args.base_colors,
        thicknesses=args.thicknesses,
        uv_options=args.uv_options,
        metallics=args.metallics,
        roughnesses=args.roughnesses,
        transmission_weights=args.transmission_weights,
        task_options=args.task_options
    )

    total_combinations = len(args.source_images) * len(args.base_colors)

    # Create and save grid
    logger.info("Creating result grid...")

    # Create grid with 3 columns: source, mask, generated
    # Each row represents one result combination
    grid_images = []
    for result in all_results:
        # Create a row with 3 images: source, mask, generated
        row_images = [
            result['source_image'].resize((args.resolution // 2, args.resolution // 2)),
            result['mask_image'].resize((args.resolution // 2, args.resolution // 2)),
            result['albedo'].resize((args.resolution // 2, args.resolution // 2)),
            result['generated_image'].resize((args.resolution // 2, args.resolution // 2))
        ]
        grid_images.extend(row_images)  # Add all images to the grid list

    # Create final grid: 4 columns, with rows = total_combinations
    final_grid = make_image_grid(grid_images, rows=len(all_results), cols=4)

    grid_path = os.path.join(args.output_dir, "results_grid.png")
    final_grid.save(grid_path)
    logger.info(f"Results grid saved to: {grid_path}")

    logger.info(f"Inference complete! Generated {total_combinations} images.")
    logger.info(f"Individual images saved to: {args.output_dir}")
    logger.info(f"Results grid saved to: {grid_path}")


class InferenceConfig:
    """Configuration class for inference parameters"""
    def __init__(self, 
                 pretrained_model_name_or_path="black-forest-labs/FLUX.1-Fill-dev",
                 lora_weights_path=None,
                 resolution=512,
                 guidance_scale=30.0,
                 num_inference_steps=50,
                 seed=43,
                 mixed_precision="bf16",
                 crop=True,
                 output_dir=None):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.lora_weights_path = lora_weights_path
        self.resolution = resolution
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.seed = seed
        self.mixed_precision = mixed_precision
        self.crop = crop
        self.output_dir = output_dir


def setup_inference_pipeline(config: InferenceConfig):
    """
    Set up the inference pipeline with the given configuration.
    
    Returns:
        tuple: (pipeline, material_traits_embeddings, device, transforms_dict)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = get_dtype(config.mixed_precision)

    # Set seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Load transformer
    transformer = FluxTransformer2DModel.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=weight_dtype
    )

    # Create pipeline
    pipeline = FluxFillPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        transformer=transformer,
        torch_dtype=weight_dtype,
    )

    # Load LoRA weights
    lora_path = config.lora_weights_path
    if os.path.isdir(lora_path) and any(d.startswith("checkpoint") for d in os.listdir(lora_path)):
        lora_path = find_latest_checkpoint(lora_path)

    pipeline.load_lora_weights(lora_path)
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=False)

    # Load material trait embeddings
    material_embeddings_path = os.path.join(lora_path, "material_trait_embeddings.pt")
    if os.path.exists(material_embeddings_path):
        material_traits_embeddings = load_material_trait_embeddings(material_embeddings_path, device=device)
        logger.info(f"Loaded material trait embeddings from: {material_embeddings_path}")
    else:
        raise FileNotFoundError(f"Material trait embeddings not found at: {material_embeddings_path}")

    # Setup transforms
    transforms_dict = {
        'mask': transforms.Compose([transforms.ToTensor()]),
        'image': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    }

    return pipeline, material_traits_embeddings, device, transforms_dict


def run_inference_batch(pipeline, material_traits_embeddings, device, transforms_dict, config: InferenceConfig,
                        source_images, coating_masks, albedos, base_colors, thicknesses, metallics, roughnesses,
                        transmission_weights, uv_options, task_options):
    """
    Run inference on a batch of images and material combinations.

    Returns:
        list: List of result dictionaries containing generated images and metadata
    """
    mask_transform = transforms_dict['mask']
    image_transform = transforms_dict['image']

    all_results = []
    total_combinations = len(source_images) * len(base_colors)

    logger.info(f"Generating {total_combinations} images...")

    for i, (source_path, mask_path, albedo_path) in enumerate(zip(source_images, coating_masks, albedos)):
        source_img = load_image(source_path)
        mask_img = load_image(mask_path)
        is_uniform_color = albedo_path == "None"
        albedo_img = load_image(albedo_path) if not is_uniform_color else Image.new("RGB", source_img.size, (0, 0, 0))

        # Apply binary threshold to coating mask
        mask_array = np.array(mask_img)
        mask_array = np.where(mask_array < 10, 0, 255)
        mask_img = Image.fromarray(mask_array.astype(np.uint8))

        # Reshape images
        source_img, mask_img, albedo_img = reshape_images(i, source_img, mask_img, albedo_img, config.resolution,
                                                          config.crop)

        for j in range(len(base_colors)):
            base_color_str = base_colors[j]
            thickness = thicknesses[j]
            metallic = metallics[j]
            roughness = roughnesses[j]
            transmission_weight = transmission_weights[j]
            task_option = task_options[j]
            uv_mapping_option = uv_options[j]

            # Create material traits dictionary
            material_traits_dict = create_material_traits_dict(
                thickness, metallic, roughness, transmission_weight, task_option, uv_mapping_option
            )

            # Handle uniform colors
            current_albedo_img = albedo_img.copy()
            if is_uniform_color:
                base_color_r, base_color_g, base_color_b = parse_base_color(base_color_str)

                h, w = mask_img.size
                uniform_color_array = np.zeros((h, w, 3), dtype=np.float32)
                uniform_color_array[...] = [base_color_r, base_color_g, base_color_b]
                current_albedo_img = Image.fromarray((uniform_color_array * 255).astype(np.uint8))

            combination_num = i * len(base_colors) + j + 1
            material_description = f"seed={config.seed}, color={base_color_str}, thickness={thickness}, metallic={metallic}, roughness={roughness}, transmission={transmission_weight}, task={task_option}, uv={uv_mapping_option}"
            logger.info(f"Processing combination {combination_num}/{total_combinations}: {material_description}")

            is_remove_task = material_traits_dict["remove_task"] > 0.0

            if is_remove_task:
                _, triptych_img, triptych_mask = create_triptych_images(
                    None, None, source_img, mask_img, is_remove_task
                )
            else:
                _, triptych_img, triptych_mask = create_triptych_images(
                    source_img, current_albedo_img, None, mask_img, is_remove_task
                )

            triptych_img_tensor = image_transform(triptych_img).unsqueeze(0).to(device)
            triptych_mask_tensor = mask_transform(triptych_mask).unsqueeze(0).to(device)

            # Generate image using triptych in-context learning
            # The model will fill the right panel based on left (input) and middle (albedo) panels
            generated_triptych = generate_with_inpainting(
                pipeline=pipeline,
                image=triptych_img_tensor,
                mask_image=triptych_mask_tensor,
                material_traits_embeddings=material_traits_embeddings,
                material_traits_dict=material_traits_dict,
                mixed_precision=config.mixed_precision,
                guidance_scale=config.guidance_scale,
                num_inference_steps=config.num_inference_steps,
                generator=torch.Generator(device=device).manual_seed(config.seed),
                width=config.resolution * 3, # Triptych is 3x wider!
                height=config.resolution,
            )

            generated_image = generated_triptych

            # Save individual image if output_dir is provided
            output_path = None
            if config.output_dir:
                os.makedirs(config.output_dir, exist_ok=True)
                output_filename = f"generated_{i:03d}_{j:03d}_{Path(source_path).stem}_{material_description}.png"
                output_path = os.path.join(config.output_dir, output_filename)
                generated_image.save(output_path)

            # Store result
            result = {
                'source_image': source_img,
                'mask_image': mask_img,
                'albedo': current_albedo_img,
                'generated_image': generated_image,
                'source_path': source_path,
                'material_traits': material_traits_dict,
                'output_path': output_path,
                'material_description': material_description
            }
            all_results.append(result)

    return all_results


if __name__ == "__main__":
    main(parse_args())