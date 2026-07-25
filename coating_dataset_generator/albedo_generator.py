import torch
from diffusers import FluxPipeline
from tqdm import tqdm
import random
import sys
import os

"""
This script generates material albedo images using the FLUX model.
It takes a start index and number of images to generate as command line arguments. The generated images are saved in the specified output directory.
"""

if len(sys.argv) != 3:
    raise ValueError("Usage: python albedo_generator.py <start_index> <num_images>")

start_index = int(sys.argv[1])
num_images = int(sys.argv[2])
end_index = start_index + num_images

print(f"Generating {num_images} images from index {start_index} to {end_index - 1}")

# Ensure output directory exists
os.makedirs("coating_dataset_generator/material_albedos", exist_ok=True)

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to('cuda')
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

colors = [
    "Red",
    "Orange",
    "Yellow",
    "Green",
    "Blue",
    "Purple",
    "Brown",
    "Pink",
    "Black",
    "White",
    "Gray",
    "Cyan",
    "Magenta",
    "Lavender",
    "Maroon",
    "Teal",
    "Turquoise"
]

img_type = ["pattern", "logo", "graffiti"]

for i in tqdm(range(start_index, end_index)):
    prompt = f"{random.choice(img_type)} where the dominant color is {random.choice(colors)}"

    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(i)
    ).images[0]
    image.save(f"coating_dataset_generator/Benchmarks/material_albedos_userstudy/flux-dev_{i}.png")
