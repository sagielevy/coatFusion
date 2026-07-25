import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from torchvision import transforms
from math import ceil

class AlbedoEncoder(nn.Module):
    def __init__(self, output_dim, image_resolution, dino_model_name="facebook/dinov2-small", device="cuda"):
        super().__init__()
        self.dino = AutoModel.from_pretrained(dino_model_name).to(device=device)
        self.dino.requires_grad_(False)  # Freeze DinoV2
        
        self.dino_dim = self.dino.config.hidden_size
        self.output_dim = output_dim
        
        # Small MLP to project DinoV2 tokens to Flux text embedding dimension
        self.mlp = nn.Sequential(
            nn.Linear(self.dino_dim, self.dino_dim * 2),
            nn.GELU(),
            nn.Linear(self.dino_dim * 2, output_dim)
        ).to(device=device)
        
        # Positional embeddings
        # Assuming 518x518 input for DinoV2 (closest multiple of 14 to 512)
        # 518 / 14 = 37. 37 * 37 = 1369 patches.
        side_patch = int(ceil(image_resolution / 14))
        closest_fitting_image_res = int(side_patch * 14)
        self.num_patches = side_patch * side_patch
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, output_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Image processor for normalization
        self.processor = AutoImageProcessor.from_pretrained(dino_model_name)
        self.normalize = transforms.Normalize(mean=self.processor.image_mean, std=self.processor.image_std)
        self.resize = transforms.Resize((closest_fitting_image_res, closest_fitting_image_res), interpolation=transforms.InterpolationMode.BICUBIC)

    def forward(self, pixel_values, device):
        """
        Args:
            pixel_values: Tensor of shape (B, C, H, W) in range [-1, 1]
        Returns:
            projected_tokens: Tensor of shape (B, num_patches, output_dim)
        """
        # 1. Convert [-1, 1] to [0, 1]
        images = (pixel_values + 1.0) / 2.0
        
        # 2. Resize to 518x518
        images = self.resize(images)
        
        # 3. Normalize
        images = self.normalize(images).to(device=device)
        
        # 4. Pass through DinoV2
        outputs = self.dino(pixel_values=images)
        last_hidden_state = outputs.last_hidden_state  # (B, N+1, D)
        
        # 5. Get patch tokens (skip CLS)
        patch_tokens = last_hidden_state[:, 1:, :]  # (B, N, D)
        
        # 6. Project
        projected = self.mlp(patch_tokens)  # (B, N, output_dim)
        
        # 7. Add positional embeddings
        # Ensure pos_embed matches sequence length
        if projected.shape[1] != self.pos_embed.shape[1]:
             # If size mismatch, we might need to interpolate pos_embed or error out.
             # For now, assume fixed size.
             raise Exception(f"Mismatch between sequence length and pos_embed size: projected = {projected.shape[1]} vs pos_embed = {self.pos_embed.shape[1]}")

        return projected + self.pos_embed
