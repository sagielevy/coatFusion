"""
Material Trait Embeddings Module

This module handles trainable embeddings for material coating traits.
Each trait has both positional and value embeddings that are combined
to create the final trait representation.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import logging
import os
from default_prompt_embedding_generator import load_empty_embeddings, get_empty_embeddings_for_batch

logger = logging.getLogger(__name__)


class MaterialTraitEmbeddings(nn.Module):
    """
    Trainable embeddings for material coating traits.
    
    Each trait has:
    - positional embedding: learned embedding for the trait type
    - value embedding: learned embedding scaled by the trait value
    
    Final trait embedding = positional_embedding + (value * value_embedding)
    """
    
    TRAIT_NAMES = [
        "thickness",
        "metallic",
        "roughness",
        "transmission_weight",
        "apply_texture_task",
        "replace_task",
        "remove_task",
        "uv_mapping_spherical",
        "uv_mapping_cubic",
        "uv_mapping_original"
    ]
    
    # Binary traits that use two embeddings (on/off states)
    BINARY_TRAITS = {"metallic", "transmission_weight"}
    
    # Mapping from traits to their word keys in the pre-generated embeddings
    TRAIT_TO_WORDS = {
        "thickness": "thickness",
        "metallic": ["metallic", "dielectric"],  # on=metallic, off=dielectric
        "roughness": "roughness",
        "transmission_weight": ["transmissive", "opaque"]  # on=transmissive, off=opaque
    }
    
    
    def __init__(self, prompt_embedding_dim: int = 4096, pos_embedding_dim: int = 3,
                 word_embeddings_path: str = "material_default_embeddings.pt"):
        super().__init__()
        self.embedding_dim = prompt_embedding_dim
        self.pos_embedding_dim = pos_embedding_dim
        self.num_traits = len(self.TRAIT_NAMES)
        
        # Create mapping from trait names to embedding indices and calculate total embeddings
        self.trait_to_indices = {}
        current_idx = 0
        
        for trait_name in self.TRAIT_NAMES:
            # Positional embedding
            self.trait_to_indices[f"{trait_name}_positional"] = current_idx
            current_idx += 1
            
            # Value embedding(s)
            if trait_name in self.BINARY_TRAITS:
                self.trait_to_indices[f"{trait_name}_value_on"] = current_idx
                self.trait_to_indices[f"{trait_name}_value_off"] = current_idx + 1
                current_idx += 2
            else:
                self.trait_to_indices[f"{trait_name}_value"] = current_idx
                current_idx += 1
        
        total_embeddings = current_idx
        self.embeddings = nn.Embedding(total_embeddings, prompt_embedding_dim)
        
        # Initialize embeddings with pre-generated word embeddings if available
        self._initialize_with_word_embeddings(word_embeddings_path)
        
        logger.info(f"Initialized MaterialTraitEmbeddings with {total_embeddings} embeddings, dim={prompt_embedding_dim}")
    
    def _initialize_with_word_embeddings(self, word_embeddings_path: str):
        """Initialize embeddings with pre-generated word embeddings."""
        if not os.path.exists(word_embeddings_path):
            logger.warning(f"Word embeddings file not found: {word_embeddings_path}. Using random initialization.")
            return

        word_embeddings = torch.load(word_embeddings_path, map_location="cpu", weights_only=True)
        logger.info(f"Loaded word embeddings from {word_embeddings_path}")

        with torch.no_grad():
            for trait_name in self.TRAIT_NAMES:
                if trait_name not in self.TRAIT_TO_WORDS:
                    continue  # Skip task traits, keep random initialization

                word_keys = self.TRAIT_TO_WORDS[trait_name]

                if trait_name in self.BINARY_TRAITS:
                    # Binary trait: use two different word embeddings
                    on_word, off_word = word_keys
                    if on_word in word_embeddings and off_word in word_embeddings:
                        on_idx = self.trait_to_indices[f"{trait_name}_value_on"]
                        off_idx = self.trait_to_indices[f"{trait_name}_value_off"]

                        self.embeddings.weight[on_idx] = word_embeddings[on_word]
                        self.embeddings.weight[off_idx] = word_embeddings[off_word]
                        logger.info(f"Initialized {trait_name} binary embeddings with {on_word}/{off_word}")
                else:
                    # Continuous trait: use single word embedding
                    if word_keys in word_embeddings:
                        value_idx = self.trait_to_indices[f"{trait_name}_value"]
                        self.embeddings.weight[value_idx] = word_embeddings[word_keys]
                        logger.info(f"Initialized {trait_name} embedding with {word_keys}")

    def get_embedding(self, key: str) -> torch.Tensor:
        """
        Get a specific embedding by key.
        
        Args:
            key: Either "{trait_name}_positional" or "{trait_name}_value"
            
        Returns:
            Embedding tensor of shape [embedding_dim]
        """
        if key not in self.trait_to_indices:
            raise KeyError(f"Unknown embedding key: {key}. Available keys: {list(self.trait_to_indices.keys())}")
        
        idx = self.trait_to_indices[key]
        return self.embeddings.weight[idx]
    
    def generate_trait_embedding(self, trait_name: str, trait_value: torch.Tensor) -> torch.Tensor:
        """
        Generate the final embedding for a trait by combining positional and value embeddings.
        
        Args:
            trait_name: Name of the trait (e.g., "roughness")
            trait_value: Scalar value for the trait (0.0 or 1.0 for binary traits)
            
        Returns:
            Combined embedding: positional + value_embedding (selected based on trait type)
        """
        positional_key = f"{trait_name}_positional"
        positional_embedding = self.get_embedding(positional_key)
        
        if trait_name in self.BINARY_TRAITS:
            # Binary trait: select between on/off embeddings based on trait_value
            if trait_value > 0.5:  # treat as binary: > 0.5 = on, <= 0.5 = off
                value_embedding = self.get_embedding(f"{trait_name}_value_on")
            else:
                value_embedding = self.get_embedding(f"{trait_name}_value_off")
            # For binary traits, just add the selected embedding (no scaling)
            combined = positional_embedding + value_embedding
        else:
            # Continuous trait: scale the value embedding by trait_value
            value_embedding = self.get_embedding(f"{trait_name}_value")
            combined = positional_embedding + trait_value * value_embedding
        
        return combined

    def generate_prompt_embeddings(self, material_traits_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Generate prompt embeddings for all material traits.

        Args:
            material_traits_dict: Dictionary mapping trait names to their scalar values

        Returns:
            Embeddings tensor of shape [num_traits, embedding_dim]
        """
        trait_embeddings = []

        for trait_name in self.TRAIT_NAMES:
            if trait_name not in material_traits_dict:
                raise KeyError(f"Missing trait value for: {trait_name}")

            trait_value = material_traits_dict[trait_name]
            trait_embedding = self.generate_trait_embedding(trait_name, trait_value)
            trait_embeddings.append(trait_embedding)

        combined_embeddings = torch.stack(trait_embeddings, dim=0)
        return combined_embeddings

    def make_prompt_embeddings(self, batch_size: int, material_traits_dict: Dict[str, torch.Tensor], device: torch.device) -> tuple:
        """
        Generate prompt embeddings for material traits using provided batch values.
        
        Args:
            batch_size: Number of samples in the batch
            material_traits_dict: Dictionary mapping trait names to their batch tensor values
            device: Device to create tensors on
            
        Returns:
            Tuple of (prompt_embeds, pooled_prompt_embeds, text_ids)
        """
        # Load empty embeddings
        empty_embeddings = load_empty_embeddings()
        prompt_embeds, pooled_prompt_embeds, text_ids = get_empty_embeddings_for_batch(
            batch_size=batch_size,
            empty_embeddings_dict=empty_embeddings,
            device=device
        )

        # Generate material trait embeddings for each sample in the batch
        batch_trait_embeddings = []
        for i in range(batch_size):
            # Extract scalar values for this sample
            sample_traits_dict = {}
            for trait_name in self.TRAIT_NAMES:
                if trait_name not in material_traits_dict:
                    raise KeyError(f"Missing trait value for: {trait_name}")
                # Get the scalar value for this sample
                sample_traits_dict[trait_name] = material_traits_dict[trait_name][i]
            
            # Generate trait embeddings for this sample
            trait_embeddings = self.generate_prompt_embeddings(sample_traits_dict)
            batch_trait_embeddings.append(trait_embeddings)
        
        # Stack all sample embeddings into a batch tensor
        batch_trait_embeddings = torch.stack(batch_trait_embeddings, dim=0)
        
        # Concatenate with existing prompt embeddings
        prompt_embeds = torch.cat([prompt_embeds, batch_trait_embeddings], dim=1).to(dtype=text_ids.dtype)
        
        # Add corresponding zeros to text_ids to match the new sequence length
        trait_text_ids = torch.zeros((batch_size, batch_trait_embeddings.shape[1], self.pos_embedding_dim), dtype=text_ids.dtype, device=device)
        text_ids = torch.cat([text_ids, trait_text_ids], dim=1)
        
        return prompt_embeds, pooled_prompt_embeds, text_ids

    def __getitem__(self, key: str) -> torch.Tensor:
        """Allow dict-like access to embeddings."""
        return self.get_embedding(key)


def create_material_trait_embeddings() -> MaterialTraitEmbeddings:
    """
    Factory function to create MaterialTraitEmbeddings.
        
    Returns:
        MaterialTraitEmbeddings instance
    """
    return MaterialTraitEmbeddings()


def save_material_trait_embeddings(embeddings: MaterialTraitEmbeddings, save_path: str):
    """
    Save material trait embeddings to disk.
    
    Args:
        embeddings: MaterialTraitEmbeddings instance
        save_path: Path to save the embeddings
    """
    torch.save({
        'embedding_dim': embeddings.embedding_dim,
        'state_dict': embeddings.state_dict(),
        'trait_names': embeddings.TRAIT_NAMES,
        'trait_to_indices': embeddings.trait_to_indices
    }, save_path)
    logger.info(f"Saved material trait embeddings to {save_path}")


def load_material_trait_embeddings(save_path: str, device: Optional[torch.device] = None) -> MaterialTraitEmbeddings:
    """
    Load material trait embeddings from disk.
    
    Args:
        save_path: Path to load the embeddings from
        device: Device to load the embeddings to
        
    Returns:
        MaterialTraitEmbeddings instance
    """
    checkpoint = torch.load(save_path, map_location=device, weights_only=True)
    
    embeddings = MaterialTraitEmbeddings(prompt_embedding_dim=checkpoint['embedding_dim'])
    embeddings.load_state_dict(checkpoint['state_dict'])
    
    if device is not None:
        embeddings = embeddings.to(device)
    
    logger.info(f"Loaded material trait embeddings from {save_path}")
    return embeddings