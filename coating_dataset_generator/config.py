import bpy
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DatasetConfig:
    """Configuration settings for dataset generation"""
    output_root: str
    translation_range: float = 0
    rot_range: float = 1
    coating_materials_count: int = 64
    full_mask_probability: float = 0.4
    auto_frame_probability: float = 1.0
    frame_margin: float = 1.0
    resolution: int = 1024
    rand_seed: int = 21
    light_angle1_range: float = 45.0
    light_angle2_range: float = 60.0
    light_energy_min: float = 25.0
    light_energy_max: float = 100.0
    uniform_color_probability: float = 0.1
    transmission_probability: float = 0.35
    benchmark_mode: bool = False
    binary_thickness: bool = False
    filter_best_uv_map: bool = False

    # Continue from latest non-completed sample
    continue_generation: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.coating_materials_count <= 0:
            raise ValueError("coating_materials_count must be positive")
        if not (0 <= self.full_mask_probability <= 1):
            raise ValueError("full_mask_probability must be between 0 and 1")
        if not (0 <= self.auto_frame_probability <= 1):
            raise ValueError("auto_frame_probability must be between 0 and 1")
        if self.resolution <= 0:
            raise ValueError("resolution must be positive")
        if self.light_angle1_range < 0 or self.light_angle1_range > 180:
            raise ValueError("light_angle1_range must be between 0 and 180 degrees")
        if self.light_angle2_range < 0 or self.light_angle2_range > 90:
            raise ValueError("light_angle2_range must be between 0 and 90 degrees")
        if self.light_energy_min <= 0:
            raise ValueError("light_energy_min must be positive")
        if self.light_energy_max <= self.light_energy_min:
            raise ValueError("light_energy_max must be greater than light_energy_min")
        if not (0 <= self.uniform_color_probability <= 1):
            raise ValueError("uniform_color_probability must be between 0 and 1")
        if not (0 <= self.transmission_probability <= 1):
            raise ValueError("transmission_probability must be between 0 and 1")


def get_default_config() -> DatasetConfig:
    """Get default configuration with output path resolved"""
    output_root = os.path.abspath("coating_dataset_Userstudy")  # "coating_dataset_Training" # "coating_dataset_Benchmark" # coating_dataset_Userstudy
    config = DatasetConfig(output_root=output_root)

    config.continue_generation = True
    config.filter_best_uv_map = True

    # ========= Benchmarks =============
    # config.benchmark_mode = True
    # config.coating_materials_count = 67
    # config.rand_seed = 130

    # ========= User study =============
    # config.benchmark_mode = True
    # config.full_mask_probability = 1.0
    # config.transmission_probability = 0.1
    # config.rand_seed = 666
    # config.coating_materials_count = 5
    # config.uniform_color_probability = 0.4
    # config.binary_thickness = True

    # ========= Training =============
    config.rand_seed = 111
    # config.transmission_probability = 1.0 # TODO: for debugging.
    # config.coating_materials_count =  1 # TODO: for debugging.
    # config.resolution = 512 # TODO: for debugging.

    return config


def setup_output_directory(output_root: str) -> None:
    """Ensure output directory exists"""
    os.makedirs(output_root, exist_ok=True)