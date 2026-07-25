"""
Coating Dataset Generator Package

A modular system for generating coating datasets in Blender following the Single Responsibility Principle.

Modules:
- config: Configuration management and validation
- file_manager: File I/O operations and path utilities  
- camera_operations: Camera positioning and framing
- material_generator: Material creation and property management
- scene_manager: Scene setup and resource management
- coating_system: Coating application and UV mapping
- render_engine: Rendering operations and passes
- dataset_orchestrator: Main orchestration logic
"""

from .config import DatasetConfig, get_default_config, setup_output_directory
from .dataset_orchestrator import generate_dataset
from .dataset_generation_batch_manager import main

__version__ = "2.0.0"
__author__ = "MaterialTransfer Team"

__all__ = [
    'DatasetConfig',
    'get_default_config', 
    'setup_output_directory',
    'generate_dataset',
    'main'
]