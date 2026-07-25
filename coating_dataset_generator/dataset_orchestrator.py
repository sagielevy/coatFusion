try:
    import bpy
except ImportError:
    pass

import random
import os
import csv
from mathutils import Vector, Euler
from typing import Optional, List, Set, Dict, Any
import logging

from .config import DatasetConfig, get_default_config
from .scene_manager import (
    get_scene_resources, setup_scene_for_sample, prepare_coating_masks,
    set_object_visibility_recursive, find_layer_collection
)
from .geometry_utils import get_object_bounds_recursive
from .camera_operations import frame_object_in_camera, restore_camera_state, store_camera_state
from .light_operations import configure_lights_for_scene, restore_lights
from .material_generator import (
    create_random_textured_or_uniform_material, get_material_properties,
    store_original_material_properties, apply_material_augmentation,
    restore_material_properties, UVMappingType
)
from .coating_system import (
    get_coating_albedo_uv_map, apply_coating_material_to_cover, restore_cover_material
)
from .render_engine import (
    setup_enhanced_rendering, render_image, render_normal_pass,
    render_depth_pass, render_coating_mask, render_albedo_pass
)
from .file_manager import (
    set_output_paths, find_and_rename_passes, organize_output_files,
    cleanup_temp_files, save_prompts_json, get_material_name, is_sample_complete
)


class DatasetOrchestrator:
    """Main orchestrator for dataset generation"""
    
    def __init__(self, config: DatasetConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.resources = None
        self.compositor_nodes = None
        self.best_uv_mappings = self._load_best_uv_mappings()

    def _load_best_uv_mappings(self) -> Dict[str, str]:
        """Load the best UV mappings from the CSV file"""
        mappings = {}
        csv_path = os.path.join(os.path.dirname(__file__), 'BestUVMappings.csv')

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)

            for row in reader:
                mappings[row[0]] = row[1]

        return mappings

    def is_planar_object(self, collection_name):
        return collection_name.startswith("Quad ")

    def get_uvs_count(self, collection_name):
        if self.is_planar_object(collection_name):
            return 1  # Only generate planar UVs for quad objects
        return len(UVMappingType) - 1  # Exclude PLANAR

    def setup(self, initial_index: int = 0, 
              indices_list: Optional[List[int]] = None, 
              names_list: Optional[List[str]] = None) -> None:
        """Initialize resources and rendering setup"""
        self.logger.info("Setting up dataset generation...")

        self.resources = get_scene_resources(self.logger, self.config)
        self.compositor_nodes = setup_enhanced_rendering(self.config)
        
        first_index = initial_index
        if indices_list is not None and len(indices_list) > 0:
            first_index = min(indices_list)
        elif names_list is not None and len(names_list) > 0:
            sample_count = 0
            target_names = set(names_list)
            for object_pair in self.resources['object_pairs']:
                if object_pair.collection.name in target_names:
                    first_index = sample_count
                    break
                sample_count += len(self.resources['cameras']) * self.get_uvs_count(object_pair.collection.name)

        random.seed(self.config.rand_seed + first_index)
        
        self.logger.info(f"Found {len(self.resources['object_pairs'])} object pairs")
        self.logger.info(f"Found {len(self.resources['cameras'])} cameras")
        self.logger.info(f"Found {len(self.resources['floors'])} floors")
        self.logger.info(f"Found {len(self.resources['coating_masks'])} coating masks")
        self.logger.info(f"Found {len(self.resources['hdri_pool'])} HDRIs")
        self.logger.info(f"Found {len(self.resources['albedo_textures'])} albedo textures")
        self.logger.info(f"Found {len(self.resources['lights'])} area lights")

    def generate_dataset(self, initial_index: int = 0, num_samples: int = 1575,
                        indices_list: Optional[List[int]] = None,
                        names_list: Optional[List[str]] = None) -> None:
        """Main function to generate the dataset"""
        self.setup(initial_index, indices_list, names_list)
        
        obj_mask_output, obj_id_mask, depth_output = self.compositor_nodes
        
        sample_count = 0
        rendered_sample_count = 0
        
        target_indices = None
        target_names = None
        if indices_list is not None:
            target_indices = set(indices_list)
            self.logger.info(f"Using specific indices: {sorted(target_indices)}")
        elif names_list is not None:
            target_names = set(names_list)
            self.logger.info(f"Using specific names: {sorted(target_names)}")
        else:
            target_indices = set(range(initial_index, initial_index + num_samples))
            self.logger.info(f"Using range: {initial_index} to {initial_index + num_samples - 1}")
        
        for object_pair in self.resources['object_pairs']:
            if names_list is not None and target_names is not None and object_pair.collection.name not in target_names:
                # If generating by names, update sample_count for the skipped object but don't render it
                sample_count += len(self.resources['cameras']) * self.get_uvs_count(object_pair.collection.name)
                continue

            rendered_sample_count = self._process_object_pair(
                object_pair, sample_count, rendered_sample_count,
                target_indices if names_list is None else None, obj_mask_output, obj_id_mask, depth_output
            )
            
            sample_count += len(self.resources['cameras']) * self.get_uvs_count(object_pair.collection.name)
        
        self.logger.info(f"Dataset generation complete. Generated {sample_count} samples in {self.config.output_root}")
    
    def _process_object_pair(self, object_pair, sample_count: int, rendered_sample_count: int,
                           target_indices: Optional[Set[int]], obj_mask_output, obj_id_mask, depth_output) -> int:
        """Process a single object pair across all camera views"""
        main_obj = object_pair.main_obj
        cover_obj = object_pair.cover_obj
        obj_class_name = object_pair.name
        collection = object_pair.collection
        collection_name = collection.name
        
        layer_collection = find_layer_collection(bpy.context.view_layer.layer_collection, collection_name)
        layer_collection.exclude = False
        
        self._apply_transformations(main_obj, cover_obj)
        
        for view_idx in range(len(self.resources['cameras'])):
            if self.is_planar_object(collection_name):
                uv_mappings_to_generate = [UVMappingType.PLANAR]
                best_uv_mapping_str = UVMappingType.PLANAR.name
            else:
                best_uv_mapping_str = self.best_uv_mappings[collection_name]
                uv_mappings_to_generate = [uv_type for uv_type in UVMappingType if uv_type not in (UVMappingType.ORIGINAL, UVMappingType.PLANAR)]

                if best_uv_mapping_str == UVMappingType.ORIGINAL.name:
                    uv_mappings_to_generate.append(UVMappingType.ORIGINAL)

            for uv_map_idx, uv_mapping_type in enumerate(uv_mappings_to_generate):
                current_sample_idx = sample_count + view_idx * self.get_uvs_count(object_pair.collection.name) + uv_map_idx

                if self.config.filter_best_uv_map and best_uv_mapping_str != uv_mapping_type.name:
                    continue

                if target_indices is not None and current_sample_idx not in target_indices:
                    continue

                if self.config.continue_generation and is_sample_complete(current_sample_idx, self.config.output_root, self.config.coating_materials_count):
                    continue

                self._generate_single_sample(
                    main_obj, cover_obj, obj_class_name, collection_name, uv_mapping_type, best_uv_mapping_str, view_idx,
                    current_sample_idx, obj_mask_output, obj_id_mask, depth_output
                )
            
                rendered_sample_count += 1
                if target_indices is not None:
                    self.logger.info(f"Generated sample {rendered_sample_count}/{len(target_indices)}")
                else:
                    self.logger.info(f"Generated sample {rendered_sample_count} for {collection_name}")
        
        layer_collection.exclude = True
        
        return rendered_sample_count
    
    def _apply_transformations(self, main_obj, cover_obj) -> None:
        """Apply transformations to objects"""
        # Clear all selections first
        bpy.ops.object.select_all(action='DESELECT')
        
        # Apply transformations to main object
        bpy.context.view_layer.objects.active = main_obj
        main_obj.select_set(True)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        main_obj.select_set(False)
        
        # Apply transformations to cover object
        bpy.context.view_layer.objects.active = cover_obj
        cover_obj.select_set(True)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        cover_obj.select_set(False)
        
        # Clear selections again
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = None

        bpy.context.view_layer.update()
    
    def _generate_single_sample(self, main_obj, cover_obj, obj_class_name: str, collection_name: str, uv_mapping_type: UVMappingType,
                              best_uv_mapping_method: str, view_idx: int, sample_idx: int,
                              obj_mask_output, obj_id_mask, depth_output) -> None:
        """Generate a single sample with all coating variations"""
        camera = self.resources['cameras'][view_idx]
        
        # Setup scene
        scene_config = setup_scene_for_sample(
            self.resources['hdri_pool'], self.resources['floors'], camera
        )
        
        # Show main object, hide cover initially
        set_object_visibility_recursive(main_obj, True)
        set_object_visibility_recursive(cover_obj, False)

        transform_state = self._apply_random_transforms(main_obj, cover_obj)

        # Apply auto-framing if enabled, or direct quad framing
        if self.is_planar_object(collection_name):
            camera_state = self._apply_quad_framing(camera, main_obj)
        else:
            camera_state = self._apply_auto_framing(camera, main_obj)
        
        # Configure lights for the scene
        lights_state = configure_lights_for_scene(
            self.logger, camera, main_obj, self.resources['lights'], self.config
        )
        
        # Setup object IDs and output paths
        main_obj.pass_index = 1
        cover_obj.pass_index = 1
        obj_id_mask.index = 1
        sample_dir = set_output_paths(sample_idx, self.config.output_root, obj_mask_output, depth_output)
        
        # Store original material properties and apply augmentation
        original_material_props = store_original_material_properties(main_obj)
        apply_material_augmentation(original_material_props)
        
        # Generate prompts structure
        prompts_json = {}
        
        # Render without coating (original)
        render_image(os.path.join(sample_dir, "coating_0"))
        
        # Render additional passes
        render_normal_pass(main_obj, cover_obj, sample_dir)
        render_depth_pass(main_obj, cover_obj, sample_dir)
        
        # Create base caption and prompt
        base_caption = f"photo of a {obj_class_name} on {scene_config['floor_name']} {scene_config['environment_name']}"
        prompts_json["coating_0"] = {
            "caption": base_caption,
            "instruction": f"remove all coating from the {obj_class_name}",
            "coating": "none",
            "thickness": 0,
            "mask": "none",
            "material_properties": {}
        }
        
        # Organize initial output files
        find_and_rename_passes(sample_dir)
        organize_output_files(sample_dir)
        
        # Generate coating variations
        self._generate_coating_variations(
            main_obj, cover_obj, obj_class_name, uv_mapping_type, best_uv_mapping_method, sample_dir,
            scene_config, prompts_json
        )
        
        # Save prompts and restore states
        save_prompts_json(sample_dir, prompts_json)
        restore_material_properties(original_material_props)

        self._restore_transforms(main_obj, cover_obj, transform_state)
        
        if camera_state:
            restore_camera_state(camera, camera_state)
        
        # Restore lights to their original positions
        restore_lights(self.resources['lights'], lights_state)
        
        if scene_config['floor_obj']:
            scene_config['floor_obj'].hide_render = True
    
    def _apply_random_transforms(self, main_obj, cover_obj) -> Dict[str, Any]:
        """Apply random transformations to objects"""
        original_state = {
            'main_location': main_obj.location.copy(),
            'main_rotation': main_obj.rotation_euler.copy(),
            'cover_location': cover_obj.location.copy(),
            'cover_rotation': cover_obj.rotation_euler.copy()
        }
        
        random_offset = Vector((
            random.uniform(-self.config.translation_range, self.config.translation_range),
            random.uniform(-self.config.translation_range, self.config.translation_range),
            0
        ))
        random_rotation = Euler((0, 0, random.uniform(-self.config.rot_range, self.config.rot_range)))
        
        main_obj.location += random_offset
        main_obj.rotation_euler = random_rotation
        cover_obj.location += random_offset
        cover_obj.rotation_euler = random_rotation
        
        return original_state
    
    def _restore_transforms(self, main_obj, cover_obj, transform_state: Dict[str, Any]) -> None:
        """Restore object transformations"""
        main_obj.location = transform_state['main_location']
        main_obj.rotation_euler = transform_state['main_rotation']
        cover_obj.location = transform_state['cover_location']
        cover_obj.rotation_euler = transform_state['cover_rotation']
    
    def _apply_auto_framing(self, camera, main_obj) -> Optional[Any]:
        """Apply auto-framing if enabled"""
        if random.random() < self.config.auto_frame_probability:
            self.logger.info("Applying auto-framing")
            return frame_object_in_camera(self.logger, camera, main_obj, self.config.frame_margin)
        return None

    def _apply_quad_framing(self, camera, main_obj):
        """Apply direct framing for upright quad objects"""
        self.logger.info("Applying direct quad framing")

        original_state = store_camera_state(camera)

        track_to_constraint = None
        constraint_target = None

        for constraint in camera.constraints:
            if constraint.type == 'TRACK_TO':
                track_to_constraint = constraint
                constraint_target = constraint.target
                break

        if not track_to_constraint or not constraint_target:
            self.logger.info("Warning: Camera does not have a Track To constraint with a target")
            return original_state

        original_target_location = constraint_target.location.copy()

        # 1. Calculate the true center of the upright quad's bounding box
        bbox_corners = get_object_bounds_recursive(main_obj)
        center_x = sum([c.x for c in bbox_corners]) / 8.0
        center_y = sum([c.y for c in bbox_corners]) / 8.0
        center_z = sum([c.z for c in bbox_corners]) / 8.0
        target_center = Vector((center_x, center_y, center_z))

        # Move the camera target empty to the exact center height of the quad
        constraint_target.location = target_center
        bpy.context.view_layer.update()

        # 2. Determine the face normal.
        face_normal = Vector((1.0, 0.0, 0.0))

        # Ensure we're on the side of the quad that the camera is currently roughly facing
        to_camera = camera.location - target_center
        if to_camera.dot(face_normal) < 0:
            face_normal = -face_normal

        # 3. Position camera directly in front of the quad face at center height
        camera.location = target_center + face_normal * 5.0
        bpy.context.view_layer.update()

        # Let frame_object_in_camera calculate the exact bounding box distance
        enhanced_state = frame_object_in_camera(self.logger, camera, main_obj, self.config.frame_margin)

        # Restore the original properties so we can revert properly after render
        enhanced_state.location = original_state.location
        enhanced_state.rotation_euler = original_state.rotation_euler
        enhanced_state.rotation_quaternion = original_state.rotation_quaternion
        enhanced_state.rotation_mode = original_state.rotation_mode
        enhanced_state.scale = original_state.scale
        enhanced_state.lens = original_state.lens
        enhanced_state.sensor_width = original_state.sensor_width
        enhanced_state.sensor_height = original_state.sensor_height
        enhanced_state.clip_start = original_state.clip_start
        enhanced_state.clip_end = original_state.clip_end
        enhanced_state.target_location = original_target_location
        enhanced_state.constraint_target = constraint_target

        return enhanced_state
    
    def _generate_coating_variations(self, main_obj, cover_obj, obj_class_name: str, uv_mapping_type: UVMappingType,
                                     best_uv_mapping_method: str, sample_dir: str, scene_config: Dict[str, Any],
                                     prompts_json: Dict[str, Any]) -> None:
        """Generate all coating variations for the current sample"""
        # Show cover object for coating rendering
        set_object_visibility_recursive(cover_obj, True)

        # Prepare coating masks
        coating_mask = prepare_coating_masks(
            self.resources['coating_masks'], self.config.full_mask_probability
        )

        coating_albedo_uv_name = get_coating_albedo_uv_map(self.logger, cover_obj, uv_mapping_type)

        # Generate coating combinations
        coating_combinations = []
        for i in range(self.config.coating_materials_count):
            self.config.index = i
            coating_material, albedo_texture_name, is_uniform_color, is_transparent, thickness_value = (
                create_random_textured_or_uniform_material(self.resources['albedo_textures'], self.config))

            coating_combinations.append({
                'material': coating_material,
                'mask': coating_mask,
                'thickness': thickness_value,
                'albedo_texture_name': albedo_texture_name,
                'is_uniform_color': is_uniform_color,
                'is_transparent': is_transparent
            })
        
        # Render each coating combination
        for i, combo in enumerate(coating_combinations):
            coating_index = i + 1
            
            coating_material = combo['material']
            coating_mask = combo['mask']
            thickness_value = combo['thickness']
            is_uniform_color = combo['is_uniform_color']
            is_transparent = combo['is_transparent']

            material_properties = get_material_properties(coating_material, is_transparent, uv_mapping_type,
                                                          best_uv_mapping_method)

            # Apply coating to cover object
            original_materials, temp_materials, mask_node = apply_coating_material_to_cover(
                cover_obj, main_obj, coating_material, coating_mask, thickness_value, is_transparent,
                coating_albedo_uv_name
            )
            
            # Render with coating
            render_image(os.path.join(sample_dir, f"coating_{coating_index}.png"))
            render_coating_mask(cover_obj, coating_mask, sample_dir)
            
            # Render albedo-only version (skip for any uniform colored materials)
            if not is_uniform_color:
                render_albedo_pass(cover_obj, coating_material, coating_albedo_uv_name, sample_dir, coating_index)
            
            # Generate caption and instruction
            coating_name = get_material_name(coating_material)
            mask_file_name = coating_mask.name

            coating_caption = f"photo of a {obj_class_name} with {coating_name} coating on {scene_config['floor_name']} {scene_config['environment_name']}"
            instruction = f"apply {coating_name} coating to the {obj_class_name} with the given mask"
            
            prompts_json[str(coating_index)] = {
                "caption": coating_caption,
                "instruction": instruction,
                "coating": coating_name,
                "thickness": round(thickness_value, 3),
                "mask": mask_file_name,
                "albedo_texture": combo['albedo_texture_name'] if not is_uniform_color else "",
                "material_properties": material_properties
            }
            
            # Restore materials and cleanup
            restore_cover_material(cover_obj, original_materials, temp_materials)
            find_and_rename_passes(sample_dir)
            cleanup_temp_files(sample_dir)


def generate_dataset(logger: logging.Logger, config: Optional[DatasetConfig] = None,
                    initial_index: int = 0, num_samples: int = 1575,
                    indices_list: Optional[List[int]] = None,
                    names_list: Optional[List[str]] = None) -> None:
    """Main entry point for dataset generation"""
    if config is None:
        config = get_default_config()
    
    orchestrator = DatasetOrchestrator(config, logger)
    orchestrator.generate_dataset(initial_index, num_samples, indices_list, names_list)