import bpy
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from . import DatasetConfig
from .file_manager import get_object_class_name, get_floor_name
from .material_generator import load_albedo_textures
from .light_operations import validate_lights


@dataclass
class HDRIInfo:
    """Information about an HDRI environment"""
    world: bpy.types.World
    name: str
    node: bpy.types.Node


@dataclass
class ObjectPair:
    """Pair of main object and its cover object"""
    collection: bpy.types.Collection
    main_obj: bpy.types.Object
    cover_obj: bpy.types.Object
    name: str


def get_coating_masks() -> List[bpy.types.Image]:
    """Get all coating mask textures with 'coating_mask_' prefix"""
    coating_masks = []
    for image in bpy.data.images:
        if image.name.startswith("coating_mask_"):
            coating_masks.append(image)
    
    if not coating_masks:
        raise Exception("No coating mask textures found with 'coating_mask_' prefix")
    
    return coating_masks


def get_hdri_pool() -> List[HDRIInfo]:
    """Get all HDRI environments from the world nodes"""
    hdri_pool = []
    
    for world in bpy.data.worlds:
        if world.use_nodes:
            for node in world.node_tree.nodes:
                if node.type == 'TEX_ENVIRONMENT':
                    if node.image:
                        hdri_info = HDRIInfo(
                            world=world,
                            name=node.image.name.split(".")[0],
                            node=node
                        )
                        hdri_pool.append(hdri_info)
    
    if not hdri_pool:
        raise Exception("No HDRIs found!")
    
    return hdri_pool


def set_active_hdri(hdri_info: HDRIInfo) -> str:
    """Set the active HDRI world environment"""
    bpy.context.scene.world = hdri_info.world
    
    background_node = None
    
    for node in hdri_info.world.node_tree.nodes:
        if node.type == 'BACKGROUND':
            background_node = node
            break
    
    if hdri_info.node:
        hdri_info.world.node_tree.links.new(hdri_info.node.outputs['Color'], background_node.inputs['Color'])
    
    background_node.inputs['Strength'].default_value = random.uniform(0.3, 0.7)
    
    return hdri_info.name


def _find_layer_collection_recursive(layer_collection, collection_name: str):
    """Recursively find layer collection by name"""
    if layer_collection.collection.name == collection_name:
        return layer_collection
    for child in layer_collection.children:
        result = _find_layer_collection_recursive(child, collection_name)
        if result:
            return result
    return None


def find_layer_collection(layer_collection, collection_name: str):
    """Find layer collection by name, raise exception if not found"""
    result = _find_layer_collection_recursive(layer_collection, collection_name)
    if result is None:
        raise Exception(f"No layer collection found with name '{collection_name}'")
    return result


def get_object_pairs_from_main_objects(logger) -> List[ObjectPair]:
    """Get pairs of main objects and their corresponding cover objects"""
    main_objects_collection = bpy.data.collections.get("Main Objects")
    if not main_objects_collection:
        return []
    
    object_pairs = []
    
    for collection in main_objects_collection.children:
        main_obj = None
        cover_obj = None
        
        for obj in collection.objects:
            if obj.name.startswith("cover_"):
                cover_obj = obj
            else:
                main_obj = obj
        
        if main_obj and cover_obj:
            object_pairs.append(ObjectPair(
                collection=collection,
                main_obj=main_obj,
                cover_obj=cover_obj,
                name=get_object_class_name(main_obj)
            ))
        else:
            logger.info(f"Warning: Collection {collection.name} missing main or cover object")
    
    return object_pairs


def get_scene_resources(logger, config: DatasetConfig) -> Dict[str, Any]:
    """Get all scene resources needed for dataset generation"""
    coating_masks = get_coating_masks()
    hdri_pool = get_hdri_pool()
    object_pairs = get_object_pairs_from_main_objects(logger)
    albedo_textures = load_albedo_textures(config)
    lights = validate_lights()
    
    floors_collection = bpy.data.collections.get("Floors")
    cameras_collection = bpy.data.collections.get("Cameras")
    
    if not floors_collection or not cameras_collection:
        raise Exception("Required collections (Floors, Cameras) not found!")
    
    if not object_pairs:
        raise Exception("No object pairs found in Main Objects collection!")
    
    cameras = [obj for obj in cameras_collection.objects if obj.type == 'CAMERA']
    floors = [obj for obj in floors_collection.objects if obj.type == 'MESH']
    
    return {
        'coating_masks': coating_masks,
        'hdri_pool': hdri_pool,
        'object_pairs': object_pairs,
        'cameras': cameras,
        'floors': floors,
        'albedo_textures': albedo_textures,
        'lights': lights
    }


def hide_all_objects() -> None:
    """Hide all objects except cameras and lights"""
    for obj in bpy.context.scene.objects:
        if obj.type not in ['CAMERA', 'LIGHT']:
            obj.hide_render = True


def set_object_visibility_recursive(obj, visible: bool) -> None:
    """Set visibility of an object and all its children recursively"""
    obj.hide_render = not visible
    for child in obj.children:
        set_object_visibility_recursive(child, visible)


def setup_scene_for_sample(hdri_pool: List[HDRIInfo], floors: List[bpy.types.Object], 
                          camera: bpy.types.Object) -> Dict[str, Any]:
    """Setup scene configuration for a single sample"""
    bpy.context.scene.camera = camera
    
    hide_all_objects()
    
    hdri_info = random.choice(hdri_pool)
    environment_name = set_active_hdri(hdri_info)
    
    floor_obj = random.choice(floors)
    floor_name = get_floor_name(floor_obj)
    floor_obj.hide_render = False
    floor_obj.pass_index = 2
    
    return {
        'environment_name': environment_name,
        'floor_obj': floor_obj,
        'floor_name': floor_name
    }


def prepare_coating_masks(coating_masks: List[bpy.types.Image], full_mask_probability: float) -> Dict[str, Any]:
    """Prepare coating mask selection"""
    full_mask = None
    other_masks = []
    
    for mask in coating_masks:
        if "coating_mask_full" in mask.name.lower():
            full_mask = mask
        else:
            other_masks.append(mask)
    
    if full_mask is None:
        raise Exception("full_mask not found!")
    
    if random.random() < full_mask_probability:
        coating_mask = full_mask
    else:
        coating_mask = random.choice(other_masks)
    
    return coating_mask