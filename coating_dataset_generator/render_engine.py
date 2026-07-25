import bpy
import os
from typing import Tuple, Dict, Any, List
from .material_generator import create_normal_pass_material
from .coating_system import create_mask_emission_material, create_albedo_only_material


def setup_enhanced_rendering(config: 'DatasetConfig') -> Tuple[Any, Any, Any]:
    """Set up compositor for enhanced rendering with object mask and depth pass"""
    bpy.context.scene.render.engine = 'CYCLES'
    
    bpy.context.scene.render.resolution_x = config.resolution
    bpy.context.scene.render.resolution_y = config.resolution
    
    view_layer = bpy.context.view_layer
    view_layer.use_pass_object_index = True
    view_layer.use_pass_z = True
    
    bpy.context.view_layer.update()
    bpy.context.scene.frame_set(bpy.context.scene.frame_current)
    
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    
    for node in tree.nodes:
        tree.nodes.remove(node)
    
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    render_layers.location = (-600, 0)
    
    obj_id_mask = tree.nodes.new('CompositorNodeIDMask')
    obj_id_mask.location = (-300, 200)
    obj_id_mask.index = 1
    obj_id_mask.name = "OBJ_ID_MASK"
    
    obj_mask_output = tree.nodes.new('CompositorNodeOutputFile')
    obj_mask_output.location = (200, 200)
    obj_mask_output.format.file_format = 'PNG'
    obj_mask_output.format.color_mode = 'BW'
    obj_mask_output.format.color_depth = '8'
    obj_mask_output.name = "OBJ_MASK_OUTPUT"
    
    depth_normalize = tree.nodes.new('CompositorNodeNormalize')
    depth_normalize.location = (0, -100)
    
    depth_output = tree.nodes.new('CompositorNodeOutputFile')
    depth_output.location = (200, -100)
    depth_output.format.file_format = 'PNG'
    depth_output.format.color_mode = 'BW'
    depth_output.format.color_depth = '8'
    depth_output.name = "DEPTH_OUTPUT"
    
    render_output = tree.nodes.new('CompositorNodeComposite')
    render_output.location = (400, 400)
    
    tree.links.new(render_layers.outputs['IndexOB'], obj_id_mask.inputs[0])
    tree.links.new(obj_id_mask.outputs[0], obj_mask_output.inputs[0])
    
    tree.links.new(render_layers.outputs['Depth'], depth_normalize.inputs[0])
    tree.links.new(depth_normalize.outputs[0], depth_output.inputs[0])
    
    tree.links.new(render_layers.outputs['Image'], render_output.inputs[0])
    
    return obj_mask_output, obj_id_mask, depth_output


def store_render_settings() -> Dict[str, Any]:
    """Store current render settings for restoration"""
    scene = bpy.context.scene
    return {
        'samples': scene.cycles.samples,
        'max_bounces': scene.cycles.max_bounces,
        'diffuse_bounces': scene.cycles.diffuse_bounces,
        'glossy_bounces': scene.cycles.glossy_bounces,
        'transmission_bounces': scene.cycles.transmission_bounces,
        'volume_bounces': scene.cycles.volume_bounces,
        'transparent_max_bounces': scene.cycles.transparent_max_bounces,
    }


def restore_render_settings(settings: Dict[str, Any]) -> None:
    """Restore render settings"""
    scene = bpy.context.scene
    scene.cycles.samples = settings['samples']
    scene.cycles.max_bounces = settings['max_bounces']
    scene.cycles.diffuse_bounces = settings['diffuse_bounces']
    scene.cycles.glossy_bounces = settings['glossy_bounces']
    scene.cycles.transmission_bounces = settings['transmission_bounces']
    scene.cycles.volume_bounces = settings['volume_bounces']
    scene.cycles.transparent_max_bounces = settings['transparent_max_bounces']


def set_minimal_render_settings() -> None:
    """Set minimal render settings for fast rendering"""
    scene = bpy.context.scene
    scene.cycles.samples = 1
    scene.cycles.max_bounces = 0
    scene.cycles.diffuse_bounces = 0
    scene.cycles.glossy_bounces = 0
    scene.cycles.transmission_bounces = 0
    scene.cycles.volume_bounces = 0
    scene.cycles.transparent_max_bounces = 1


def store_visibility_states() -> Dict[str, bool]:
    """Store visibility states of all objects"""
    visibility_states = {}
    for obj in bpy.context.scene.objects:
        visibility_states[obj.name] = obj.hide_render
    return visibility_states


def restore_visibility_states(visibility_states: Dict[str, bool]) -> None:
    """Restore object visibility states"""
    for obj_name, was_hidden in visibility_states.items():
        if obj_name in bpy.context.scene.objects:
            bpy.context.scene.objects[obj_name].hide_render = was_hidden


def create_black_world() -> bpy.types.World:
    """Create a temporary black world for clean rendering"""
    temp_world = bpy.data.worlds.new("temp_black_world")
    temp_world.use_nodes = True
    temp_world.node_tree.nodes.clear()
    bg_node = temp_world.node_tree.nodes.new('ShaderNodeBackground')
    output_node = temp_world.node_tree.nodes.new('ShaderNodeOutputWorld')
    bg_node.inputs['Color'].default_value = (0, 0, 0, 1)
    bg_node.inputs['Strength'].default_value = 0.0
    temp_world.node_tree.links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])
    return temp_world


def render_normal_pass(main_obj, cover_obj, sample_dir: str) -> None:
    """Render normal pass using geometry node material with floor hidden"""
    original_main_materials = [slot.material for slot in main_obj.material_slots]
    original_cover_materials = [slot.material for slot in cover_obj.material_slots]
    
    original_visibility = store_visibility_states()
    original_settings = store_render_settings()
    
    set_minimal_render_settings()
    
    for obj in bpy.context.scene.objects:
        if obj != main_obj and obj != cover_obj and obj.type not in ['CAMERA', 'LIGHT']:
            obj.hide_render = True
    
    main_obj.hide_render = False
    cover_obj.hide_render = False
    
    original_world = bpy.context.scene.world.copy()
    temp_world = create_black_world()
    bpy.context.scene.world = temp_world
    
    main_original_material = main_obj.material_slots[0].material if main_obj.material_slots else None
    cover_original_material = cover_obj.material_slots[0].material if cover_obj.material_slots else None
    
    main_normal_material = create_normal_pass_material(main_original_material)
    cover_normal_material = create_normal_pass_material(cover_original_material)
    
    for slot in main_obj.material_slots:
        slot.material = main_normal_material
    
    for slot in cover_obj.material_slots:
        slot.material = cover_normal_material
    
    original_filepath = bpy.context.scene.render.filepath
    bpy.context.scene.render.filepath = os.path.join(sample_dir, "normal")
    bpy.ops.render.render(write_still=True)
    bpy.context.scene.render.filepath = original_filepath
    
    bpy.context.scene.world = original_world
    bpy.data.worlds.remove(temp_world)
    
    restore_visibility_states(original_visibility)
    
    for i, slot in enumerate(main_obj.material_slots):
        if i < len(original_main_materials) and original_main_materials[i]:
            slot.material = original_main_materials[i]
    
    for i, slot in enumerate(cover_obj.material_slots):
        if i < len(original_cover_materials) and original_cover_materials[i]:
            slot.material = original_cover_materials[i]
    
    if main_normal_material and main_normal_material.users == 0:
        bpy.data.materials.remove(main_normal_material)
    if cover_normal_material and cover_normal_material.users == 0:
        bpy.data.materials.remove(cover_normal_material)
    
    restore_render_settings(original_settings)


def render_depth_pass(main_obj, cover_obj, sample_dir: str) -> None:
    """Render depth pass that includes material displacement/bumpiness"""
    original_visibility = store_visibility_states()
    original_settings = store_render_settings()
    
    set_minimal_render_settings()
    
    for obj in bpy.context.scene.objects:
        if obj != main_obj and obj != cover_obj and obj.type not in ['CAMERA', 'LIGHT']:
            obj.hide_render = True
    
    main_obj.hide_render = False
    cover_obj.hide_render = False
    
    original_world = bpy.context.scene.world.copy()
    temp_world = create_black_world()
    bpy.context.scene.world = temp_world
    
    original_filepath = bpy.context.scene.render.filepath
    bpy.context.scene.render.filepath = os.path.join(sample_dir, "depth")
    bpy.ops.render.render(write_still=True)
    bpy.context.scene.render.filepath = original_filepath
    
    bpy.context.scene.world = original_world
    bpy.data.worlds.remove(temp_world)
    
    restore_visibility_states(original_visibility)
    restore_render_settings(original_settings)


def render_coating_mask(cover_obj, coating_mask, sample_dir: str) -> None:
    """Render a coating mask using emission"""
    original_materials = [slot.material for slot in cover_obj.material_slots]
    original_pass_index = cover_obj.pass_index
    
    original_settings = store_render_settings()
    set_minimal_render_settings()
    
    original_visibility = store_visibility_states()
    
    for obj in bpy.context.scene.objects:
        if obj != cover_obj and obj.type not in ['CAMERA', 'LIGHT']:
            obj.hide_render = True
    
    cover_obj.hide_render = False
    cover_obj.pass_index = 3
    
    original_world = bpy.context.scene.world.copy()
    temp_world = create_black_world()
    bpy.context.scene.world = temp_world
    
    mask_materials = []
    for i, slot in enumerate(cover_obj.material_slots):
        mask_material = create_mask_emission_material(coating_mask)
        mask_material.name = f"temp_mask_emitter_{i}"
        mask_materials.append(mask_material)
        slot.material = mask_material
    
    if not cover_obj.material_slots:
        raise Exception("No material slots!")
    
    original_filepath = bpy.context.scene.render.filepath
    bpy.context.scene.render.filepath = os.path.join(sample_dir, "coating_mask")
    bpy.ops.render.render(write_still=True)
    bpy.context.scene.render.filepath = original_filepath
    
    bpy.context.scene.world = original_world
    bpy.data.worlds.remove(temp_world)
    
    restore_visibility_states(original_visibility)
    
    for i, slot in enumerate(cover_obj.material_slots):
        if i < len(original_materials) and original_materials[i]:
            slot.material = original_materials[i]
    cover_obj.pass_index = original_pass_index
    
    for mask_material in mask_materials:
        if mask_material and mask_material.users == 0:
            bpy.data.materials.remove(mask_material)
    
    restore_render_settings(original_settings)


def render_albedo_pass(cover_obj, coating_material, coating_albedo_uv_name: str, sample_dir: str, coating_index: int) -> None:
    """Render albedo-only pass showing just the texture projection"""
    original_materials = [slot.material for slot in cover_obj.material_slots]
    original_pass_index = cover_obj.pass_index
    
    original_settings = store_render_settings()
    set_minimal_render_settings()
    
    original_visibility = store_visibility_states()
    
    for obj in bpy.context.scene.objects:
        if obj != cover_obj and obj.type not in ['CAMERA', 'LIGHT']:
            obj.hide_render = True
    
    cover_obj.hide_render = False
    cover_obj.pass_index = 4
    
    original_world = bpy.context.scene.world.copy()
    temp_world = create_black_world()
    bpy.context.scene.world = temp_world
    
    albedo_material = create_albedo_only_material(coating_material, coating_albedo_uv_name)
    
    for slot in cover_obj.material_slots:
        slot.material = albedo_material
    
    if not cover_obj.material_slots:
        raise Exception("No material slots!")
    
    original_filepath = bpy.context.scene.render.filepath
    bpy.context.scene.render.filepath = os.path.join(sample_dir, f"coating_{coating_index}_albedo")
    bpy.ops.render.render(write_still=True)
    bpy.context.scene.render.filepath = original_filepath
    
    bpy.context.scene.world = original_world
    bpy.data.worlds.remove(temp_world)
    
    restore_visibility_states(original_visibility)
    
    for i, slot in enumerate(cover_obj.material_slots):
        if i < len(original_materials) and original_materials[i]:
            slot.material = original_materials[i]
    cover_obj.pass_index = original_pass_index
    
    if albedo_material and albedo_material.users == 0:
        bpy.data.materials.remove(albedo_material)
    
    restore_render_settings(original_settings)


def render_image(filepath: str) -> None:
    """Render current scene to specified filepath"""
    original_filepath = bpy.context.scene.render.filepath
    bpy.context.scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)
    bpy.context.scene.render.filepath = original_filepath