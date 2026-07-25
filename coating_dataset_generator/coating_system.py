from typing import List, Tuple, Any

import bpy
from mathutils import Vector

from .material_generator import UVMappingType
from .geometry_utils import get_average_bounds_size


def get_coating_albedo_uv_map(logger, obj, uv_mapping_type: UVMappingType) -> str:
    """Select an existing UV map based on the given type."""

    if uv_mapping_type == UVMappingType.ORIGINAL:
        uv_map_name = "Original UV Map"
    elif uv_mapping_type == UVMappingType.CUBIC or uv_mapping_type == UVMappingType.PLANAR:
        uv_map_name = "Albedo Cubic UV map"
    elif uv_mapping_type == UVMappingType.SPHERICAL:
        uv_map_name = "Albedo Spherical UV map"
    else:
        raise Exception(f"Unknown UV mapping type: {uv_mapping_type}")

    if uv_map_name not in obj.data.uv_layers:
        raise Exception(f"UV map '{uv_map_name}' not found on object '{obj.name}'. Defaulting to the first available UV map.")

    logger.info(f"Using existing UV map: '{uv_map_name}' for {obj.name}")
    return uv_map_name


def create_mask_node_group() -> bpy.types.NodeTree:
    """Create a node group for the mask processing"""
    if "CoatingMask" in bpy.data.node_groups:
        return bpy.data.node_groups["CoatingMask"]
    
    mask_group = bpy.data.node_groups.new(name="CoatingMask", type='ShaderNodeTree')
    
    group_inputs = mask_group.nodes.new('NodeGroupInput')
    group_outputs = mask_group.nodes.new('NodeGroupOutput')
    
    mask_group.interface.new_socket('Image', in_out='INPUT', socket_type='NodeSocketColor')
    mask_group.interface.new_socket('Alpha', in_out='INPUT', socket_type='NodeSocketFloat')
    
    mask_group.interface.new_socket('Fac', in_out='OUTPUT', socket_type='NodeSocketFloat')
    
    color_separate = mask_group.nodes.new('ShaderNodeSeparateColor')

    group_inputs.location = (-400, 0)
    color_separate.location = (-200, 0)
    group_outputs.location = (200, 0)
    
    mask_group.links.new(group_inputs.outputs['Image'], color_separate.inputs['Color'])
    mask_group.links.new(color_separate.outputs['Red'], group_outputs.inputs['Fac'])
    
    return mask_group


def _add_mask_nodes_subtree(nodes, links, coating_mask):
    """Helper function to create and connect mask nodes using generated coordinates."""
    tex_coord_node = nodes.new('ShaderNodeTexCoord')
    tex_coord_node.location = (-1000, 300)
    
    coating_mask_node = nodes.new('ShaderNodeTexImage')
    coating_mask_node.image = coating_mask
    coating_mask_node.extension = 'CLIP'
    coating_mask_node.projection = 'SPHERE' # BOX did not work well, there were some odd looking dark spots artifacts.
    coating_mask_node.location = (-800, 300)
    
    links.new(tex_coord_node.outputs['Generated'], coating_mask_node.inputs['Vector'])
    
    mask_group_ref = create_mask_node_group()
    mask_node = nodes.new('ShaderNodeGroup')
    mask_node.node_tree = mask_group_ref
    mask_node.location = (-500, 300)
    
    links.new(coating_mask_node.outputs['Color'], mask_node.inputs['Image'])
    links.new(coating_mask_node.outputs['Alpha'], mask_node.inputs['Alpha'])
    
    return mask_node


def apply_coating_material_to_cover(cover_obj, main_obj, coating_material,
                                    coating_mask, thickness_value: float, is_transparent: bool,
                                    coating_albedo_uv_name: str) -> Tuple[List, List, Any]:
    """Apply coating material with mask to cover object"""
    original_materials = []
    for slot in cover_obj.material_slots:
        original_materials.append(slot.material)
    
    original_normal_textures = []
    for slot in cover_obj.material_slots:
        normal_texture = None
        if slot.material and slot.material.use_nodes:
            for node in slot.material.node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    normal_input = node.inputs.get('Normal')
                    if normal_input and normal_input.is_linked:
                        linked_node = normal_input.links[0].from_node
                        if linked_node.type == 'NORMAL_MAP':
                            color_input = linked_node.inputs.get('Color')
                            if color_input and color_input.is_linked:
                                texture_node = color_input.links[0].from_node
                                if texture_node.type == 'TEX_IMAGE' and texture_node.image:
                                    normal_texture = texture_node.image
                                    break
        original_normal_textures.append(normal_texture)
    
    coating_materials = []
    mask_node = None
    
    cover_avg_bounds = get_average_bounds_size(cover_obj)
    main_avg_bounds = get_average_bounds_size(main_obj)
    obj_delta = cover_avg_bounds - main_avg_bounds
    
    reference_delta = 0.0009 # This delta was determined manually to work well with the base density value.
    density_to_scale_factor = reference_delta / max(obj_delta, 1e-8)

    print(f"density_to_scale_factor = {density_to_scale_factor}")
    
    for i, original_material in enumerate(original_materials):
        coating_mat_copy = coating_material.copy()
        coating_mat_copy.name = f"temp_coating_{coating_material.name}_{i}"
        
        nodes = coating_mat_copy.node_tree.nodes
        links = coating_mat_copy.node_tree.links
        
        output_node = None
        coating_bsdf = None
        albedo_texture_node = None
        for node in list(nodes):
            if node.type in 'OUTPUT_MATERIAL':
                output_node = node
            elif node.type == 'BSDF_PRINCIPLED':
                coating_bsdf = node
            elif node.type == 'TEX_IMAGE':
                albedo_texture_node = node
            else:
                nodes.remove(node)
        
        if not output_node:
            raise Exception("output node is missing from the coating material!")
        if not coating_bsdf:
            raise Exception("BSDF node not found for coating material!")

        mask_node = _add_mask_nodes_subtree(nodes, links, coating_mask)
        
        original_uv_node = nodes.new('ShaderNodeUVMap')
        if len(cover_obj.data.uv_layers) > 0:
            original_uv_node.uv_map = cover_obj.data.uv_layers[0].name
        original_uv_node.location = (-1000, -100)
        
        mix_shader = nodes.new('ShaderNodeMixShader')
        mix_shader.location = (-100, 0)
        
        transparent_bsdf = nodes.new('ShaderNodeBsdfTransparent')
        transparent_bsdf.location = (-300, 100)
        
        ao_node = nodes.new('ShaderNodeAmbientOcclusion')
        ao_node.samples = 16
        ao_node.inputs['Distance'].default_value = 1.0
        ao_node.location = (200, 200)
        
        ao_mix_shader = nodes.new('ShaderNodeMixShader')
        ao_mix_shader.location = (400, 0)
        
        normal_map = nodes.new('ShaderNodeNormalMap')
        normal_map.location = (100, -300)
        
        normal_texture_node = nodes.new('ShaderNodeTexImage')
        normal_texture_node.image = original_normal_textures[i] if i < len(original_normal_textures) else None
        normal_texture_node.location = (-200, -300)
        
        if original_normal_textures[i] if i < len(original_normal_textures) else None:
            links.new(original_uv_node.outputs['UV'], normal_texture_node.inputs['Vector'])
            links.new(normal_texture_node.outputs['Color'], normal_map.inputs['Color'])

        normal_reduction_node = nodes.new('ShaderNodeGroup')
        normal_reduction_node.node_tree = bpy.data.node_groups["Normal Reduction"]
        normal_reduction_node.location = (-100, -400)
        normal_reduction_node.inputs['Thickness'].default_value = thickness_value
        links.new(normal_reduction_node.outputs['Reduced Normal'], normal_map.inputs['Strength'])
        
        output_node.location = (600, 0)

        if albedo_texture_node:
            # Use the specified UV map for the albedo texture
            coating_albedo_uv_map_node = nodes.new(type='ShaderNodeUVMap')
            coating_albedo_uv_map_node.uv_map = coating_albedo_uv_name
            coating_albedo_uv_map_node.location = (-300, 0)
            links.new(coating_albedo_uv_map_node.outputs['UV'], albedo_texture_node.inputs['Vector'])

        links.new(mask_node.outputs['Fac'], mix_shader.inputs['Fac'])
        links.new(transparent_bsdf.outputs['BSDF'], mix_shader.inputs[1])

        if is_transparent:
            transmissive_coat_node = nodes.new('ShaderNodeGroup')
            transmissive_coat_node.node_tree = bpy.data.node_groups["TransmissiveCoat"]
            transmissive_coat_node.location = (100, -500)
            transmissive_coat_node.inputs['Thickness'].default_value = thickness_value
            transmissive_coat_node.inputs['Density To Scale Factor'].default_value = density_to_scale_factor

            links.new(mask_node.outputs['Fac'], transmissive_coat_node.inputs['Generated Tex Coord Mask'])

            if albedo_texture_node:
                links.new(albedo_texture_node.outputs['Color'], transmissive_coat_node.inputs['Coat Albedo sRGB'])
            else:
                transmissive_coat_node.inputs['Coat Albedo sRGB'].default_value = coating_bsdf.inputs[
                    'Base Color'].default_value

            links.new(normal_map.outputs['Normal'], transmissive_coat_node.inputs['Normal'])

            links.new(transmissive_coat_node.outputs['BSDF'], mix_shader.inputs[2])
            links.new(transmissive_coat_node.outputs['Volume'], output_node.inputs['Volume'])
        else:
            opaque_albedo_blend_node = nodes.new('ShaderNodeGroup')
            opaque_albedo_blend_node.node_tree = bpy.data.node_groups["Opaque Albedo Blend"]
            opaque_albedo_blend_node.location = (100, -500)
            opaque_albedo_blend_node.inputs['Thickness'].default_value = thickness_value

            original_principled_bsdf = original_material.node_tree.nodes.get('Principled BSDF')
            if original_principled_bsdf:
                base_color_input = original_principled_bsdf.inputs['Base Color']
                if base_color_input.is_linked:
                    original_albedo_texture_node = nodes.new('ShaderNodeTexImage')
                    original_albedo_texture_node.image = base_color_input.links[0].from_node.image
                    original_albedo_texture_node.location = (-200, -500)
                    links.new(original_uv_node.outputs['UV'], original_albedo_texture_node.inputs['Vector'])
                    links.new(original_albedo_texture_node.outputs['Color'], opaque_albedo_blend_node.inputs['Original Albedo sRGB'])
                else:
                    opaque_albedo_blend_node.inputs['Original Albedo sRGB'].default_value = base_color_input.default_value
            
            if albedo_texture_node:
                links.new(albedo_texture_node.outputs['Color'], opaque_albedo_blend_node.inputs['Coat Albedo sRGB'])
            else:
                opaque_albedo_blend_node.inputs['Coat Albedo sRGB'].default_value = coating_bsdf.inputs['Base Color'].default_value

            links.new(opaque_albedo_blend_node.outputs['Blended Albedo sRGB'], coating_bsdf.inputs['Base Color'])
            
            links.new(ao_node.outputs['Color'], ao_mix_shader.inputs['Fac'])
            links.new(coating_bsdf.outputs['BSDF'], ao_mix_shader.inputs[2])
            links.new(ao_mix_shader.outputs['Shader'], mix_shader.inputs[2])
            
            links.new(normal_map.outputs['Normal'], coating_bsdf.inputs['Normal'])

        links.new(mix_shader.outputs['Shader'], output_node.inputs['Surface'])
        
        coating_materials.append(coating_mat_copy)
    
    for i, slot in enumerate(cover_obj.material_slots):
        if i < len(coating_materials):
            slot.material = coating_materials[i]
    
    if not cover_obj.material_slots and coating_materials:
        for coating_mat in coating_materials:
            cover_obj.data.materials.append(coating_mat)
    
    return original_materials, coating_materials, mask_node


def restore_cover_material(cover_obj, original_materials: List, temp_materials: List) -> None:
    """Restore original materials to cover object and clean up temp materials"""
    for i, slot in enumerate(cover_obj.material_slots):
        if i < len(original_materials) and original_materials[i]:
            slot.material = original_materials[i]
    
    if temp_materials:
        for temp_material in temp_materials:
            if temp_material and temp_material.users == 0:
                bpy.data.materials.remove(temp_material)


def create_mask_emission_material(coating_mask) -> bpy.types.Material:
    """Create a material that emits the coating mask pattern using the specified UV map"""
    mask_material = bpy.data.materials.new(name="temp_mask_emitter")
    mask_material.use_nodes = True
    
    nodes = mask_material.node_tree.nodes
    links = mask_material.node_tree.links
    nodes.clear()
    
    output_node = nodes.new('ShaderNodeOutputMaterial')
    output_node.location = (200, 0)
    
    mask_node = _add_mask_nodes_subtree(nodes, links, coating_mask)
    
    emission_node = nodes.new('ShaderNodeEmission')
    emission_node.location = (0, 0)

    links.new(mask_node.outputs['Fac'], emission_node.inputs['Strength'])
    links.new(emission_node.outputs['Emission'], output_node.inputs['Surface'])
    
    return mask_material


def create_albedo_only_material(coating_material, coating_albedo_uv_name: str) -> bpy.types.Material:
    """Create a material that only shows the albedo texture from coating material"""
    albedo_material = bpy.data.materials.new(name="temp_albedo_only")
    albedo_material.use_nodes = True
    
    nodes = albedo_material.node_tree.nodes
    links = albedo_material.node_tree.links
    nodes.clear()
    
    # Find albedo texture from coating material
    albedo_texture = None
    for node in coating_material.node_tree.nodes:
        if node.type == 'TEX_IMAGE':
            albedo_texture = node.image
            break
    
    if not albedo_texture:
        raise Exception("No albedo texture found in coating material")
    
    output = nodes.new('ShaderNodeOutputMaterial')
    emission = nodes.new('ShaderNodeEmission')
    texture_node = nodes.new('ShaderNodeTexImage')
    uv_node = nodes.new('ShaderNodeUVMap')
    
    uv_node.location = (-400, 0)
    texture_node.location = (-200, 0)
    emission.location = (0, 0)
    output.location = (200, 0)
    
    uv_node.uv_map = coating_albedo_uv_name
    texture_node.image = albedo_texture
    
    links.new(uv_node.outputs['UV'], texture_node.inputs['Vector'])
    links.new(texture_node.outputs['Color'], emission.inputs['Color'])
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    return albedo_material