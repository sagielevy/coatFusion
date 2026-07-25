import bpy
import random
import colorsys
import csv
from pathlib import Path
from typing import Dict, Any, Tuple, List
from coating_dataset_generator import DatasetConfig
from enum import Enum


class UVMappingType(Enum):
    ORIGINAL = "Original"
    CUBIC = "Cubic"
    SPHERICAL = "Spherical"
    PLANAR = "Planar"


def load_best_uv_mappings(file_path: Path) -> Dict[str, str]:
    """Load the best UV mappings from the CSV file."""
    uv_mappings = {}
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 2:
                uv_mappings[row[0]] = row[1]
    return uv_mappings


def load_albedo_textures(config: DatasetConfig) -> List[bpy.types.Image]:
    """Load all albedo textures from material_albedos once at startup"""
    albedo_textures = []
    if config.benchmark_mode:
        # albedo_dir = Path(__file__).parent / "Benchmarks/material_albedos"
        albedo_dir = Path(__file__).parent / "Benchmarks/material_albedos_userstudy"
    else:
        albedo_dir = Path(__file__).parent / "material_albedos"

    files = (p for p in albedo_dir.glob("*") if p.suffix in {".png", ".jpg", ".jpeg"})

    for img_path in files:
        img = bpy.data.images.load(str(img_path))
        albedo_textures.append(img)
    
    if not albedo_textures:
        raise Exception(f"No albedo textures found in {albedo_dir}")
    
    return albedo_textures

def _generate_random_material_properties(config) -> Tuple[float, float, float, str, Tuple[float, float, float, float], float]:
    """Generate random material properties and return (transmission, alpha, metallic, roughness, material_name, base_color)"""
    transmission = random.random() < config.transmission_probability

    if transmission == 1.0:
        metallic = 0.0
        roughness = 0.0

        # Generate colors for transmissive materials with HSL to avoid choosing colors that are too dark.
        h = random.random()
        s = random.uniform(0.6, 1.0)
        l = random.uniform(0.7, 1.0)
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        base_color = (r, g, b, 1.0)
    else:
        metallic = random.choice([0.0, 1.0])
        roughness = random.random()
        base_color = (random.random(), random.random(), random.random(), 1.0)

    name_parts = []
    
    if roughness > 0.8:
        name_parts.append("matte")
    else:
        name_parts.append("glossy")
    
    if metallic == 1.0:
        name_parts.append("metallic")
    
    if transmission > 0.0:
        base_name = random.choice(["liquid", "varnish"])
    else:
        base_name = ""
    
    if transmission == 0.0:
        if base_name:
            base_name = "paint"
        else:
            base_name = "paint"
    
    if base_name == "liquid":
        material_name = base_name
    else:
        name_parts.append(base_name)
        material_name = " ".join(name_parts)

    # 20% chance to have a binary thickness, even if binary_thickness = False
    if transmission == 0.0:
        thickness_value = random.uniform(0.7, 1)
    elif config.binary_thickness or random.random() < 0.2:
        thickness_value = random.choice([0.0, 1.0])
    else:
        thickness_value = random.uniform(0, 1)

    return transmission, metallic, roughness, material_name, base_color, thickness_value


def get_material_properties(material: bpy.types.Material, is_transparent: bool, uv_mapping: UVMappingType,
                            best_uv_mapping_method: str) -> Dict[str, float]:
    """Extract material properties from a Principled BSDF material"""
    principled_node = None
    for node in material.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            principled_node = node
            break
    
    if not principled_node:
        raise Exception("Not principled BSDF!")
    
    base_color = principled_node.inputs['Base Color'].default_value
    properties = {
        'base_color_r': round(base_color[0], 3),
        'base_color_g': round(base_color[1], 3),
        'base_color_b': round(base_color[2], 3),
        'metallic': round(principled_node.inputs['Metallic'].default_value, 3),
        'roughness': round(principled_node.inputs['Roughness'].default_value, 3),
        'transmission_weight': 1.0 if is_transparent else 0.0,
        'uv_mapping': uv_mapping.name,
        'best_mapping_method': best_uv_mapping_method
    }
    
    return properties


def store_original_material_properties(obj) -> Dict[str, Dict[str, Any]]:
    """Store original material properties for restoration"""
    original_properties = {}
    
    def process_object(obj):
        for slot in obj.material_slots:
            if slot.material and slot.material.use_nodes:
                mat = slot.material
                if mat.name not in original_properties:
                    nodes = mat.node_tree.nodes
                    
                    principled_node = None
                    for node in nodes:
                        if node.type == 'BSDF_PRINCIPLED':
                            principled_node = node
                            break
                    
                    if principled_node:
                        original_properties[mat.name] = {
                            'base_color': principled_node.inputs['Base Color'].default_value[:],
                            'roughness': principled_node.inputs['Roughness'].default_value,
                            'principled_node': principled_node
                        }
        
        for child in obj.children:
            if child.type == 'MESH':
                process_object(child)
    
    process_object(obj)
    return original_properties


def apply_material_augmentation(original_properties: Dict[str, Dict[str, Any]]) -> None:
    """Apply random augmentation to material properties"""
    augmentation_factor = 0.05
    
    for mat_name, props in original_properties.items():
        principled_node = props['principled_node']
        
        original_color = props['base_color']
        
        h, s, v = colorsys.rgb_to_hsv(original_color[0], original_color[1], original_color[2])
        
        hue_shift = random.uniform(-0.0139, 0.0139)
        h = (h + hue_shift) % 1.0
        
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        
        principled_node.inputs['Base Color'].default_value = (r, g, b, original_color[3])
        
        original_roughness = props['roughness']
        roughness_change = random.uniform(-augmentation_factor, augmentation_factor)
        new_roughness = max(0.0, min(1.0, original_roughness + (original_roughness * roughness_change)))
        principled_node.inputs['Roughness'].default_value = new_roughness


def restore_material_properties(original_properties: Dict[str, Dict[str, Any]]) -> None:
    """Restore original material properties"""
    for mat_name, props in original_properties.items():
        principled_node = props['principled_node']
        principled_node.inputs['Base Color'].default_value = props['base_color']
        principled_node.inputs['Roughness'].default_value = props['roughness']


def create_random_textured_or_uniform_material(albedo_textures: List[bpy.types.Image],
                                               config: DatasetConfig
                                               ) -> Tuple[bpy.types.Material, str, bool, bool, float]:
    """Create a random material using standalone albedo textures"""
    if not albedo_textures:
        raise Exception("No albedo textures provided")
    
    base_color_texture = random.choice(albedo_textures)
    
    transmission, metallic, roughness, material_name, base_color, thickness_value = _generate_random_material_properties(config)

    is_transparent = transmission > 0.0
    is_uniform_color = random.random() < config.uniform_color_probability
    
    if not is_transparent and not material_name.endswith("paint"):
        material_name = material_name.replace("paint", "textured_paint")
    
    material = bpy.data.materials.new(name=material_name)
    material.use_nodes = True
    
    material.node_tree.nodes.clear()
    
    principled = material.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
    principled.location = (300, 0)
    
    output = material.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (600, 0)
    
    if is_transparent or is_uniform_color: # Odds: 30 trans / 10 u / 60 tex
        # For transmissive materials, use uniform color instead of texture
        principled.inputs['Base Color'].default_value = base_color
    else:
        # For non-transmissive materials, use albedo texture
        texture_node = material.node_tree.nodes.new(type='ShaderNodeTexImage')
        texture_node.image = base_color_texture
        texture_node.location = (0, 0)
        material.node_tree.links.new(texture_node.outputs['Color'], principled.inputs['Base Color'])

    material.node_tree.links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    
    principled.inputs['Metallic'].default_value = metallic
    principled.inputs['Roughness'].default_value = roughness

    principled.inputs['Transmission Weight'].default_value = 1 - thickness_value * 0.1 if is_transparent else 0.0
    
    return material, base_color_texture.name, is_uniform_color or is_transparent, is_transparent, thickness_value


def create_normal_pass_material(original_material: bpy.types.Material = None) -> bpy.types.Material:
    """Create a material that outputs normals from the original material's normal map"""
    normal_material = bpy.data.materials.new(name="temp_NormalPassMaterial")
    normal_material.use_nodes = True
    
    nodes = normal_material.node_tree.nodes
    links = normal_material.node_tree.links
    nodes.clear()
    
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (800, 0)
    
    final_normal_output = None
    
    if original_material:
        for node in original_material.node_tree.nodes:
            if node.type == 'NORMAL_MAP':
                normal_map = nodes.new('ShaderNodeNormalMap')
                normal_map.location = (-200, 0)
                
                color_input = node.inputs.get('Color')
                if color_input and color_input.is_linked:
                    texture_node = color_input.links[0].from_node
                    if texture_node.type == 'TEX_IMAGE' and texture_node.image:
                        normal_tex = nodes.new('ShaderNodeTexImage')
                        normal_tex.image = texture_node.image
                        normal_tex.location = (-400, 0)
                        
                        vector_input = texture_node.inputs.get('Vector')
                        if vector_input and vector_input.is_linked:
                            uv_node = vector_input.links[0].from_node
                            if uv_node.type == 'UVMAP':
                                uv_map = nodes.new('ShaderNodeUVMap')
                                uv_map.uv_map = uv_node.uv_map
                                uv_map.location = (-600, -200)
                                links.new(uv_map.outputs['UV'], normal_tex.inputs['Vector'])
                        
                        links.new(normal_tex.outputs['Color'], normal_map.inputs['Color'])
                        final_normal_output = normal_map.outputs['Normal']
                        break
    
    separate = nodes.new('ShaderNodeSeparateXYZ')
    separate.location = (200, 0)
    
    math_r = nodes.new('ShaderNodeMath')
    math_r.operation = 'MULTIPLY'
    math_r.inputs[1].default_value = 2.2
    math_r.location = (400, 100)
    
    math_g = nodes.new('ShaderNodeMath')
    math_g.operation = 'MULTIPLY'
    math_g.inputs[1].default_value = 2.2
    math_g.location = (400, 0)
    
    math_b = nodes.new('ShaderNodeMath')
    math_b.operation = 'MULTIPLY'
    math_b.inputs[1].default_value = -2.2
    math_b.location = (400, -100)
    
    combine = nodes.new('ShaderNodeCombineXYZ')
    combine.location = (600, 0)
    
    emission = nodes.new('ShaderNodeEmission')
    emission.location = (700, 0)
    
    if final_normal_output:
        links.new(final_normal_output, separate.inputs['Vector'])
    
    links.new(separate.outputs['X'], math_r.inputs[0])
    links.new(separate.outputs['Z'], math_g.inputs[0])
    links.new(separate.outputs['Y'], math_b.inputs[0])
    links.new(math_r.outputs['Value'], combine.inputs['X'])
    links.new(math_g.outputs['Value'], combine.inputs['Y'])
    links.new(math_b.outputs['Value'], combine.inputs['Z'])
    links.new(combine.outputs['Vector'], emission.inputs['Color'])
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    return normal_material