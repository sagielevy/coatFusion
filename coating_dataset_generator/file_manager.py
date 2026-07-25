import os
import json
import glob
import re
import shutil
from typing import Dict, List, Tuple, Any
import bpy


def get_object_class_name(obj) -> str:
    """Extract a clean class name from the object name"""
    name = obj.name.split('.')[0]
    name = ''.join([c if c.isalpha() or c.isspace() else ' ' for c in name])
    name = ' '.join([word.lower() for word in name.split()])
    return name


def get_coating_mask_name(name: str) -> str:
    """Clean coating mask name"""
    name = name.split('.')[0]
    name = ''.join([c if c.isalpha() or c.isspace() else ' ' for c in name])
    name = ' '.join([word.lower() for word in name.split()])
    return name


def get_material_name(material) -> str:
    """Extract a clean material name"""
    name = material.name.split('.')[0]
    name = ''.join([c if c.isalpha() or c.isspace() else ' ' for c in name])
    
    split_words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])', name)
    cleaned_name = ' '.join(word.lower() for word in split_words)
    
    return cleaned_name


def get_floor_name(floor_obj) -> str:
    """Extract floor type name from the floor object"""
    if floor_obj.material_slots and floor_obj.material_slots[0].material:
        mat_name = get_material_name(floor_obj.material_slots[0].material)
        return mat_name

    name = floor_obj.name.split('.')[0].lower()
    if "floor" not in name:
        return f"{name} floor"
    return name

def is_sample_complete(sample_idx: int, output_root: str, coating_materials_count: int) -> bool:
    sample_path = os.path.join(output_root, f"sample_{sample_idx}")

    required_files = [
        "depth.png",
        "normal.png",
        "coating_mask.png",
        "obj_mask.png",
        "text_data.json"
    ]

    # 2. Add the dynamic coating files (0 to coating_materials_count inclusive)
    for i in range(coating_materials_count + 1):
        required_files.append(f"coating_{i}.png")

    # 3. Check if every required file exists in the sample directory
    for file_name in required_files:
        file_path = os.path.join(sample_path, file_name)
        if not os.path.exists(file_path):
            return False

    return True


def set_output_paths(sample_idx: int, output_root: str, obj_mask_output, depth_output) -> str:
    """Set output paths for the current sample"""
    sample_dir = os.path.join(output_root, f"sample_{sample_idx}")

    # Clear current dir first.
    if os.path.exists(sample_dir):
        shutil.rmtree(sample_dir)
    os.makedirs(sample_dir)
    
    bpy.context.scene.render.filepath = os.path.join(sample_dir, "image")
    
    obj_mask_output.base_path = sample_dir
    obj_mask_output.file_slots[0].path = "temp_obj_mask_"
    
    depth_output.base_path = sample_dir
    depth_output.file_slots[0].path = "temp_depth_"
    
    return sample_dir


def find_and_rename_passes(sample_dir: str) -> None:
    """Find and rename all pass files to standard names"""
    file_patterns = [
        ("temp_obj_mask_*.png", "temp_obj_mask.png"),
        ("temp_normal_*.png", "temp_normal.png"),
        ("temp_depth_*.png", "temp_depth.png")
    ]
    
    for pattern, target_name in file_patterns:
        files = glob.glob(os.path.join(sample_dir, pattern))
        if files:
            files.sort(key=os.path.getctime)
            latest_file = files[-1]
            target_path = os.path.join(sample_dir, target_name)
            
            if os.path.exists(target_path):
                os.remove(target_path)
            os.rename(latest_file, target_path)
            
            for file in files[:-1]:
                try:
                    os.remove(file)
                except:
                    pass


def organize_output_files(sample_dir: str) -> None:
    """Rename and organize output files"""
    files_to_move = [
        ("temp_obj_mask.png", "obj_mask.png"),
        ("temp_depth.png", "depth.png"),
    ]
    
    for old_name, new_name in files_to_move:
        old_path = os.path.join(sample_dir, old_name)
        new_path = os.path.join(sample_dir, new_name)
        if os.path.exists(old_path):
            if os.path.exists(new_path):
                os.remove(new_path)
            os.rename(old_path, new_path)


def cleanup_temp_files(sample_dir: str) -> None:
    """Clean up temporary files"""
    temp_files = [
        "temp_obj_mask.png",
        "temp_normal.png", 
        "temp_depth.png"
    ]
    
    for temp_file in temp_files:
        temp_path = os.path.join(sample_dir, temp_file)
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


def save_prompts_json(sample_dir: str, prompts_data: Dict[str, Any]) -> None:
    """Save prompts JSON to sample directory"""
    with open(os.path.join(sample_dir, "text_data.json"), 'w') as f:
        json.dump(prompts_data, f, indent=2)