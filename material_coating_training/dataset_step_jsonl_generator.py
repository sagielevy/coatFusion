import json
from pathlib import Path

from tqdm import tqdm

def create_entry(root_dir, src_img, tgt_img, mask_path, coating_mask_path, normal_path, projected_albedos_path,
                 albedos_path, depth_path, text_data, data_key, apply_texture_task, replace_task, remove_task):
    """
    Create a dataset entry with the given parameters.
    
    Args:
        root_dir: Root directory path
        src_img: Source image path
        tgt_img: Target image path
        mask_path: Object mask path
        coating_mask_path: Coating mask path
        normal_path: Normal map path
        projected_albedos_path: Path to projected albedos image (or None)
        text_data: Text data dictionary
        data_key: Key to extract data from text_data
        apply_texture_task: Value for apply_texture_task field
        replace_task: Value for replace_task field
        remove_task: Value for remove_task field
    
    Returns:
        Dictionary representing the dataset entry
    """
    caption = text_data[data_key]["caption"]
    instruction = text_data[data_key]["instruction"]
    material_props = text_data[data_key]["material_properties"]

    base_color_r = material_props["base_color_r"]
    base_color_g = material_props["base_color_g"]
    base_color_b = material_props["base_color_b"]

    thickness = text_data[data_key]["thickness"]
    metallic = material_props["metallic"]
    roughness = material_props["roughness"]
    transmission_weight = material_props["transmission_weight"]

    uv_mapping = material_props["uv_mapping"]
    best_mapping_method =  material_props["best_mapping_method"] # Use to filter data for best method only approach.

    return {
        "image": str((root_dir / src_img.relative_to(root_dir)).as_posix()),
        "caption": caption,
        "obj_mask": str((root_dir / mask_path.relative_to(root_dir)).as_posix()),
        "instruction": instruction,
        "target_image": str((root_dir / tgt_img.relative_to(root_dir)).as_posix()),
        "albedo_image": str(albedos_path.as_posix()) if albedos_path else None,
        "projected_albedos": str((root_dir / projected_albedos_path.relative_to(root_dir)).as_posix()) if projected_albedos_path else None,
        "coating": text_data[data_key]["coating"],
        "thickness": thickness,
        "coating_mask": str((root_dir / coating_mask_path.relative_to(root_dir)).as_posix()),
        "normal": str((root_dir / normal_path.relative_to(root_dir)).as_posix()),
        "depth": str((root_dir / depth_path.relative_to(root_dir)).as_posix()),
        "base_color_r": base_color_r,
        "base_color_g": base_color_g,
        "base_color_b": base_color_b,
        "metallic": metallic,
        "roughness": roughness,
        "transmission_weight": transmission_weight,
        "uv_mapping": uv_mapping,
        "best_mapping_method": best_mapping_method,
        "apply_texture_task": apply_texture_task,
        "replace_task": replace_task,
        "remove_task": remove_task
    }


def generate_jsonl(root_dir, output_file, replace_factor, should_remove, albedos_path):
    root_dir = Path(root_dir)
    sample_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")])

    entries = []

    for sample_dir in tqdm(sample_dirs, desc="Processing samples"):
        text_data_path = sample_dir / "text_data.json"
        mask_path = sample_dir / "obj_mask.png"
        normal_path = sample_dir / "normal.png"
        coating_mask_path = sample_dir / "coating_mask.png"
        depth_path = sample_dir / "depth.png"

        if not text_data_path.exists() or not mask_path.exists() or not coating_mask_path.exists() or not depth_path.exists():
            raise ValueError(f"Required files missing for sample {sample_dir.name}")

        with open(text_data_path, 'r') as f:
            text_data = json.load(f)

        available_ids = list(text_data.keys())  # e.g., ["coating_0", "1", "2", ...]
        coating_ids = [cid for cid in available_ids if cid != "coating_0"]
        
        for coating_id in coating_ids:
            src_img = sample_dir / "coating_0.png"
            tgt_img = sample_dir / f"coating_{coating_id}.png"
            projected_albedos_path = sample_dir / f"coating_{coating_id}_albedo.png"
            albedo_coating_path = albedos_path / text_data[coating_id]["albedo_texture"]

            # Check if projected albedos exist to determine task type
            if projected_albedos_path.exists():
                # Generate ADD TEXTURE task entries (coating_0 -> coated with texture)
                entry = create_entry(
                    root_dir, src_img, tgt_img, mask_path, coating_mask_path, normal_path, projected_albedos_path,
                    albedo_coating_path, depth_path, text_data, coating_id, 1.0, 0.0, 0.0
                )
                entries.append(entry)
            else:
                # Generate ADD UNIFORM task entries (coating_0 -> coated with uniform color)
                entry = create_entry(
                    root_dir, src_img, tgt_img, mask_path, coating_mask_path, normal_path, None, None,
                    depth_path, text_data, coating_id, 1.0, 0.0, 0.0
                )
                entries.append(entry)

            # Generate REMOVE task entries (coated -> coating_0)
            if should_remove:
                if projected_albedos_path.exists():
                    entry = create_entry(
                        root_dir, tgt_img, src_img, mask_path, coating_mask_path, normal_path, projected_albedos_path, albedo_coating_path,
                        depth_path, text_data, coating_id, 0.0, 0.0, 1.0
                    )
                else:
                    entry = create_entry(
                        root_dir, tgt_img, src_img, mask_path, coating_mask_path, normal_path, None, None,
                        depth_path, text_data, coating_id, 0.0, 0.0, 1.0
                    )
                entries.append(entry)

        # Generate REPLACE task entries (coated -> different_coated)
        other_coatings_count = len(coating_ids) - 1

        for i in range(other_coatings_count * replace_factor):
            src_coating_id = coating_ids[i // other_coatings_count]
            tgt_coating_id = coating_ids[(i + 1) % len(coating_ids)]

            src_img = sample_dir / f"coating_{src_coating_id}.png"
            tgt_img = sample_dir / f"coating_{tgt_coating_id}.png"
            projected_albedos_path = sample_dir / f"coating_{tgt_coating_id}_albedo.png"
            albedo_coating_path = albedos_path / text_data[tgt_coating_id]["albedo_texture"]

            # Check if projected albedos exist for replace task
            if projected_albedos_path.exists():
                # Replace with texture task
                entry = create_entry(
                    root_dir, src_img, tgt_img, mask_path, coating_mask_path, normal_path, projected_albedos_path, albedo_coating_path,
                    depth_path, text_data, tgt_coating_id, 0.0, 1.0, 0.0
                )
            else:
                # Replace with uniform task
                entry = create_entry(
                    root_dir, src_img, tgt_img, mask_path, coating_mask_path, normal_path, None, None,
                    depth_path, text_data, tgt_coating_id, 0.0, 1.0, 0.0
                )
            entries.append(entry)


    print(f"Writing {len(entries)} entries to {output_file}")
    with open(output_file, 'w') as out_file:
        for e in entries:
            out_file.write(json.dumps(e) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=str, help="Path to dataset root")
    parser.add_argument("output", type=str, help="Path to output JSONL file")
    parser.add_argument("--replace-factor", type=int, default=1, help="Factor for replace task generation")
    parser.add_argument("--no-remove", action="store_true", help="Disable remove task generation")
    parser.add_argument("--albedos-path", type=str, required=True, help="Path to albedos directory")
    args = parser.parse_args()

    generate_jsonl(args.root_dir, args.output, args.replace_factor, not args.no_remove, Path(args.albedos_path))