import re
import bpy
import os
import bmesh
import gc
from mathutils import Vector

"""
This code should be run in Blender when adjusting models. It re-imports and preprocesses the models to ensure they are ready for dataset generation. 
It handles scaling, centering, UV mapping, and LOD filtering.
"""

def clean_filename_to_collection_name(filename):
    """Convert filename like 'brass_pot_01_8k.fbx' to 'Brass Pot'"""
    # Remove extension
    name = os.path.splitext(filename)[0]
    
    # Split by underscores and filter out numbers and common suffixes
    parts = name.split('_')
    cleaned_parts = []
    
    for part in parts:
        # Skip parts that are purely numeric or common suffixes
        if (not part.isdigit() and 
            part.lower() not in ['8k', '4k', '2k', '1k', 'lod', 'hp', 'lp', '01', '02', '03', '04', '05']):
            cleaned_parts.append(part.capitalize())
    
    return ' '.join(cleaned_parts)

def get_unique_collection_name(main_collection, base_name):
    """Generate a unique collection name using Blender's .XXX suffix convention"""
    # Check if base name is available
    existing_names = [col.name for col in main_collection.children]
    
    if base_name not in existing_names:
        return base_name
    
    # Generate numbered suffix like Blender does (.001, .002, etc.)
    counter = 1
    while True:
        new_name = f"{base_name}.{counter:03d}"
        if new_name not in existing_names:
            return new_name
        counter += 1
        
        # Safety check to prevent infinite loop
        if counter > 999:
            import time
            return f"{base_name}.{int(time.time())}"

def clean_main_objects_collection():
    """Remove all child collections and their contents from Main Objects"""
    main_collection_name = "Main Objects"
    
    if main_collection_name in bpy.data.collections:
        main_collection = bpy.data.collections[main_collection_name]
        
        # Get list of child collections to remove
        child_collections = list(main_collection.children)
        
        print(f"Cleaning up {len(child_collections)} existing collections in Main Objects...")
        
        for child_collection in child_collections:
            print(f"Removing collection: {child_collection.name}")
            
            # Remove all objects in the collection first
            objects_to_remove = list(child_collection.objects)
            for obj in objects_to_remove:
                # Remove from all collections
                for col in obj.users_collection[:]:
                    col.objects.unlink(obj)
                # Remove from Blender data
                bpy.data.objects.remove(obj, do_unlink=True)
            
            # Unlink child collection from main collection
            main_collection.children.unlink(child_collection)
            
            # Remove the collection from Blender data
            bpy.data.collections.remove(child_collection)
        
        print("Main Objects collection cleaned successfully")
    else:
        print("Main Objects collection doesn't exist yet - will be created")

def ensure_main_objects_collection():
    """Ensure 'Main Objects' collection exists and return it"""
    main_collection_name = "Main Objects"
    
    if main_collection_name not in bpy.data.collections:
        main_collection = bpy.data.collections.new(main_collection_name)
        bpy.context.scene.collection.children.link(main_collection)
    else:
        main_collection = bpy.data.collections[main_collection_name]
    
    return main_collection

def find_layer_collection(layer_collection, collection_name):
    """Recursively find layer collection by name"""
    if layer_collection.collection.name == collection_name:
        return layer_collection
    for child in layer_collection.children:
        result = find_layer_collection(child, collection_name)
        if result:
            return result
    return None

def hide_main_objects_children_except(main_collection, current_collection):
    """Hide only children of Main Objects collection, leave everything else untouched"""
    for collection in main_collection.children:
        layer_collection = find_layer_collection(bpy.context.view_layer.layer_collection, collection.name)
        
        if collection != current_collection:
            # Exclude from view layer (saves memory)
            layer_collection.exclude = True
            collection.hide_viewport = True
        else:
            layer_collection.exclude = False

def show_collection(collection):
    """Show a specific collection"""
    collection.hide_viewport = False
    layer_collection = find_layer_collection(bpy.context.view_layer.layer_collection, collection.name)
    if layer_collection:
        layer_collection.exclude = False

def safe_cleanup():
    """Safe memory cleanup - only removes truly unused data, doesn't touch existing collections"""
    # Force update to ensure changes are applied
    bpy.context.view_layer.update()
    
    # Only clean up data blocks that have 0 users (truly unused)
    # This won't touch anything that's part of existing collections
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh, do_unlink=True)
    
    for material in list(bpy.data.materials):
        if material.users == 0:
            bpy.data.materials.remove(material, do_unlink=True)
            
    for texture in list(bpy.data.textures):
        if texture.users == 0:
            bpy.data.textures.remove(texture, do_unlink=True)
            
    for image in list(bpy.data.images):
        if image.users == 0:
            bpy.data.images.remove(image, do_unlink=True)
    
    # Force Python garbage collection
    gc.collect()
    
    # Force Blender to update memory usage
    bpy.context.view_layer.update()

def get_object_dimensions(obj):
    """Get the maximum dimension of an object"""
    if obj.type != 'MESH':
        return 0
    
    # Get bounding box dimensions
    dimensions = obj.dimensions
    max_dimension = max(dimensions.x, dimensions.y, dimensions.z)
    return max_dimension

def scale_object_to_max_size(obj, max_size=3.0):
    """Scale an object uniformly if it exceeds the maximum size"""
    if obj.type != 'MESH':
        return False
    
    current_max_dimension = get_object_dimensions(obj)
    
    if current_max_dimension > max_size:
        # Calculate scale factor
        scale_factor = max_size / current_max_dimension
        
        # Apply uniform scaling
        obj.scale = (scale_factor, scale_factor, scale_factor)
        
        # Apply the transformation to make it permanent
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        
        print(f"Scaled '{obj.name}' from {current_max_dimension:.3f}m to {max_size:.3f}m (factor: {scale_factor:.3f})")
        return True
    
    return False

def scale_objects_in_collection(collection, max_size=3.0):
    """Scale all objects in a collection that exceed the maximum size"""
    scaled_count = 0
    
    for obj in collection.objects:
        if scale_object_to_max_size(obj, max_size):
            scaled_count += 1
    
    if scaled_count > 0:
        print(f"Scaled {scaled_count} objects in collection '{collection.name}'")
    
    return scaled_count

def center_object_on_xy_plane(obj):
    """Center object on XY plane (X=0, Y=0) based on its bounding box"""
    if obj.type != 'MESH':
        return False
    
    # Get the object's bounding box in world coordinates
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    
    # Find the center of the bounding box in X and Y
    min_x = min(corner.x for corner in bbox_corners)
    max_x = max(corner.x for corner in bbox_corners)
    min_y = min(corner.y for corner in bbox_corners)
    max_y = max(corner.y for corner in bbox_corners)
    
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # Calculate offset to center the object
    x_offset = -center_x
    y_offset = -center_y
    
    # Apply the offset
    obj.location.x += x_offset
    obj.location.y += y_offset
    
    print(f"Centered '{obj.name}' on XY plane (offset: X={x_offset:.3f}m, Y={y_offset:.3f}m)")
    return True

def move_object_to_floor(obj):
    """Move object so its bottom sits on Z=0"""
    if obj.type != 'MESH':
        return False
    
    # Get the object's bounding box in world coordinates
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    
    # Find the minimum Z coordinate
    min_z = min(corner.z for corner in bbox_corners)
    
    # Calculate how much to move the object up to put its bottom at Z=0
    z_offset = -min_z
    
    # Move the object
    obj.location.z += z_offset
    
    print(f"Moved '{obj.name}' up by {z_offset:.3f}m to sit on floor")
    return True

def center_and_floor_object(obj):
    """Center object on XY plane and move to floor (Z=0)"""
    if obj.type != 'MESH':
        return False
    
    centered = center_object_on_xy_plane(obj)
    floored = move_object_to_floor(obj)
    
    return centered or floored

def center_and_floor_collection_objects(collection):
    """Center all mesh objects in a collection on XY plane and move to floor"""
    processed_count = 0
    
    for obj in collection.objects:
        if center_and_floor_object(obj):
            processed_count += 1
    
    if processed_count > 0:
        print(f"Centered and floored {processed_count} objects in collection '{collection.name}'")
    
    return processed_count

def move_collection_objects_to_floor(collection):
    """Move all mesh objects in a collection to sit on the floor (Z=0)"""
    moved_count = 0
    
    for obj in collection.objects:
        if move_object_to_floor(obj):
            moved_count += 1
    
    if moved_count > 0:
        print(f"Moved {moved_count} objects to floor in collection '{collection.name}'")
    
    return moved_count

def is_lod_object(obj_name):
    """Check if an object name indicates it's a LOD variant"""
    name_lower = obj_name.lower()
    
    # Common LOD patterns
    lod_patterns = [
        '_lod', '_level', '_detail', '_l0', '_l1', '_l2', '_l3', '_l4', '_l5'
    ]
    
    for pattern in lod_patterns:
        if pattern in name_lower:
            return True
    
    return False

def get_lod_level(obj_name):
    """Extract LOD level from object name. Returns -1 if not a LOD object, 0 for LOD0, etc."""
    name_lower = obj_name.lower()
    
    # Check for various LOD naming patterns
    import re
    
    # Pattern: _LOD0, _LOD1, _LOD2, etc.
    lod_match = re.search(r'_lod(\d+)', name_lower)
    if lod_match:
        return int(lod_match.group(1))
    
    # Pattern: _Level0, _Level1, _Level2, etc.
    level_match = re.search(r'_level(\d+)', name_lower)
    if level_match:
        return int(level_match.group(1))
    
    # Pattern: _Detail0, _Detail1, _Detail2, etc.
    detail_match = re.search(r'_detail(\d+)', name_lower)
    if detail_match:
        return int(detail_match.group(1))
    
    # Pattern: _L0, _L1, _L2, etc.
    l_match = re.search(r'_l(\d+)', name_lower)
    if l_match:
        return int(l_match.group(1))
    
    # If it contains LOD but no number, assume it's LOD0
    if is_lod_object(obj_name):
        return 0
    
    return -1  # Not a LOD object

def get_base_name_without_lod(obj_name):
    """Get the base name of an object without LOD suffix"""
    name_lower = obj_name.lower()
    
    # Remove common LOD patterns
    import re
    
    # Remove _LOD + number
    name = re.sub(r'_lod\d*', '', obj_name, flags=re.IGNORECASE)
    
    # Remove _Level + number
    name = re.sub(r'_level\d*', '', name, flags=re.IGNORECASE)
    
    # Remove _Detail + number
    name = re.sub(r'_detail\d*', '', name, flags=re.IGNORECASE)
    
    # Remove _L + number
    name = re.sub(r'_l\d*', '', name, flags=re.IGNORECASE)
    
    return name

def filter_lod_objects(collection):
    """Remove all LOD objects except LOD0 from a collection"""
    mesh_objects = [obj for obj in collection.objects if obj.type == 'MESH']
    
    if not mesh_objects:
        return 0
    
    # Group objects by base name
    object_groups = {}
    
    for obj in mesh_objects:
        base_name = get_base_name_without_lod(obj.name)
        lod_level = get_lod_level(obj.name)
        
        if base_name not in object_groups:
            object_groups[base_name] = []
        
        object_groups[base_name].append((obj, lod_level))
    
    removed_count = 0
    
    # For each group, keep only LOD0 (or the lowest LOD if no LOD0 exists)
    for base_name, obj_list in object_groups.items():
        if len(obj_list) <= 1:
            continue  # Only one object, no LODs to filter
        
        # Sort by LOD level (-1 for non-LOD objects, then 0, 1, 2, etc.)
        obj_list.sort(key=lambda x: (x[1] if x[1] >= 0 else 999, x[0].name))
        
        # Find the best object to keep (prefer LOD0, then lowest LOD, then non-LOD)
        keep_obj = None
        
        # First, look for LOD0
        for obj, lod_level in obj_list:
            if lod_level == 0:
                keep_obj = obj
                break
        
        # If no LOD0, keep the lowest numbered LOD
        if keep_obj is None:
            lod_objects = [(obj, lod) for obj, lod in obj_list if lod >= 0]
            if lod_objects:
                keep_obj = lod_objects[0][0]  # Already sorted, so first is lowest
        
        # If no LOD objects at all, keep the first non-LOD object
        if keep_obj is None:
            non_lod_objects = [(obj, lod) for obj, lod in obj_list if lod == -1]
            if non_lod_objects:
                keep_obj = non_lod_objects[0][0]
        
        # Remove all other objects in this group
        for obj, lod_level in obj_list:
            if obj != keep_obj:
                print(f"Removing LOD variant: {obj.name} (LOD{lod_level if lod_level >= 0 else 'N/A'})")
                
                # Remove from collection
                collection.objects.unlink(obj)
                
                # Remove from Blender data
                bpy.data.objects.remove(obj, do_unlink=True)
                
                removed_count += 1
        
        if removed_count > 0:
            keep_lod = get_lod_level(keep_obj.name)
            print(f"Kept: {keep_obj.name} (LOD{keep_lod if keep_lod >= 0 else 'N/A'}) for base: {base_name}")
    
    if removed_count > 0:
        print(f"Removed {removed_count} LOD variants from collection '{collection.name}'")
    
    return removed_count

def add_sphere_projected_uv(obj):
    """Adds a new UV map and applies Sphere Projection to all vertices"""
    map_name = "Albedo Spherical UV map"
    
    # Make sure object is active and selected
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    # Create the new UV map
    if map_name not in obj.data.uv_layers:
        obj.data.uv_layers.new(name=map_name)
    
    # Switch to Edit Mode to perform projection
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    
    # Set the new UV map as active so the projection targets it
    obj.data.uv_layers[map_name].active = True
    
    # Perform Sphere Projection. Uses ALIGN_TO_OBJECT to NOT account for viewport camera look direction.
    # clip_to_bounds ensures the UVs fit in the 0-1 space
    bpy.ops.uv.sphere_project(direction='ALIGN_TO_OBJECT', align='POLAR_ZX', clip_to_bounds=True)
    
    # Return to Object Mode
    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"Added '{map_name}' with Sphere Projection to {obj.name}")

def add_cubic_projected_uv(obj):
    """Adds a new UV map and applies Cube Projection based on vertex mean distance"""
    map_name = "Albedo Cubic UV map"
    
    # Ensure we are in Object Mode to access vertex data
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    auto_cube_size = max(obj.dimensions)
    
    # ------------------------------------------------------------
    
    # Create the new UV map if it doesn't exist
    if map_name not in obj.data.uv_layers:
        obj.data.uv_layers.new(name=map_name)
    
    # Switch to Edit Mode
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    
    # Set the new UV map as active
    obj.data.uv_layers[map_name].active = True
    
    # Perform Cube Projection
    # cube_size: Uses the calculated mean radius
    # clip_to_bounds=False: Keeps UVs in world-space scale (doesn't force 0-1)
    bpy.ops.uv.cube_project(
        cube_size=auto_cube_size, 
        correct_aspect=True, 
        clip_to_bounds=True
    )
    
    # Return to Object Mode
    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"Added '{map_name}' to {obj.name} with auto-size: {auto_cube_size:.4f}")

def standardize_uv_maps(objects):
    """Standardize UV map names across all objects before joining"""
    standard_uv_name = "Original UV Map"
    
    mesh_objects = [obj for obj in objects if obj.type == 'MESH' and obj.data.uv_layers]
    uv_maps_renamed = 0
    
    for obj in mesh_objects:
        if obj.data.uv_layers:
            for uv_layer in obj.data.uv_layers:
                if uv_layer.name != standard_uv_name:
                    old_name = uv_layer.name
                    uv_layer.name = standard_uv_name
                    print(f"  {obj.name}: '{old_name}' -> '{standard_uv_name}'")
                    uv_maps_renamed += 1
    
    if uv_maps_renamed > 0:
        print(f"Renamed {uv_maps_renamed} UV maps to match standard name")
    else:
        print("All UV maps already have standardized names")

def join_objects_in_collection(collection):
    """Join all mesh objects in a collection into a single object"""
    mesh_objects = [obj for obj in collection.objects if obj.type == 'MESH']
    
    # Standardize UV map names before joining
    standardize_uv_maps(mesh_objects)
    
    if len(mesh_objects) <= 1:
        return mesh_objects[0] if mesh_objects else None
    
    # Deselect all objects
    bpy.ops.object.select_all(action='DESELECT')
    
    # Select all mesh objects in the collection
    for obj in mesh_objects:
        obj.select_set(True)
    
    # Set the first object as active
    bpy.context.view_layer.objects.active = mesh_objects[0]
    
    # Join objects
    bpy.ops.object.join()
    
    return bpy.context.active_object

def extrude_along_normals(obj, distance):
    """Apply extrude along normals to an object using the same operator as the UI"""
    # Ensure the object is selected and active
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    # Enter edit mode
    bpy.ops.object.mode_set(mode='EDIT')
    
    # Select all faces
    bpy.ops.mesh.select_all(action='SELECT')
    
    # Use the actual "Extrude Along Normals" operator that matches the UI
    bpy.ops.mesh.extrude_region_shrink_fatten(
        MESH_OT_extrude_region={"use_normal_flip": False, "use_dissolve_ortho_edges": False, "mirror": False},
        TRANSFORM_OT_shrink_fatten={"value": distance, "use_even_offset": False, "mirror": False, "use_proportional_edit": False}
    )
    
    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')

def setup_materials_with_ao_and_displacement(new_objects):
    """Setup AO nodes and connect normal maps to displacement for all materials in new objects"""
    processed_materials = set()  # Avoid processing the same material multiple times
    
    for obj in new_objects:
        if obj.type == 'MESH' and obj.data.materials:
            for material in obj.data.materials:
                if material and material not in processed_materials:
                    processed_materials.add(material)
                    
                    if not material.use_nodes:
                        continue
                    
                    nodes = material.node_tree.nodes
                    links = material.node_tree.links
                    
                    # Find the Material Output node
                    output_node = None
                    for node in nodes:
                        if node.type == 'OUTPUT_MATERIAL':
                            output_node = node
                            break
                    
                    # Find the main shader node (usually Principled BSDF)
                    main_shader = None
                    for node in nodes:
                        if node.type == 'BSDF_PRINCIPLED':
                            main_shader = node
                            break
                    
                    if not main_shader:
                        # In strict mode, we should fail loud. 
                        # But standard setup usually has Principled. Let's warn instead of aborting the whole script.
                        print(f"Warning: No Principled BSDF found in material: {material.name}. Skipping AO setup.")
                        continue
                    
                    # === AO NODE SETUP ===
                    # Create AO node
                    ao_node = nodes.new('ShaderNodeAmbientOcclusion')
                    ao_node.samples = 16  # Higher samples for better quality
                    ao_node.inputs['Distance'].default_value = 1
                    ao_node.location = (main_shader.location.x - 400, main_shader.location.y + 200)
                    
                    # Create Mix Shader node
                    mix_shader = nodes.new('ShaderNodeMixShader')
                    mix_shader.location = (main_shader.location.x + 200, main_shader.location.y)
                    
                    # Connect AO to Mix Shader
                    links.new(ao_node.outputs['Color'], mix_shader.inputs[0])  # AO Color to Fac
                    links.new(main_shader.outputs[0], mix_shader.inputs[2])  # Second shader (same, but will be darkened by AO)
                    
                    # Connect Mix Shader to Material Output
                    # First, check if there's already a connection to Surface
                    if output_node.inputs['Surface'].is_linked:
                        # Disconnect existing connection
                        for link in output_node.inputs['Surface'].links:
                            links.remove(link)
                    
                    # Connect Mix Shader to Surface
                    links.new(mix_shader.outputs[0], output_node.inputs['Surface'])
                    
                    print(f"Added AO node setup to material: {material.name}")
                    
                    
def process_single_file(file_path, main_collection, max_object_size):
    """Process a single FBX or GLB file with safe memory management"""
    filename = os.path.basename(file_path)
    extension = os.path.splitext(filename)[1].lower()
    
    # ------------------------------------------------------------
    # A bit redundant since os.walk is already filtering for standard models,
    # but added to explicitly ignore the 'Quads' subdirectory if it was accidentally caught.
    if "Quads" in file_path.split(os.sep):
        return
    # ------------------------------------------------------------
    
    print(f"Processing standard mesh: {file_path}")
    
    # Generate base collection name
    base_collection_name = clean_filename_to_collection_name(filename)
    
    # Get unique collection name
    collection_name = get_unique_collection_name(main_collection, base_collection_name)
    
    # Create new collection
    new_collection = bpy.data.collections.new(collection_name)
    main_collection.children.link(new_collection)
    
    # Hide others to save viewport memory
    hide_main_objects_children_except(main_collection, new_collection)
    
    try:
        safe_cleanup()
        objects_before = set(bpy.context.scene.objects)
        
        # --- THE FUNNEL: Import based on extension ---
        if extension == '.fbx':
            bpy.ops.import_scene.fbx(
                filepath=file_path,
                use_custom_normals=False,
                use_image_search=True,
                use_anim=False,
                ignore_leaf_bones=True
            )
        elif extension in ['.glb', '.gltf']:
            # GLTF import usually handles textures better for GLBs
            bpy.ops.import_scene.gltf(
                filepath=file_path,
                import_shading='NORMALS'
            )
        
        # Get newly imported objects
        objects_after = set(bpy.context.scene.objects)
        new_objects = objects_after - objects_before
        
        if not new_objects:
            # RaiseException is 'loud' enough.
            raise Exception(f"Warning: No objects imported from {filename}")
        
        # --- REST OF THE FLOW IS THE SAME ---
        # Handle collections assignment
        for obj in new_objects:
            for col in obj.users_collection[:]:
                col.objects.unlink(obj)
            new_collection.objects.link(obj)
            
        # Standard Processing Pipeline
        filter_lod_objects(new_collection)
        joined_object = join_objects_in_collection(new_collection)
        
        if joined_object:
            bpy.ops.object.select_all(action='DESELECT')
            joined_object.select_set(True)
            bpy.context.view_layer.objects.active = joined_object
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
            
            setup_materials_with_ao_and_displacement([joined_object])
            scale_objects_in_collection(new_collection, max_object_size)
            center_and_floor_object(joined_object)
            add_sphere_projected_uv(joined_object)
            add_cubic_projected_uv(joined_object)
            
            # Cover object logic
            bpy.ops.object.select_all(action='DESELECT')
            joined_object.select_set(True)
            bpy.context.view_layer.objects.active = joined_object
            bpy.ops.object.duplicate(linked=False)
            cover_object = bpy.context.active_object
            cover_object.name = f"cover_{joined_object.name}"
            
            extrude_along_normals(cover_object, 0.0004)
            cover_object.scale = (1.0005, 1.0005, 1.0005)
            
            cover_object.visible_shadow = False
            
            # Apply ALL transforms to BOTH objects simultaneously
            bpy.ops.object.select_all(action='DESELECT')
            joined_object.select_set(True)
            cover_object.select_set(True)
            bpy.context.view_layer.objects.active = joined_object
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            
            print(f"Successfully processed {filename} -> Collection: {collection_name}")
            
    except Exception as e:
        print(f"CRITICAL ERROR processing standard mesh {filename}: {str(e)}")
    finally:    
        safe_cleanup()

def process_models_folder(root_path, max_imports, max_object_size=3.0):
    """Process standard FBX and GLB files safely, ignoring Quads folder"""
    print("\n=== CLEANING UP MAIN OBJECTS COLLECTION ===")
    clean_main_objects_collection()
    
    bpy.context.preferences.filepaths.use_relative_paths = True
    main_collection = ensure_main_objects_collection()
    
    # Search for standard extensions
    valid_extensions = ('.fbx', '.glb', '.gltf')
    all_files = []
    
    for subdir, dirs, files in os.walk(root_path):
        # ------------------------------------------------------------
        # MODIFICATION: Skip the 'Quads' directory entirely for standard meshes pipeline
        if "Quads" in subdir.split(os.sep):
            continue
        # ------------------------------------------------------------
            
        target_files = [os.path.join(subdir, f) for f in files if f.lower().endswith(valid_extensions)]
        all_files.extend(target_files)
    
    print(f"Found {len(all_files)} compatible FBX/GLB files to process")
    safe_cleanup()
    
    processed_count = 0
    for i, file_path in enumerate(all_files):
        print(f"\n--- Processing file {i+1}/{len(all_files)} ---")
        process_single_file(file_path, main_collection, max_object_size)
        processed_count += 1
        
        if max_imports is not None and processed_count >= max_imports:
            break
    
    # Hide remaining children after pipeline finishes
    hide_main_objects_children_except(main_collection, None)

# ==============================================================================
# NEW: QUADS (THIN CUBE) PIPELINE FUNCTIONS
# ==============================================================================

def process_single_quad_material(quad_folder_path, main_collection, max_object_size):
    """Process a single material folder from the Quads directory, appending exact material and generating upright thin cube geometry"""
    import math
    folder_name = os.path.basename(quad_folder_path)
    base_file_name = folder_name.replace('.blend', '')
    
    # 1. Clean the name using your existing function
    cleaned_name = clean_filename_to_collection_name(base_file_name) # e.g., "Asphalt Floor"
    
    # Convert cleaned name to snake_case to match exact material data block and mesh names
    snake_case_name = cleaned_name.lower().replace(' ', '_') # e.g., "asphalt_floor"
    
    blend_file_path = os.path.join(quad_folder_path, folder_name)
    
    if not os.path.isfile(blend_file_path):
        print(f"Skipping quad {folder_name}: Could not find inner .blend file at {blend_file_path}")
        return
        
    print(f"\nProcessing Quad Material: {cleaned_name}")
    
    # Collection Setup (e.g., "Quad Asphalt Floor")
    collection_base = f"Quad {cleaned_name}"
    collection_name = get_unique_collection_name(main_collection, collection_base)
    new_collection = bpy.data.collections.new(collection_name)
    main_collection.children.link(new_collection)
    hide_main_objects_children_except(main_collection, new_collection)
    
    try:
        safe_cleanup()
        
        # 2. Append the material using the perfectly matched snake_case_name
        with bpy.data.libraries.load(blend_file_path, link=False) as (data_from, data_to):
            if snake_case_name not in data_from.materials:
                raise ValueError(f"Material '{snake_case_name}' missing in {blend_file_path}. Available: {data_from.materials}")
            
            data_to.materials = [snake_case_name]
                
        mat = bpy.data.materials.get(snake_case_name)
        if not mat:
            raise RuntimeError(f"Material '{snake_case_name}' failed to load into the current scene data.")

        # 3. Generate the Geometry: "Thin Cube"
        bpy.ops.mesh.primitive_cube_add(size=max_object_size)
        cube = bpy.context.active_object
        
        # Name the instance using the snake_case name (e.g., "quad_asphalt_floor")
        cube.name = f"quad_{snake_case_name}"

        # Scale Z down severely to make it a thin board
        cube.scale.z = 0.2

        # Rotate 90 degrees around X axis to stand it upright
        cube.rotation_euler[0] = math.radians(90)

        # Apply BOTH scale and rotation so the object properties zero out correctly
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

        # Assign Material
        cube.data.materials.append(mat)

        # Move to correct collection
        for col in cube.users_collection[:]:
            col.objects.unlink(cube)
        new_collection.objects.link(cube)

        # 4. Standard Pipeline Processing
        setup_materials_with_ao_and_displacement([cube])
        standardize_uv_maps([cube])
        center_and_floor_object(cube)
        add_sphere_projected_uv(cube)
        add_cubic_projected_uv(cube)

        # 5. Cover Object Logic
        bpy.ops.object.select_all(action='DESELECT')
        cube.select_set(True)
        bpy.context.view_layer.objects.active = cube
        bpy.ops.object.duplicate(linked=False)

        cover_object = bpy.context.active_object
        cover_object.name = f"cover_{cube.name}" # Yields "cover_quad_asphalt_floor"

        extrude_along_normals(cover_object, 0.0004)
        cover_object.scale = (1.0005, 1.0005, 1.0005)
        
        cover_object.visible_shadow = False
        
        # Apply ALL transforms to BOTH objects simultaneously
        bpy.ops.object.select_all(action='DESELECT')
        cube.select_set(True)
        cover_object.select_set(True)
        bpy.context.view_layer.objects.active = cube
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
        print(f"Successfully processed quad material -> Collection: {collection_name}")
        
    except Exception as e:
        print(f"CRITICAL ERROR processing quad material {cleaned_name}: {str(e)}")
    finally:    
        safe_cleanup()


def process_quads_directory(quads_root_path, main_collection, max_object_size=3.0):
    """Finds all material folders in the Quads directory structure and processes them using the Quads pipeline"""
    if not os.path.exists(quads_root_path):
        print(f"Quads directory not found at {quads_root_path}. Skipping.")
        return
        
    print("\n=== PROCESSING QUADS FOLDER ===")
    
    # Iterate through immediate subdirectories that end with .blend (the constant structure)
    # Filter using os.path.isdir to ignore stray files
    quad_folders = []
    for d in os.listdir(quads_root_path):
        folder_path = os.path.join(quads_root_path, d)
        if os.path.isdir(folder_path) and d.lower().endswith('.blend'):
            quad_folders.append(folder_path)
    
    print(f"Found {len(quad_folders)} quad material folders (folder_name.blend/) to process using Quads pipeline.")
    
    for i, folder_path in enumerate(quad_folders):
        print(f"\n--- Processing Quad material {i+1}/{len(quad_folders)} ---")
        process_single_quad_material(folder_path, main_collection, max_object_size)

# ==============================================================================
# VIEWPORT MANAGEMENT UTILITIES
# ==============================================================================

def show_main_objects_collection_by_name(collection_name):
    """Show a specific Main Objects child collection by name"""
    main_collection = ensure_main_objects_collection()
    
    # Hide all Main Objects children first
    hide_main_objects_children_except(main_collection, None)
    
    # Find and show the requested collection
    for collection in main_collection.children:
        if collection.name == collection_name:
            show_collection(collection)
            print(f"Showing collection: {collection_name}")
            return
    
    print(f"Collection '{collection_name}' not found in Main Objects")

def list_main_objects_collections():
    """List all collections under Main Objects"""
    main_collection = ensure_main_objects_collection()
    print("Available Main Objects collections:")
    for i, collection in enumerate(main_collection.children):
        status = "hidden" if collection.hide_viewport else "visible"
        print(f"{i+1}. {collection.name} ({status})")

def hide_all_main_objects_collections():
    """Hide all Main Objects child collections to save maximum memory"""
    main_collection = ensure_main_objects_collection()
    hide_main_objects_children_except(main_collection, None)
    print("All Main Objects collections are now hidden (maximum memory savings)")

# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    root_directory = bpy.path.abspath(os.path.join("//", "Models/"))
    quads_directory = os.path.join(root_directory, "Quads")
    dataset_max_size = 0.7
    
    print(f"Root models directory established: {root_directory}")
    print(f"Quads materials directory established: {quads_directory}")
    
    # ------------------------------------------------------------
    # Pipeline Part 1: Process standard FBX/GLB meshes
    # (Updated logic automatically ignores the 'Quads' folder)
    # ------------------------------------------------------------
    process_models_folder(root_directory, max_imports=None, max_object_size=dataset_max_size)
    
    # ------------------------------------------------------------
    # Pipeline Part 2: Process the Quads materials folders
    # ------------------------------------------------------------
    main_collection = ensure_main_objects_collection()
    process_quads_directory(quads_directory, main_collection, max_object_size=dataset_max_size)
    
    # ------------------------------------------------------------
    # Final cleanup, save, and export preparation
    # ------------------------------------------------------------
    # Hide all collections to save memory before saving the file
    hide_main_objects_children_except(main_collection, None)
    bpy.ops.file.make_paths_relative()
    bpy.ops.wm.save_mainfile()
    
    print("\n=== ALL DATASET PIPELINES COMPLETE ===")