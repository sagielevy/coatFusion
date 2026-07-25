import bpy
import math
from mathutils import Vector
from typing import List


def get_object_bounds_recursive(obj) -> List[Vector]:
    """Get the world space bounding box of an object and all its children"""
    all_coords = []

    if obj.type == 'MESH' and obj.data.vertices:
        world_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        all_coords.extend(world_corners)

    for child in obj.children:
        all_coords.extend(get_object_bounds_recursive(child))
    return all_coords


def get_average_bounds_size(obj) -> float:
    bounds = get_object_bounds_recursive(obj)
    if not bounds:
        return 0.0
    
    min_x = min(coord.x for coord in bounds)
    max_x = max(coord.x for coord in bounds)
    min_y = min(coord.y for coord in bounds)
    max_y = max(coord.y for coord in bounds)
    min_z = min(coord.z for coord in bounds)
    max_z = max(coord.z for coord in bounds)
    
    width = max_x - min_x
    depth = max_y - min_y
    height = max_z - min_z
    
    return (width + depth + height) / 3.0


def calculate_required_distance_from_object(obj, camera, margin_factor: float) -> tuple[Vector, float]:
    """Calculate required distance from object center to frame it properly"""
    world_coords = get_object_bounds_recursive(obj)

    if not world_coords:
        raise ValueError("Could not get object bounds")

    object_center_world = Vector((0, 0, 0))
    for coord in world_coords:
        object_center_world += coord
    object_center_world /= len(world_coords)

    min_x = min(coord.x for coord in world_coords)
    max_x = max(coord.x for coord in world_coords)
    min_y = min(coord.y for coord in world_coords)
    max_y = max(coord.y for coord in world_coords)
    min_z = min(coord.z for coord in world_coords)
    max_z = max(coord.z for coord in world_coords)

    width = max_x - min_x
    height = max_z - min_z
    depth = max_y - min_y

    max_dimension = max(width, height, depth)

    scene = bpy.context.scene
    render_width = scene.render.resolution_x
    render_height = scene.render.resolution_y
    aspect_ratio = render_width / render_height

    sensor_width = camera.data.sensor_width
    lens = camera.data.lens

    if camera.data.sensor_fit == 'HORIZONTAL':
        fov_y = 2 * math.atan((sensor_width / aspect_ratio) / (2 * lens))
    else:
        fov_y = 2 * math.atan(sensor_width / (2 * lens))

    effective_size = max(width, height) * margin_factor
    required_distance = effective_size / (2 * math.tan(fov_y / 2))

    min_distance = max_dimension * 1.5
    required_distance = max(required_distance, min_distance)

    return object_center_world, required_distance
