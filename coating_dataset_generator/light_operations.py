import bpy
import random
import math
from mathutils import Vector, Euler
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .geometry_utils import calculate_required_distance_from_object


@dataclass
class LightState:
    """Store light state for restoration"""
    location: Vector
    rotation_euler: tuple
    energy: float
    size: float


def store_light_state(light) -> LightState:
    """Store the current light state for restoration"""
    return LightState(
        location=light.location.copy(),
        rotation_euler=light.rotation_euler.copy(),
        energy=light.data.energy,
        size=light.data.size if hasattr(light.data, 'size') else 0.0
    )


def restore_light_state(light, state: LightState) -> None:
    """Restore the light to its previous state"""
    light.location = state.location
    light.rotation_euler = state.rotation_euler
    light.data.energy = state.energy
    if hasattr(light.data, 'size'):
        light.data.size = state.size


def get_area_lights() -> List[bpy.types.Object]:
    """Get all area lights in the scene that are enabled for rendering"""
    lights = []
    for obj in bpy.context.scene.objects:
        if (obj.type == 'LIGHT' and 
            obj.data.type == 'AREA' and 
            not obj.hide_render):
            lights.append(obj)
    return lights


def validate_lights() -> List[bpy.types.Object]:
    """Validate that exactly 2 area lights exist in the scene"""
    lights = get_area_lights()
    if len(lights) != 2:
        raise RuntimeError(f"Expected exactly 2 area lights in scene, found {len(lights)}")
    return lights


def configure_lights_for_scene(logger, camera, obj, lights: List[bpy.types.Object], config) -> Dict[str, LightState]:
    """Configure lights for the current scene based on camera position and object bounds"""
    if len(lights) != 2:
        raise ValueError("Expected exactly 2 lights")
    
    # Store original states
    original_states = {}
    for i, light in enumerate(lights):
        original_states[f'light_{i}'] = store_light_state(light)
    
    try:
        # Get object center and required distance for framing
        object_center_world, required_distance = calculate_required_distance_from_object(obj, camera, 1.2)
    except ValueError as e:
        logger.info(f"Warning: Could not calculate object bounds for light positioning: {e}")
        return original_states
    
    # Normalize camera position relative to object center
    camera_direction = (camera.location - object_center_world).normalized()
    
    # Generate random angles for light positioning using config values
    angle1 = random.uniform(-config.light_angle1_range, config.light_angle1_range)  # degrees, angle from camera in XY plane
    angle2 = random.uniform(-config.light_angle2_range, config.light_angle2_range)  # degrees, angle in Z plane (up/down)
    
    # Convert to radians
    angle1_rad = math.radians(angle1)
    angle2_rad = math.radians(angle2)
    
    # Position first light
    light1 = lights[0]
    
    # Create rotation matrix for XY plane offset
    xy_rotation = Vector((
        camera_direction.x * math.cos(angle1_rad) - camera_direction.y * math.sin(angle1_rad),
        camera_direction.x * math.sin(angle1_rad) + camera_direction.y * math.cos(angle1_rad),
        camera_direction.z
    )).normalized()
    
    # Apply Z-axis (up/down) rotation
    light1_direction = Vector((
        xy_rotation.x * math.cos(angle2_rad),
        xy_rotation.y * math.cos(angle2_rad),
        xy_rotation.z * math.cos(angle2_rad) + math.sin(angle2_rad)
    )).normalized()
    
    # Set light1 position at required distance from object center
    light1.location = object_center_world + (light1_direction * required_distance)

    # Position second light 90 degrees away from first light, rotated on Z axis
    light2 = lights[1]
    
    # Rotate light1's direction 90 degrees around Z axis
    z_rotation_rad = math.radians(90)
    light2_direction = Vector((
        light1_direction.x * math.cos(z_rotation_rad) - light1_direction.y * math.sin(z_rotation_rad),
        light1_direction.x * math.sin(z_rotation_rad) + light1_direction.y * math.cos(z_rotation_rad),
        light1_direction.z
    )).normalized()
    
    # Set light2 position
    light2.location = object_center_world + (light2_direction * required_distance)

    # Randomize light energy using config values
    for light in lights:
        light.data.energy = random.uniform(config.light_energy_min, config.light_energy_max)
    
    logger.info(f"Configured lights: Light1 at {light1.location}, Light2 at {light2.location}, distance: {required_distance}")
    
    return original_states


def restore_lights(lights: List[bpy.types.Object], original_states: Dict[str, LightState]) -> None:
    """Restore all lights to their original states"""
    for i, light in enumerate(lights):
        state_key = f'light_{i}'
        if state_key in original_states:
            restore_light_state(light, original_states[state_key])