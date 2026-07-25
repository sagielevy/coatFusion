import bpy
from mathutils import Vector
from typing import Any, Optional
from dataclasses import dataclass
from .geometry_utils import calculate_required_distance_from_object


@dataclass
class CameraState:
    """Store camera state for restoration"""
    location: Vector
    rotation_euler: tuple
    rotation_quaternion: tuple
    rotation_mode: str
    scale: Vector
    lens: float
    sensor_width: float
    sensor_height: float
    clip_start: float
    clip_end: float
    target_location: Optional[Vector] = None
    constraint_target: Optional[Any] = None


def store_camera_state(camera) -> CameraState:
    """Store the current camera state for restoration"""
    return CameraState(
        location=camera.location.copy(),
        rotation_euler=camera.rotation_euler.copy(),
        rotation_quaternion=camera.rotation_quaternion.copy(),
        rotation_mode=camera.rotation_mode,
        scale=camera.scale.copy(),
        lens=camera.data.lens,
        sensor_width=camera.data.sensor_width,
        sensor_height=camera.data.sensor_height,
        clip_start=camera.data.clip_start,
        clip_end=camera.data.clip_end
    )


def restore_camera_state(camera, state: CameraState) -> None:
    """Restore the camera and its constraint target to their previous states"""
    camera.location = state.location
    camera.rotation_euler = state.rotation_euler
    camera.rotation_quaternion = state.rotation_quaternion
    camera.rotation_mode = state.rotation_mode
    camera.scale = state.scale
    camera.data.lens = state.lens
    camera.data.sensor_width = state.sensor_width
    camera.data.sensor_height = state.sensor_height
    camera.data.clip_start = state.clip_start
    camera.data.clip_end = state.clip_end
    
    if state.target_location is not None and state.constraint_target is not None:
        state.constraint_target.location = state.target_location



def frame_object_in_camera(logger, camera, obj, margin_factor: float) -> CameraState:
    """Frame the object in the camera view using the camera's Track To constraint target"""
    original_state = store_camera_state(camera)
    
    track_to_constraint = None
    constraint_target = None
    
    for constraint in camera.constraints:
        if constraint.type == 'TRACK_TO':
            track_to_constraint = constraint
            constraint_target = constraint.target
            break
    
    if not track_to_constraint or not constraint_target:
        logger.info("Warning: Camera does not have a Track To constraint with a target")
        return original_state
    
    original_target_location = constraint_target.location.copy()
    
    bpy.context.view_layer.update()
    
    try:
        object_center_world, required_distance = calculate_required_distance_from_object(obj, camera, margin_factor)
    except ValueError as e:
        logger.info(f"Warning: {e}")
        return original_state
    
    constraint_target.location = object_center_world
    
    camera_direction = (camera.location - object_center_world).normalized()
    new_camera_location = object_center_world + (camera_direction * required_distance)
    camera.location = new_camera_location
    
    bpy.context.view_layer.update()
    
    logger.info(
        f"Framed object: target at {object_center_world}, camera at {camera.location}, distance: {required_distance}")
    
    enhanced_state = CameraState(
        location=original_state.location,
        rotation_euler=original_state.rotation_euler,
        rotation_quaternion=original_state.rotation_quaternion,
        rotation_mode=original_state.rotation_mode,
        scale=original_state.scale,
        lens=original_state.lens,
        sensor_width=original_state.sensor_width,
        sensor_height=original_state.sensor_height,
        clip_start=original_state.clip_start,
        clip_end=original_state.clip_end,
        target_location=original_target_location,
        constraint_target=constraint_target
    )
    
    return enhanced_state