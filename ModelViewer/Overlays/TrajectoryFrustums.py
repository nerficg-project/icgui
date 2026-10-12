"""ModelViewer/Overlays/TrajectoryFrustum.py: Camera frustum overlay showing trajectory keyframe and interpolated poses in the scene."""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import OpenGL.GL as gl
from imgui_bundle import imgui
from OpenGL.arrays.vbo import VBO
from OpenGL.GL.shaders import ShaderProgram

from .Base import BaseOverlay
from Cameras.Equirectangular import EquirectangularCamera
from Cameras.Perspective import PerspectiveCamera
from Datasets.utils import View
from ICGui.Components.StyledToggle import styled_toggle
from ICGui.ModelViewer.Shaders import get_point_shader, get_frustum_shader, get_line_shader
from ICGui.ModelViewer.Shaders.ContextManagers import bind_shader_program, bind_textures, bind_vertex_buffer, bind_vertex_attributes
from ICGui.ModelViewer.Shaders.utils import fill_uniforms
from ICGui.State.Volatile import CameraState, GlobalState, OverlayState
from Logging import Logger


@dataclass
class RenderBuffers:
    point_vertices: VBO | None
    point_vertex_count: int | None
    line_vertices: VBO
    line_indices: VBO | None
    line_index_count: int


class InterpolatedFramesVisualization(Enum):
    NONE = 0
    FRUSTUMS = 1
    LINES = 2


class TrajectoryFrustums(BaseOverlay):
    """OpenGL component that renders trajectory frustums into the scene."""
    def __init__(self, initial_scale: float = 0.1):
        super().__init__()
        self._point_shader_program: ShaderProgram = get_point_shader()
        self._frustum_shader_program: ShaderProgram = get_frustum_shader()
        self._line_shader_program: ShaderProgram = get_line_shader()
        self._vao_location: int = gl.glGenVertexArrays(1)

        self._keyframe_buffers: RenderBuffers | None = None
        self._keyframe_highlight_buffers: RenderBuffers | None = None
        self._interpolated_buffers: RenderBuffers | None = None
        self._interpolated_line_buffers: RenderBuffers | None = None
        self._vis_version = -1  # Counter to identify changes in the visualization state

        # gl uniform functions don't accept their arguments as kwargs, so specify additional ones (e.g. transpose) in order
        self._uniform_locations: dict[str, dict[str, tuple[int, str, tuple]]] = {
            'point': {
                'view': (gl.glGetUniformLocation(self._point_shader_program, 'view'), 'mat4', (True, )),  # transpose=True
                'projection': (gl.glGetUniformLocation(self._point_shader_program, 'projection'), 'mat4', (True, )),  # transpose=True
                'pointSize': (gl.glGetUniformLocation(self._point_shader_program, 'pointSize'), 'float', tuple()),
                'viewportSize': (gl.glGetUniformLocation(self._point_shader_program, 'viewportSize'), 'vec2', tuple()),
                'near': (gl.glGetUniformLocation(self._point_shader_program, 'near'), 'float', tuple()),
                'far': (gl.glGetUniformLocation(self._point_shader_program, 'far'), 'float', tuple()),
                'color': (gl.glGetUniformLocation(self._point_shader_program, 'color'), 'vec4', tuple()),
            },
            'frustum': {
                'view': (gl.glGetUniformLocation(self._frustum_shader_program, 'viewMatrix'), 'mat4', (True,)),  # transpose=True
                'projection': (gl.glGetUniformLocation(self._frustum_shader_program, 'projectionMatrix'), 'mat4', (True,)),  # transpose=True
                'lineWidth': (gl.glGetUniformLocation(self._frustum_shader_program, 'lineWidth'), 'float', tuple()),
                'frustumScale': (gl.glGetUniformLocation(self._frustum_shader_program, 'frustumScale'), 'float', tuple()),
                'viewportSize': (gl.glGetUniformLocation(self._frustum_shader_program, 'viewportSize'), 'vec2', tuple()),
                'focalAspectScale': (gl.glGetUniformLocation(self._frustum_shader_program, 'focalAspectScale'), 'vec2', tuple()),
                'near': (gl.glGetUniformLocation(self._frustum_shader_program, 'near'), 'float', tuple()),
                'far': (gl.glGetUniformLocation(self._frustum_shader_program, 'far'), 'float', tuple()),
                'color': (gl.glGetUniformLocation(self._frustum_shader_program, 'color'), 'vec4', tuple()),
            },
            'line': {
                'view': (gl.glGetUniformLocation(self._line_shader_program, 'viewMatrix'), 'mat4', (True, )),  # transpose=True
                'projection': (gl.glGetUniformLocation(self._line_shader_program, 'projectionMatrix'), 'mat4', (True, )),  # transpose=True
                'lineWidth': (gl.glGetUniformLocation(self._line_shader_program, 'lineWidth'), 'float', tuple()),
                'frustumScale': (gl.glGetUniformLocation(self._line_shader_program, 'frustumScale'), 'float', tuple()),
                'viewportSize': (gl.glGetUniformLocation(self._line_shader_program, 'viewportSize'), 'vec2', tuple()),
                'focalAspectScale': (gl.glGetUniformLocation(self._line_shader_program, 'focalAspectScale'), 'vec2', tuple()),
                'near': (gl.glGetUniformLocation(self._line_shader_program, 'near'), 'float', tuple()),
                'far': (gl.glGetUniformLocation(self._line_shader_program, 'far'), 'float', tuple()),
                'color': (gl.glGetUniformLocation(self._line_shader_program, 'color'), 'vec4', tuple()),
            }
        }

        self._point_size: float = 0.005 * initial_scale
        self._frustum_line_width: float = 0.00125 * initial_scale
        self._frustum_scale: float = 0.1 * initial_scale
        self._interpolated_frames_vis = InterpolatedFramesVisualization.FRUSTUMS
        self._interpolated_line_width: float = 0.0005 * initial_scale
        self._color_keyframes = np.array([0.9, 0.9, 0.9, 1.0], dtype=np.float32)
        self._color_interpolated = np.array([0.6, 0.6, 0.8, 0.7], dtype=np.float32)
        self._highlight_color = np.array([1.0, 0.0, 0.0, 0.8], dtype=np.float32)
        self._highlight: bool = True

        self._warning_camera_type = None

    @property
    def name(self) -> str:
        """Returns the name of the extra"""
        return 'Trajectory Frustums'

    @property
    def help(self) -> str | None:
        return ('Renders camera frustums (and an optional trajectory path) for the current trajectory defined in the '
                '"Trajectory Renderer" section.')

    @staticmethod
    def _generate_frustum(poses: list[np.ndarray]) -> tuple[VBO, VBO, int]:
        # Each frustum is a rectangular pyramid: 5 vertices (camera center + 4 corners) and 8 edges
        model_transform = GlobalState().input_manager.control_scheme.model_transform

        # Put corner positions
        line_vertices = np.empty((5 * len(poses), 4 + 16), dtype=np.float32)  # 4 for corner position, 16 for the c2w matrix
        line_vertices[0::5, 0:4] = np.array([ 0,  0,  0, 1], dtype=np.float32)
        line_vertices[1::5, 0:4] = np.array([-1, -1, -1, 1], dtype=np.float32)
        line_vertices[2::5, 0:4] = np.array([-1,  1, -1, 1], dtype=np.float32)
        line_vertices[3::5, 0:4] = np.array([ 1, -1, -1, 1], dtype=np.float32)
        line_vertices[4::5, 0:4] = np.array([ 1,  1, -1, 1], dtype=np.float32)

        # Put edge indices (camera center to corners + between corners)
        indices = np.array([
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [2, 4],
            [4, 3],
            [3, 1],
        ], dtype=np.uint32).ravel()
        offsets = 5 * np.arange(len(poses), dtype=np.uint32)
        line_indices = (indices + offsets[:, None]).flatten()

        # Put c2w matrices
        for i, camera_pose in enumerate(poses):
            c2w = (model_transform @ camera_pose @ model_transform.T).T.ravel()
            line_vertices[i*5:(i+1)*5, 4:20] = c2w

        vertex_vbo = VBO(line_vertices, gl.GL_STATIC_DRAW)
        index_vbo = VBO(line_indices, target=gl.GL_ELEMENT_ARRAY_BUFFER)
        return vertex_vbo, index_vbo, len(line_indices)

    @staticmethod
    def _generate_trajectory_paths(frames: list[list[np.ndarray]], loop: bool) -> tuple[VBO | None, int]:
        model_transform = GlobalState().input_manager.control_scheme.model_transform[:3, :3]
        poses = [pose for frame in frames for pose in frame]
        if len(poses) < 2:
            return None, 0  # Not enough frames to form a trajectory
        if loop:
            poses.append(poses[0])  # Connect last frame to first frame for looping trajectory
        vertices = (np.stack(poses, axis=0)[..., :3, 3] @ model_transform.T).astype(np.float32)
        vertex_vbo = VBO(vertices, gl.GL_STATIC_DRAW)
        return vertex_vbo, len(vertices)

    def _generate_trajectory_buffers(self) -> None:
        model_transform = GlobalState().input_manager.control_scheme.model_transform[:3, :3]
        overlay_data = OverlayState().additional_overlay_data.get(self.name, {})
        vis_version = overlay_data.get('_vis_version', -1)
        if vis_version <= self._vis_version:
            return  # No new data available, skip regeneration
        self._vis_version = vis_version

        keyframe_poses: list[np.ndarray] = overlay_data.get('keyframe_poses', [])
        selected_keyframe_poses: list[np.ndarray] = overlay_data.get('selected_keyframe_poses', [])
        interpolated_poses: list[list[np.ndarray]] = overlay_data.get('interpolated_poses', [])
        loop: bool = overlay_data.get('does_loop', False)

        # Get camera positions and transform to model space
        keyframe_positions = (np.stack(keyframe_poses, axis=0)[..., :3, 3] @ model_transform.T).astype(np.float32) if len(keyframe_poses) > 0 else None
        selected_keyframe_positions = (np.stack(selected_keyframe_poses, axis=0)[..., :3, 3] @ model_transform.T).astype(np.float32) if len(selected_keyframe_poses) > 0 else None
        interpolated_positions = (np.concatenate(interpolated_poses, axis=0)[..., :3, 3] @ model_transform.T).astype(np.float32) if len(interpolated_poses) > 0 else None

        # Each point consists of 6 vertices (to form a billboard quad)
        keyframe_vertices = np.repeat(keyframe_positions, repeats=6, axis=0) if keyframe_positions is not None else None
        selected_keyframe_vertices = np.repeat(selected_keyframe_positions, repeats=6, axis=0) if selected_keyframe_positions is not None else None
        interpolated_vertices = np.repeat(interpolated_positions, repeats=6, axis=0) if interpolated_positions is not None else None
        keyframe_vbo = VBO(keyframe_vertices, gl.GL_STATIC_DRAW) if keyframe_vertices is not None else None
        selected_keyframe_vbo = VBO(selected_keyframe_vertices, gl.GL_STATIC_DRAW) if selected_keyframe_vertices is not None else None
        interpolated_vbo = VBO(interpolated_vertices, gl.GL_STATIC_DRAW) if interpolated_vertices is not None else None

        # Generate camera frustums
        keyframe_line_vertices, keyframe_line_indices, keyframe_line_count = self._generate_frustum(keyframe_poses) if len(keyframe_poses) > 0 else (None, None, 0)
        selected_line_vertices, selected_line_indices, selected_line_count = self._generate_frustum(selected_keyframe_poses) if len(selected_keyframe_poses) > 0 else (None, None, 0)
        interpolated_poses_flattened = [pose for pose_set in interpolated_poses for pose in pose_set[1:]] # Flatten, and remove first pose (which is the initial keyframe)
        interpolated_line_vertices, interpolated_line_indices, interpolated_line_count = self._generate_frustum(interpolated_poses_flattened) if len(interpolated_poses_flattened) > 0 else (None, None, 0)

        # Generate trajectory line strip
        trajectory_line_vertices, trajectory_line_count = self._generate_trajectory_paths(interpolated_poses, loop=loop)

        # Store buffers and counts
        self._keyframe_buffers = RenderBuffers(
            point_vertices=keyframe_vbo,
            point_vertex_count=len(keyframe_vertices) if keyframe_vertices is not None else 0,
            line_vertices=keyframe_line_vertices,
            line_indices=keyframe_line_indices,
            line_index_count=keyframe_line_count,
        )
        self._keyframe_highlight_buffers = RenderBuffers(
            point_vertices=selected_keyframe_vbo,
            point_vertex_count=len(selected_keyframe_vertices) if selected_keyframe_vertices is not None else 0,
            line_vertices=selected_line_vertices,
            line_indices=selected_line_indices,
            line_index_count=selected_line_count,
        )
        self._interpolated_buffers = RenderBuffers(
            point_vertices=interpolated_vbo,
            point_vertex_count=len(interpolated_vertices) if interpolated_vertices is not None else 0,
            line_vertices=interpolated_line_vertices,
            line_indices=interpolated_line_indices,
            line_index_count=interpolated_line_count,
        )
        self._interpolated_line_buffers = RenderBuffers(
            line_vertices=trajectory_line_vertices,
            line_index_count=trajectory_line_count,
            point_vertices=None,
            point_vertex_count=None,
            line_indices=None,
        )

    def render_options(self):
        """Renders imgui inputs for the options of the extra."""
        # Point Size
        min_point_size, max_point_size = 1e-9, 0.1
        changed, point_size = imgui.drag_float(
            f'Point Size##{self.name}', self._point_size,
            v_speed=0.0001, v_min=min_point_size, v_max=max_point_size, format='%.5f',
        )
        if changed:
            self._point_size = max(min_point_size, min(point_size, max_point_size))

        # Line Width
        min_line_width, max_line_width = 1e-9, 0.1
        changed, line_width = imgui.drag_float(
            f'Frustum Line Width##{self.name}', self._frustum_line_width,
            v_speed=0.000025, v_min=min_line_width, v_max=max_line_width, format='%.5f',
        )
        if changed:
            self._frustum_line_width = max(min_line_width, min(line_width, max_line_width))

        # Frustum scaling factor
        min_frustum_scale, max_frustum_scale = 1e-9, 100.0
        changed, frustum_scale = imgui.drag_float(
            f'Frustum Scale##{self.name}', self._frustum_scale,
            v_speed=0.0001, v_min=min_frustum_scale, v_max=max_frustum_scale, format='%.5f',
        )
        if changed:
            self._frustum_scale = max(min_frustum_scale, min(frustum_scale, max_frustum_scale))

        # Interpolated frames visualization mode
        changed, vis_mode = imgui.combo(
            f'Interpolated Frames Visualization##{self.name}',
            self._interpolated_frames_vis.value,
            ['None', 'Frustums', 'Lines']
        )
        if changed:
            self._interpolated_frames_vis = InterpolatedFramesVisualization(vis_mode)

        if self._interpolated_frames_vis == InterpolatedFramesVisualization.LINES:
            changed, line_width = imgui.drag_float(
                f'Trajectory Line Width##{self.name}', self._interpolated_line_width,
                v_speed=0.000025, v_min=min_line_width, v_max=max_line_width, format='%.5f',
            )
            if changed:
                self._interpolated_line_width = max(min_line_width, min(line_width, max_line_width))

        # Color and highlight options
        changed, color = imgui.color_edit4(f'Keyframe Color##{self.name}', self._color_keyframes.tolist())
        if changed:
            self._color_keyframes = np.array(color, dtype=np.float32)

        if self._interpolated_frames_vis != InterpolatedFramesVisualization.NONE:
            label = (f'{"Interpolated Frustums" if self._interpolated_frames_vis == InterpolatedFramesVisualization.FRUSTUMS else "Trajectory Line"}'
                     f' Color##{self.name}')
            changed, color = imgui.color_edit4(label, self._color_interpolated.tolist())
            if changed:
                self._color_interpolated = np.array(color, dtype=np.float32)

        _, self._highlight = styled_toggle(f'Highlight Selected Keyframe##{self.name}', self._highlight)
        if self._highlight:
            changed, color = imgui.color_edit4(f'Highlight Color##{self.name}', self._highlight_color.tolist())
            if changed:
                self._highlight_color = np.array(color, dtype=np.float32)

    def _render_points(self, vertices: VBO, count: int, uniform_values: dict[str, Any], model_tex_locations: dict[str, int]):
        with (
            bind_shader_program(self._point_shader_program),
            bind_textures(model_tex_locations['depth'], model_tex_locations['alpha']),
            bind_vertex_buffer(self._vao_location, vertices),
            bind_vertex_attributes(3, gl.GL_FLOAT, 12)
        ):
            fill_uniforms(self._uniform_locations['point'], uniform_values)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, count)

    def _render_line_strip(self, vertices: VBO, count: int, uniform_values: dict[str, Any], model_tex_locations: dict[str, int]):
        with (
            bind_shader_program(self._line_shader_program),
            bind_textures(model_tex_locations['depth'], model_tex_locations['alpha']),
            bind_vertex_buffer(self._vao_location, vertices),
            bind_vertex_attributes(3, gl.GL_FLOAT, 12)
        ):
            fill_uniforms(self._uniform_locations['line'], uniform_values)
            gl.glDrawArrays(gl.GL_LINE_STRIP, 0, count)

    def _render_frustum_lines(self, vertices: VBO, indices: VBO, count: int, uniform_values: dict[str, Any], model_tex_locations: dict[str, int]):
        with (
            bind_shader_program(self._frustum_shader_program),
            bind_textures(model_tex_locations['depth'], model_tex_locations['alpha']),
            bind_vertex_buffer(self._vao_location, vertices, indices),
            bind_vertex_attributes([4, 4, 4, 4, 4], gl.GL_FLOAT, 80, [0, 16, 32, 48, 64])
        ):
            fill_uniforms(self._uniform_locations['frustum'], uniform_values)
            gl.glDrawElements(gl.GL_LINES, count, gl.GL_UNSIGNED_INT, None)

    def render(self, view: View, extra_params: dict[str, Any]):
        """Renders the extra into the scene."""
        # Regenerate buffers if trajectory has changed
        self._generate_trajectory_buffers()

        # Set uniforms
        focal_aspect_scale = np.empty(2, dtype=np.float32)
        model_transform = GlobalState().input_manager.control_scheme.model_transform
        camera = CameraState().current_camera
        if isinstance(camera, PerspectiveCamera):
            focal_aspect_scale[0] = camera.center_x / camera.focal_x
            focal_aspect_scale[1] = camera.center_y / camera.focal_y
        elif isinstance(camera, EquirectangularCamera):
            focal_aspect_scale[:] = math.pi
        else:
            if camera.__class__.__name__ != self._warning_camera_type:
                self._warning_camera_type = camera.__class__.__name__
                Logger.log_warning(f'Frustum visualizations will break for any camera except PerspectiveCamera and EquirectangularCamera'
                                   f', got {camera.__class__.__name__}')
            focal_aspect_scale[:] = 1.0
        uniform_values = {
            'view': (model_transform @ view.w2c_numpy.astype(np.float32) @ model_transform.T).copy(order='C'),
            'projection': camera.get_projection_matrix(invert_z=True).cpu().numpy().astype(np.float32).copy(order='C'),
            'viewportSize': np.array(extra_params['viewportSize'], dtype=np.float32),
            'near': camera.near_plane,
            'far': camera.far_plane,
            'pointSize': self._point_size,
            'lineWidth': self._frustum_line_width,
            'frustumScale': self._frustum_scale,
            'focalAspectScale': focal_aspect_scale,
            'color': self._color_interpolated,
        }

        # Enable depth testing; this causes transparency artifacts especially due to us rendering lines as quads,
        # but is likely worth it to ensure better spatial understanding. Rendering order along with the choice of
        # depth function minimizes artifacts especially on important items such as keyframes.
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)

        # Render interpolated frames first, so that keyframes are on top
        if self._interpolated_frames_vis == InterpolatedFramesVisualization.FRUSTUMS:
            uniform_values['lineWidth'] = self._frustum_line_width
            # Render interpolated frustums if available
            if self._interpolated_buffers is not None and self._interpolated_buffers.point_vertices is not None:
                self._render_points(self._interpolated_buffers.point_vertices, self._interpolated_buffers.point_vertex_count, uniform_values, extra_params['modelTextureLocations'])
                self._render_frustum_lines(self._interpolated_buffers.line_vertices, self._interpolated_buffers.line_indices, self._interpolated_buffers.line_index_count, uniform_values, extra_params['modelTextureLocations'])

        elif self._interpolated_frames_vis == InterpolatedFramesVisualization.LINES:
            uniform_values['lineWidth'] = self._interpolated_line_width
            # Render trajectory line strip if available
            if self._interpolated_line_buffers is not None and self._interpolated_line_buffers.line_vertices is not None:
                self._render_line_strip(self._interpolated_line_buffers.line_vertices, self._interpolated_line_buffers.line_index_count, uniform_values, extra_params['modelTextureLocations'])

        # Render keyframes if available
        if self._keyframe_buffers is not None and self._keyframe_buffers.point_vertices is not None:
            uniform_values['color'] = self._color_keyframes
            uniform_values['lineWidth'] = self._frustum_line_width
            self._render_points(self._keyframe_buffers.point_vertices, self._keyframe_buffers.point_vertex_count, uniform_values, extra_params['modelTextureLocations'])
            self._render_frustum_lines(self._keyframe_buffers.line_vertices, self._keyframe_buffers.line_indices, self._keyframe_buffers.line_index_count, uniform_values, extra_params['modelTextureLocations'])

        # Render highlighted keyframe if available
        if self._keyframe_highlight_buffers is not None and self._keyframe_highlight_buffers.point_vertices is not None:
            uniform_values['color'] = self._color_keyframes
            if self._highlight:
                # Alpha blend color with highlight color
                uniform_values['color'] = (
                        self._highlight_color * self._highlight_color[3]
                        + self._color_keyframes * (1 - self._highlight_color[3])
                )
                uniform_values['color'][3] = self._highlight_color[3] + self._color_keyframes[3] * (1 - self._highlight_color[3])
            self._render_points(self._keyframe_highlight_buffers.point_vertices, self._keyframe_highlight_buffers.point_vertex_count, uniform_values, extra_params['modelTextureLocations'])
            self._render_frustum_lines(self._keyframe_highlight_buffers.line_vertices, self._keyframe_highlight_buffers.line_indices, self._keyframe_highlight_buffers.line_index_count, uniform_values, extra_params['modelTextureLocations'])

        gl.glDisable(gl.GL_DEPTH_TEST)