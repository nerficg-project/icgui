"""ModelViewer/Overlays/CameraFrustum.py: Camera frustum overlay showing dataset poses in the scene."""

from typing import Any
import math

import numpy as np
import OpenGL.GL as gl
from imgui_bundle import imgui
from OpenGL.arrays.vbo import VBO
from OpenGL.GL.shaders import ShaderProgram

from Cameras.Equirectangular import EquirectangularCamera
from .Base import BaseOverlay
from Cameras.Perspective import PerspectiveCamera
from Datasets.utils import View
from ICGui.ModelViewer.Shaders import get_point_shader, get_frustum_shader
from ICGui.ModelViewer.Shaders.ContextManagers import bind_shader_program, bind_textures, bind_vertex_buffer, bind_vertex_attributes
from ICGui.ModelViewer.Shaders.utils import fill_uniforms
from ICGui.State.Volatile import CameraState, GlobalState
from Logging import Logger


class CameraFrustums(BaseOverlay):
    """OpenGL component that renders camera frustums into the scene."""
    def __init__(self, initial_scale: float = 0.1):
        super().__init__()
        self._point_shader_program: ShaderProgram = get_point_shader()
        self._point_vao_location: int = gl.glGenVertexArrays(1)
        self._line_shader_program: ShaderProgram = get_frustum_shader()
        self._line_vao_location: int = gl.glGenVertexArrays(1)

        self._point_vertices_vbos: dict[str, VBO] = {}
        self._point_vertex_counts: dict[str, int] = {}
        self._line_vertices_vbos: dict[str, VBO] = {}
        self._line_index_vbos: dict[str, VBO] = {}
        self._line_vertex_counts: dict[str, int] = {}

        # gl uniform functions don't accept their arguments as kwargs, so specify additional ones in order
        self._uniform_locations: dict[str, dict[str, tuple[int, str, tuple]]] = {
            'point': {
                'view': (gl.glGetUniformLocation(self._point_shader_program,
                                                 'view'), 'mat4', (True, )),  # transpose=True
                'projection': (gl.glGetUniformLocation(self._point_shader_program,
                                                       'projection'), 'mat4', (True, )),  # transpose=True
                'pointSize': (gl.glGetUniformLocation(self._point_shader_program, 'pointSize'), 'float', tuple()),
                'viewportSize': (gl.glGetUniformLocation(self._point_shader_program, 'viewportSize'), 'vec2', tuple()),
                'near': (gl.glGetUniformLocation(self._point_shader_program, 'near'), 'float', tuple()),
                'far': (gl.glGetUniformLocation(self._point_shader_program, 'far'), 'float', tuple()),
                'color': (gl.glGetUniformLocation(self._point_shader_program, 'color'), 'vec4', tuple()),
            },
            'line': {
                'view': (gl.glGetUniformLocation(self._line_shader_program,
                                                 'viewMatrix'), 'mat4', (True, )),  # transpose=True
                'projection': (gl.glGetUniformLocation(self._line_shader_program,
                                                       'projectionMatrix'), 'mat4', (True, )),  # transpose=True
                'lineWidth': (gl.glGetUniformLocation(self._line_shader_program, 'lineWidth'), 'float', tuple()),
                'frustumScale': (gl.glGetUniformLocation(self._line_shader_program,
                                                         'frustumScale'), 'float', tuple()),
                'viewportSize': (gl.glGetUniformLocation(self._line_shader_program, 'viewportSize'), 'vec2', tuple()),
                'near': (gl.glGetUniformLocation(self._line_shader_program, 'near'), 'float', tuple()),
                'far': (gl.glGetUniformLocation(self._line_shader_program, 'far'), 'float', tuple()),
                'color': (gl.glGetUniformLocation(self._line_shader_program, 'color'), 'vec4', tuple()),
                'highlightColor': (gl.glGetUniformLocation(self._line_shader_program,
                                                           'highlightColor'), 'vec4', tuple()),
                'highlightedCamera': (gl.glGetUniformLocation(self._line_shader_program,
                                                              'highlightedCamera'), 'int', tuple()),
            }
        }

        self._point_size: float = 0.005 * initial_scale
        self._line_width: float = 0.00125 * initial_scale
        self._frustum_scale: float = 0.1 * initial_scale
        self._color = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self._highlight: bool = False
        self._highlight_color = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)

    @property
    def name(self) -> str:
        """Returns the name of the extra."""
        return 'Camera Frustums'

    @staticmethod
    def _generate_camera_points(split: str) -> tuple[VBO, int]:
        # Get model transform from input manager
        model_transform = GlobalState().input_manager.control_scheme.model_transform[:3, :3]

        # Get camera positions and transform to model space
        camera_poses = CameraState().dataset_poses[split]
        positions = np.stack([pose.position_numpy for pose in camera_poses], axis=0)
        transformed = (positions @ model_transform.T).astype(np.float32)

        # Each point consists of 6 vertices (to form a billboard quad)
        vertices = np.repeat(transformed, repeats=6, axis=0)
        vertex_count = len(vertices)
        vbo = VBO(vertices, gl.GL_STATIC_DRAW)
        return vbo, vertex_count

    @staticmethod
    def _generate_camera_lines(split: str) -> tuple[VBO, VBO, int]:
        camera_poses = CameraState().dataset_poses[split]
        line_vertices = []
        line_indices = []
        warning_printed = False

        # Get model transform from input manager
        model_transform = GlobalState().input_manager.control_scheme.model_transform

        for i, camera_pose in enumerate(camera_poses):
            if isinstance(camera_pose.camera, PerspectiveCamera):
                x = camera_pose.camera.center_x / camera_pose.camera.focal_x
                y = camera_pose.camera.center_y / camera_pose.camera.focal_y
            elif isinstance(camera_pose.camera, EquirectangularCamera):
                x = y = math.pi
            else:
                if not warning_printed:
                    Logger.log_warning('Found unsupported camera in dataset, skipping rendering their frustums')
                    warning_printed = True
                continue

            c2w = (model_transform @ camera_pose.c2w_numpy @ model_transform.T).T.flatten()
            vertices = [
                np.concatenate((np.array([ 0,  0,  0, 1], dtype=np.float32), c2w)),
                np.concatenate((np.array([-x, -y, -1, 1], dtype=np.float32), c2w)),
                np.concatenate((np.array([-x,  y, -1, 1], dtype=np.float32), c2w)),
                np.concatenate((np.array([ x, -y, -1, 1], dtype=np.float32), c2w)),
                np.concatenate((np.array([ x,  y, -1, 1], dtype=np.float32), c2w)),
            ]
            indices = [
                [0, 1],
                [0, 2],
                [0, 3],
                [0, 4],
                [1, 2],
                [2, 4],
                [4, 3],
                [3, 1],
            ]

            line_indices.extend(np.array(indices, dtype=np.uint32).flatten() + i * len(vertices))
            line_vertices.extend(np.array(vertices).flatten())

        vertices = np.array(line_vertices, dtype=np.float32)
        index_count = len(line_indices)
        vertex_vbo = VBO(vertices, gl.GL_STATIC_DRAW)
        index_vbo = VBO(np.array(line_indices, dtype=np.uint32), target=gl.GL_ELEMENT_ARRAY_BUFFER)
        return vertex_vbo, index_vbo, index_count

    def _generate_camera_frustums(self, split: str):
        """Generates the vertices for the camera frustums."""
        self._point_vertices_vbos[split], self._point_vertex_counts[split] \
            = self._generate_camera_points(split)
        self._line_vertices_vbos[split], self._line_index_vbos[split], self._line_vertex_counts[split] \
            = self._generate_camera_lines(split)

    def render_options(self):
        """Renders imgui inputs for the options of the extra."""
        # Point Size
        min_point_size, max_point_size = 1e-9, 0.1
        changed, point_size = imgui.drag_float(
            f'Point Size##{self.name}', self._point_size,
            v_speed=0.0001, v_min=min_point_size, v_max=max_point_size, format='%.5f'
        )
        if changed:
            self._point_size = max(min_point_size, min(point_size, max_point_size))

        # Line Width
        min_line_width, max_line_width = 1e-9, 0.1
        changed, line_width = imgui.drag_float(
            f'Line Width##{self.name}', self._line_width,
            v_speed=0.000025, v_min=min_line_width, v_max=max_line_width, format='%.5f'
        )
        if changed:
            self._line_width = max(min_line_width, min(line_width, max_line_width))

        # Frustum scaling factor
        min_frustum_scale, max_frustum_scale = 1e-9, 100.0
        changed, frustum_scale = imgui.drag_float(
            f'Frustum Scale##{self.name}',
                                                   self._frustum_scale, 0.0001,
                                                   min_frustum_scale, max_frustum_scale, format='%.5f')
        if changed:
            self._frustum_scale = max(min_frustum_scale, min(frustum_scale, max_frustum_scale))

        # Color and highlight options
        changed, color = imgui.color_edit4(f'Color##{self.name}', self._color.tolist())
        if changed:
            self._color = np.array(color, dtype=np.float32)

        _, self._highlight = imgui.checkbox(f'Highlight Current Timestep##{self.name}', self._highlight)
        if self._highlight:
            changed, color = imgui.color_edit4(f'Highlight Color##{self.name}', self._highlight_color.tolist())
            if changed:
                self._highlight_color = np.array(color, dtype=np.float32)

    def _render_points(self, split: str, uniform_values: dict[str, Any], model_tex_locations: dict[str, int]):
        with (
            bind_shader_program(self._point_shader_program),
            bind_textures(model_tex_locations['depth'], model_tex_locations['alpha']),
            bind_vertex_buffer(self._point_vao_location, self._point_vertices_vbos[split]),
            bind_vertex_attributes(3, gl.GL_FLOAT, 0)
        ):
            fill_uniforms(self._uniform_locations['point'], uniform_values)

            gl.glDrawArrays(gl.GL_TRIANGLES, 0, self._point_vertex_counts[split])

    def _render_lines(self, split: str, uniform_values: dict[str, Any], model_tex_locations: dict[str, int]):
        with (
            bind_shader_program(self._line_shader_program),
            bind_textures(model_tex_locations['depth'], model_tex_locations['alpha']),
            bind_vertex_buffer(self._line_vao_location, self._line_vertices_vbos[split], self._line_index_vbos[split]),
            bind_vertex_attributes([4, 4, 4, 4, 4], gl.GL_FLOAT, 80, [0, 4 * 4, 4 * 4 * 2, 4 * 4 * 3, 4 * 4 * 4])
        ):
            fill_uniforms(self._uniform_locations['line'], uniform_values)

            gl.glDrawElements(gl.GL_LINES, self._line_vertex_counts[split], gl.GL_UNSIGNED_INT, None)

    def render(self, view: View, extra_params: dict[str, Any]):
        """Renders the extra into the scene."""
        split = CameraState().dataset_split
        camera = view.camera
        if not isinstance(camera, PerspectiveCamera):
            Logger.log_warning(f'Camera frustums can only be rendered with a PerspectiveCamera, got {type(camera)}')
            return

        # Get model transform from input manager
        model_transform = GlobalState().input_manager.control_scheme.model_transform

        # Set uniforms
        uniform_values = {
            'view': (model_transform @ view.w2c_numpy.astype(np.float32) @ model_transform.T).copy(order='C'),
            # TODO: Make a get_projection_matrix_numpy function to avoid unnecessary GPU up/download
            'projection': camera.get_projection_matrix(invert_z=True).cpu().numpy().astype(np.float32).copy(order='C'),
            'viewportSize': np.array(extra_params['viewportSize'], dtype=np.float32),
            'near': camera.near_plane,
            'far': camera.far_plane,
            'pointSize': self._point_size,
            'lineWidth': self._line_width,
            'frustumScale': self._frustum_scale,
            'color': self._color,
            'highlightColor': self._highlight_color,
            'highlightedCamera': CameraState().view_idx_current_timestamp if self._highlight else -1,
        }

        # Generate camera frustums if not already generated
        if split not in self._point_vertices_vbos:
            self._generate_camera_frustums(split)

        self._render_points(split, uniform_values, extra_params['modelTextureLocations'])
        self._render_lines(split, uniform_values, extra_params['modelTextureLocations'])
