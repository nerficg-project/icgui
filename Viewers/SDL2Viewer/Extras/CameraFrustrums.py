# -- coding: utf-8 --

"""SDL2Viewer/Extras/CameraFrustrum.py: Camera frustrum overlay showing dataset poses in the scene."""

from typing import Any

import imgui
import numpy as np
import OpenGL.GL as gl
from OpenGL.arrays.vbo import VBO
from OpenGL.GL.shaders import ShaderProgram

from .Base import BaseGLExtra
from ..Shaders import (getPointShader, getFrustrumShader,
                       bindShaderProgram, bindTextures, bindVertexBuffer, bindVertexAttributes,
                       fillUniforms)

from Cameras.Base import BaseCamera


class CameraFrustrums(BaseGLExtra):
    """OpenGL component that renders camera frustrums into the scene."""
    def __init__(self, initial_scale: float = 0.1):
        super().__init__()
        self._point_shader_program: ShaderProgram = getPointShader()
        self._point_vao_location: int = gl.glGenVertexArrays(1)
        self._line_shader_program: ShaderProgram = getFrustrumShader()
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
                'frustrumScale': (gl.glGetUniformLocation(self._line_shader_program,
                                                          'frustrumScale'), 'float', tuple()),
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

        self._point_size: float = 0.005
        self._line_width: float = 0.0025
        self._frustrum_scale: float = initial_scale
        self._color = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self._highlight: bool = False
        self._highlight_color = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)

    @property
    def name(self) -> str:
        """Returns the name of the extra."""
        return 'Camera Frustrums'

    @staticmethod
    def _generateCameraPoints(nerf_config: 'NeRFConfig', split: str) -> tuple[VBO, int]:
        camera_poses = nerf_config.dataset_poses[split]
        vertices = []
        for camera_pose in camera_poses:
            # Each point consists of 6 vertices (to form a billboard quad)
            vertices.append(camera_pose.T)
            vertices.append(camera_pose.T)
            vertices.append(camera_pose.T)
            vertices.append(camera_pose.T)
            vertices.append(camera_pose.T)
            vertices.append(camera_pose.T)

        vertices = np.array(vertices, dtype=np.float32)
        vertex_count = len(vertices)
        vbo = VBO(vertices, gl.GL_STATIC_DRAW)
        return vbo, vertex_count

    @staticmethod
    def _generateCameraLines(nerf_config: 'NeRFConfig', split: str) -> tuple[VBO, VBO, int]:
        camera_poses = nerf_config.dataset_poses[split]
        line_vertices = []
        line_indices = []
        for i, camera_pose in enumerate(camera_poses):
            x = (0.5 * camera_pose.width + camera_pose.principal_offset_x) / camera_pose.focal_x
            y = (0.5 * camera_pose.height + camera_pose.principal_offset_y) / camera_pose.focal_y

            vertices = [
                np.concatenate((np.array([ 0,  0,  0, 1], dtype=np.float32), camera_pose.c2w.T.flatten())),
                np.concatenate((np.array([-x, -y, -1, 1], dtype=np.float32), camera_pose.c2w.T.flatten())),
                np.concatenate((np.array([-x,  y, -1, 1], dtype=np.float32), camera_pose.c2w.T.flatten())),
                np.concatenate((np.array([ x, -y, -1, 1], dtype=np.float32), camera_pose.c2w.T.flatten())),
                np.concatenate((np.array([ x,  y, -1, 1], dtype=np.float32), camera_pose.c2w.T.flatten())),
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

    def _generateCameraFrustrums(self, nerf_config: 'NeRFConfig', split: str):
        """Generates the vertices for the camera frustrums."""
        self._point_vertices_vbos[split], self._point_vertex_counts[split] \
            = self._generateCameraPoints(nerf_config, split)
        self._line_vertices_vbos[split], self._line_index_vbos[split], self._line_vertex_counts[split] \
            = self._generateCameraLines(nerf_config, split)

    def renderOptions(self):
        """Renders imgui inputs for the options of the extra."""
        min_point_size, max_point_size = 1e-9, 0.1
        changed, point_size = imgui.drag_float(f'Point Size##{self.name}',
                                               self._point_size, 0.0001,
                                               min_point_size, max_point_size, format='%.5f')
        if changed:
            self._point_size = max(min_point_size, min(point_size, max_point_size))

        min_line_width, max_line_width = 1e-9, 0.1
        changed, line_width = imgui.drag_float(f'Line Width##{self.name}',
                                               self._line_width, 0.000025,
                                               min_line_width, max_line_width, format='%.5f')
        if changed:
            self._line_width = max(min_line_width, min(line_width, max_line_width))

        min_frustrum_scale, max_frustrum_scale = 1e-9, 100.0
        changed, frustrum_scale = imgui.drag_float(f'Frustrum Scale##{self.name}',
                                                   self._frustrum_scale, 0.0001,
                                                   min_frustrum_scale, max_frustrum_scale, format='%.5f')
        if changed:
            self._frustrum_scale = max(min_frustrum_scale, min(frustrum_scale, max_frustrum_scale))

        changed, color = imgui.color_edit4(f'Color##{self.name}', *self._color)
        if changed:
            self._color = np.array(color, dtype=np.float32)

        _, self._highlight = imgui.checkbox(f'Highlight Current Timestep##{self.name}', self._highlight)
        if self._highlight:
            changed, color = imgui.color_edit4(f'Highlight Color##{self.name}', *self._highlight_color)
            if changed:
                self._highlight_color = np.array(color, dtype=np.float32)

    def _renderPoints(self, split: str, uniform_values: dict[str, Any], model_tex_locations: dict[str, int]):
        with (
            bindShaderProgram(self._point_shader_program),
            bindTextures(model_tex_locations['depth'], model_tex_locations['alpha']),
            bindVertexBuffer(self._point_vao_location, self._point_vertices_vbos[split]),
            bindVertexAttributes(3, gl.GL_FLOAT, 0)
        ):
            fillUniforms(self._uniform_locations['point'], uniform_values)

            gl.glDrawArrays(gl.GL_TRIANGLES, 0, self._point_vertex_counts[split])

    def _renderLines(self, split: str, uniform_values: dict[str, Any], model_tex_locations: dict[str, int]):
        with (
            bindShaderProgram(self._line_shader_program),
            bindTextures(model_tex_locations['depth'], model_tex_locations['alpha']),
            bindVertexBuffer(self._line_vao_location, self._line_vertices_vbos[split], self._line_index_vbos[split]),
            bindVertexAttributes([4, 4, 4, 4, 4], gl.GL_FLOAT, 80, [0, 4*4, 4*4*2, 4*4*3, 4*4*4])
        ):
            fillUniforms(self._uniform_locations['line'], uniform_values)

            gl.glDrawElements(gl.GL_LINES, self._line_vertex_counts[split], gl.GL_UNSIGNED_INT, None)

    def render(self, camera: BaseCamera, extra_params: dict[str, Any]):
        """Renders the extra into the scene."""
        nerf_config: 'NeRFConfig' = extra_params['nerfconfig']
        split = nerf_config.camera.dataset_split

        # Calculate model view projection matrices
        uniform_values = {
            'view': camera.properties.w2c.cpu().numpy().astype('f4').copy(order='C'),
            'projection': camera.getProjectionMatrix(invert_z=True).cpu().numpy().astype('f4').copy(order='C'),
            'viewportSize': np.array(extra_params['viewportSize'], dtype='f4'),
            'near': camera.near_plane,
            'far': camera.far_plane,
            'pointSize': self._point_size,
            'lineWidth': self._line_width,
            'frustrumScale': self._frustrum_scale,
            'color': self._color,
            'highlightColor': self._highlight_color,
            'highlightedCamera': nerf_config.camera.cam_idx_current_timestamp if self._highlight else -1,
        }

        if split not in self._point_vertices_vbos:
            self._generateCameraFrustrums(nerf_config, split)

        self._renderPoints(split, uniform_values, extra_params['modelTextureLocations'])
        self._renderLines(split, uniform_values, extra_params['modelTextureLocations'])
