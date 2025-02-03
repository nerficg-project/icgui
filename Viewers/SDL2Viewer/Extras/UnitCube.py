# -- coding: utf-8 --

"""SDL2Viewer/Extras/UnitCube.py: Unit Cube overlayed on model output."""

from typing import Any

import imgui
import numpy as np
import OpenGL.GL as gl
from OpenGL.arrays.vbo import VBO
from OpenGL.GL.shaders import ShaderProgram

from .Base import BaseGLExtra
from ..Shaders import (getLineShader, getCubeWireframe,
                       bindShaderProgram, bindTextures, bindVertexBuffer, bindVertexAttributes,
                       fillUniforms)

from Cameras.Base import BaseCamera


class UnitCube(BaseGLExtra):
    """OpenGL component that renders a unit cube into the scene."""
    def __init__(self):
        super().__init__()
        self._wireframe_shader_program: ShaderProgram = getLineShader()
        self._cube_vao_location: int = gl.glGenVertexArrays(1)

        self._cube_vertices_vbo: VBO
        self._cube_indices_vbo: VBO
        self._cube_vertices_vbo, self._cube_indices_vbo = getCubeWireframe()

        self._uniform_locations: dict[str, tuple[int, str, tuple]] = {
            'view': (gl.glGetUniformLocation(self._wireframe_shader_program,
                                             'viewMatrix'), 'mat4', (True, )),
            'projection': (gl.glGetUniformLocation(self._wireframe_shader_program,
                                                   'projectionMatrix'), 'mat4', (True, )),
            'viewportSize': (gl.glGetUniformLocation(self._wireframe_shader_program, 'viewportSize'), 'vec2', tuple()),
            'near': (gl.glGetUniformLocation(self._wireframe_shader_program, 'near'), 'float', tuple()),
            'far': (gl.glGetUniformLocation(self._wireframe_shader_program, 'far'), 'float', tuple()),
            'lineWidth': (gl.glGetUniformLocation(self._wireframe_shader_program, 'lineWidth'), 'float', tuple()),
            'color': (gl.glGetUniformLocation(self._wireframe_shader_program, 'color'), 'vec4', tuple()),
        }

        self._line_width = 0.0025
        self._color = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    @property
    def name(self) -> str:
        """Returns the name of the extra."""
        return 'Unit Cube'

    def renderOptions(self):
        """Renders imgui inputs for the options of the extra."""
        min_line_width, max_line_width = 1e-9, 0.1
        changed, line_width = imgui.drag_float(f'Line Width##{self.name}',
                                               self._line_width, 0.000025,
                                               min_line_width, max_line_width, format='%.5f')
        if changed:
            self._line_width = max(min_line_width, min(line_width, max_line_width))

        changed, color = imgui.color_edit4(f'Color##{self.name}', *self._color)
        if changed:
            self._color = np.array(color, dtype=np.float32)

    def render(self, camera: BaseCamera, extra_params: dict[str, Any]):
        """Renders the extra into the scene."""
        model_tex_locations = extra_params['modelTextureLocations']
        uniform_values = {
            'view': camera.properties.w2c.cpu().numpy().astype('f4').copy(order='C'),
            'projection': camera.getProjectionMatrix(invert_z=True).cpu().numpy().astype('f4').copy(order='C'),
            'viewportSize': np.array(extra_params['viewportSize'], dtype='f4'),
            'near': camera.near_plane,
            'far': camera.far_plane,
            'lineWidth': self._line_width,
            'color': self._color
        }

        with (
            bindShaderProgram(self._wireframe_shader_program),
            bindTextures(model_tex_locations['depth'], model_tex_locations['alpha']),
            bindVertexBuffer(self._cube_vao_location, self._cube_vertices_vbo, self._cube_indices_vbo),
            bindVertexAttributes(3, gl.GL_FLOAT, 0)
        ):
            fillUniforms(self._uniform_locations, uniform_values)
            gl.glDrawElements(gl.GL_LINES, 24, gl.GL_UNSIGNED_INT, None)
