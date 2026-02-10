"""ModelViewer/Overlays/UnitCube.py: Unit Cube overlay for the model viewer."""

from typing import Any

import numpy as np
import OpenGL.GL as gl
from imgui_bundle import imgui
from OpenGL.arrays.vbo import VBO
from OpenGL.GL.shaders import ShaderProgram

from .Base import BaseOverlay
from Cameras.Perspective import PerspectiveCamera
from Datasets.utils import View
from ICGui.ModelViewer.Shaders import get_line_shader
from ICGui.ModelViewer.Shaders.ContextManagers import bind_shader_program, bind_textures, bind_vertex_buffer, bind_vertex_attributes
from ICGui.ModelViewer.Shaders.Geometry import get_cube_wireframe
from ICGui.ModelViewer.Shaders.utils import fill_uniforms
from ICGui.State.Volatile import GlobalState
from Logging import Logger


class UnitCube(BaseOverlay):
    """OpenGL component that renders a unit cube into the scene."""
    def __init__(self):
        super().__init__()
        self._wireframe_shader_program: ShaderProgram = get_line_shader()
        self._cube_vao_location: int = gl.glGenVertexArrays(1)

        self._cube_vertices_vbo: VBO
        self._cube_indices_vbo: VBO
        self._cube_vertices_vbo, self._cube_indices_vbo = get_cube_wireframe()

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

        self._line_width = 0.005
        self._color = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    @property
    def name(self) -> str:
        """Returns the name of the extra."""
        return 'Unit Cube'

    def render_options(self):
        """Renders imgui inputs for the options of the extra."""
        # Line Width
        min_line_width, max_line_width = 1e-9, 0.1
        changed, line_width = imgui.drag_float(f'Line Width##{self.name}',
                                               self._line_width, 0.000025,
                                               min_line_width, max_line_width, format='%.5f')
        if changed:
            self._line_width = max(min_line_width, min(line_width, max_line_width))

        # Color
        changed, color = imgui.color_edit4(f'Color##{self.name}', self._color.tolist())
        if changed:
            self._color = np.array(color, dtype=np.float32)

    def render(self, view: View, extra_params: dict[str, Any]):
        """Renders the extra into the scene."""
        camera = view.camera
        if not isinstance(camera, PerspectiveCamera):
            Logger.log_warning(f'Unit cube can only be rendered with a PerspectiveCamera, got {type(camera)}')
            return

        # Get model transform from input manager
        model_transform = GlobalState().input_manager.control_scheme.model_transform

        # Set uniforms
        model_tex_locations = extra_params['modelTextureLocations']
        uniform_values = {
            'view': (model_transform @ view.w2c_numpy @ model_transform.T).astype(np.float32).copy(order='C'),
            'projection': camera.get_projection_matrix(invert_z=True).cpu().numpy().astype(np.float32).copy(order='C'),
            'viewportSize': np.array(extra_params['viewportSize'], dtype=np.float32),
            'near': camera.near_plane,
            'far': camera.far_plane,
            'lineWidth': self._line_width,
            'color': self._color
        }

        with (
            bind_shader_program(self._wireframe_shader_program),
            bind_textures(model_tex_locations['depth'], model_tex_locations['alpha']),
            bind_vertex_buffer(self._cube_vao_location, self._cube_vertices_vbo, self._cube_indices_vbo),
            bind_vertex_attributes(3, gl.GL_FLOAT, 0)
        ):
            fill_uniforms(self._uniform_locations, uniform_values)
            gl.glDrawElements(gl.GL_LINES, 24, gl.GL_UNSIGNED_INT, None)
