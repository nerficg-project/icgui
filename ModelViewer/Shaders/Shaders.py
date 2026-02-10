"""ModelViewer/Shaders/Shaders.py: Provides the ShaderProgram helper class along with functions to easily load specific shaders."""

from pathlib import Path

import numpy as np

import OpenGL.GL as gl
from OpenGL.arrays import vbo
from OpenGL.GL import GL_VERTEX_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER
from OpenGL.GL.shaders import compileProgram, compileShader, ShaderProgram as GLShaderProgram


class ShaderProgram:
    """Provides utility functions related to OpenGL shaders."""
    SHADER_ROOT = Path(__file__).parent / 'ShaderSource'
    ShaderType = type[GL_VERTEX_SHADER | GL_GEOMETRY_SHADER | GL_FRAGMENT_SHADER]
    ShaderPath = str | Path

    SHADER_EXTENSIONS = {
        'vert': GL_VERTEX_SHADER,
        'geom': GL_GEOMETRY_SHADER,
        'frag': GL_FRAGMENT_SHADER
    }

    shader_cache: dict[tuple[ShaderPath, ...], GLShaderProgram] = {}

    @classmethod
    def determine_shader_type(cls, shader_path: ShaderPath):
        """Determines the shader type from the file extension."""
        shader_path = Path(shader_path)
        shader_extension = shader_path.suffix[1:]

        try:
            return cls.SHADER_EXTENSIONS[shader_extension]
        except KeyError as err:
            raise ValueError(f'Unknown shader extension "{shader_extension}" for shader "{shader_path}".') from err

    @classmethod
    def load_shader_from_file(cls, shader_type: ShaderType, shader_path: ShaderPath):
        """Loads a shader from a file and returns the compiled shader."""
        with open(cls.SHADER_ROOT / shader_path, 'r', encoding='utf-8') as shader_file:
            shader_source = shader_file.read()
        return compileShader(shader_source, shader_type)

    @classmethod
    def load_shader_program(cls, *shaders: ShaderPath) -> GLShaderProgram:
        """Loads a shader program from a list of shaders and returns the compiled shader program."""
        if shaders in cls.shader_cache:
            return cls.shader_cache[shaders]

        loaded_shaders = []
        for shader in shaders:
            loaded_shaders.append(cls.load_shader_from_file(cls.determine_shader_type(shader), shader))

        program = compileProgram(*loaded_shaders)
        cls.shader_cache[shaders] = program
        return program


def get_tensor_shader():
    """Returns the shader program for the Viewer to draw the texture to the screen."""
    return ShaderProgram.load_shader_program('TensorWindow.vert', 'TensorWindow.frag')


def get_line_shader():
    """Returns the shader program for the Viewer to draw lines, e.g. the wireframe unit cube."""
    return ShaderProgram.load_shader_program('Identity.vert',  # Vertex transformations done in geometry shader
                                             'LineThickness.geom',
                                             'DepthTest.frag')


def get_point_shader():
    """Returns the shader program for the Viewer to draw points."""
    return ShaderProgram.load_shader_program('Point.vert', 'Point.frag')


def get_frustum_shader():
    """Returns the shader program for the Viewer to draw camera frustums."""
    return ShaderProgram.load_shader_program('Frustum.vert', 'LineThickness.geom', 'DepthTest.frag')


__all__ = [
    ShaderProgram.__name__,
    get_tensor_shader.__name__,
    get_line_shader.__name__,
    get_point_shader.__name__,
    get_frustum_shader.__name__,
]
