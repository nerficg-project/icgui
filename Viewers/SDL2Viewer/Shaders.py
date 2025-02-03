# -- coding: utf-8 --

"""SDL2Viewer/Shaders.py: Provides utility functions related to shaders for the SDL2Viewer."""

from contextlib import contextmanager
from ctypes import c_void_p
from pathlib import Path
from typing import Any

import numpy as np

import OpenGL.GL as gl
from OpenGL.arrays import vbo
from OpenGL.GL import GL_VERTEX_SHADER, GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER
from OpenGL.GL.shaders import compileProgram, compileShader, ShaderProgram as GLShaderProgram


class ShaderProgram:
    """Provides utility functions related to OpenGL shaders."""
    SHADER_ROOT = Path(__file__).parent / 'Shaders'
    ShaderType = type[GL_VERTEX_SHADER | GL_GEOMETRY_SHADER | GL_FRAGMENT_SHADER]
    ShaderPath = str | Path

    SHADER_EXTENSIONS = {
        'vert': GL_VERTEX_SHADER,
        'geom': GL_GEOMETRY_SHADER,
        'frag': GL_FRAGMENT_SHADER
    }

    shader_cache: dict[tuple[ShaderPath, ...], GLShaderProgram] = {}

    @classmethod
    def determineShaderType(cls, shader_path: ShaderPath):
        """Determines the shader type from the file extension."""
        shader_path = Path(shader_path)
        shader_extension = shader_path.suffix[1:]

        try:
            return cls.SHADER_EXTENSIONS[shader_extension]
        except KeyError as err:
            raise ValueError(f'Unknown shader extension "{shader_extension}" for shader "{shader_path}".') from err

    @classmethod
    def loadShaderFromFile(cls, shader_type: ShaderType, shader_path: ShaderPath):
        """Loads a shader from a file and returns the compiled shader."""
        with open(cls.SHADER_ROOT / shader_path, 'r', encoding='utf-8') as shader_file:
            shader_source = shader_file.read()
        return compileShader(shader_source, shader_type)

    @classmethod
    def loadShaderProgram(cls, *shaders: ShaderPath) -> GLShaderProgram:
        """Loads a shader program from a list of shaders and returns the compiled shader program."""
        if shaders in cls.shader_cache:
            return cls.shader_cache[shaders]

        loaded_shaders = []
        for shader in shaders:
            loaded_shaders.append(cls.loadShaderFromFile(cls.determineShaderType(shader), shader))

        program = compileProgram(*loaded_shaders)
        cls.shader_cache[shaders] = program
        return program


def getTensorShader():
    """Returns the shader program for the SDL2Viewer to draw the texture to the screen."""
    return ShaderProgram.loadShaderProgram('TensorWindow.vert', 'TensorWindow.frag')


def getLineShader():
    """Returns the shader program for the SDL2Viewer to draw lines, e.g. the wireframe unit cube."""
    return ShaderProgram.loadShaderProgram('Identity.vert',  # Vertex transformations done in geometry shader
                                           'LineThickness.geom',
                                           'DepthTest.frag')


def getPointShader():
    """Returns the shader program for the SDL2Viewer to draw points."""
    return ShaderProgram.loadShaderProgram('Point.vert', 'Point.frag')


def getFrustrumShader():
    """Returns the shader program for the SDL2Viewer to draw camera frustrums."""
    return ShaderProgram.loadShaderProgram('Frustrum.vert', 'LineThickness.geom', 'DepthTest.frag')


def getCubeWireframe():
    """Returns the vertices for a wireframe unit cube."""
    vertices = 0.5 * np.array([
        [-1.0, -1.0, -1.0],  # 0
        [-1.0, -1.0,  1.0],  # 1
        [-1.0,  1.0, -1.0],  # 2
        [-1.0,  1.0,  1.0],  # 3
        [ 1.0, -1.0, -1.0],  # 4
        [ 1.0, -1.0,  1.0],  # 5
        [ 1.0,  1.0, -1.0],  # 6
        [ 1.0,  1.0,  1.0],  # 7
    ], dtype=np.float32)
    vertex_positions = vbo.VBO(vertices)

    indices = np.array([
        [0, 1], [0, 2], [1, 3], [2, 3],  # bottom
        [4, 5], [4, 6], [5, 7], [6, 7],  # top
        [0, 4], [1, 5], [2, 6], [3, 7],  # sides
    ], dtype=np.uint32).flatten()
    index_positions = vbo.VBO(indices, target=gl.GL_ELEMENT_ARRAY_BUFFER)

    return vertex_positions, index_positions


# Context managers for binding and unbinding shader programs and components
@contextmanager
def bindShaderProgram(shader_program: GLShaderProgram):
    """Context manager to bind a shader program."""
    gl.glUseProgram(shader_program)
    try:
        yield
    finally:
        gl.glUseProgram(0)


@contextmanager
def bindTextures(*texture_locations: int):
    """Context manager to bind a list of textures."""
    try:
        for i, texture_location in enumerate(texture_locations):
            gl.glActiveTexture(gl.GL_TEXTURE0 + i)
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture_location)
        yield
    finally:
        # Iterate in reverse order to ensure we leave with Texture0 active
        for i in range(len(texture_locations) - 1, -1, -1):
            gl.glActiveTexture(gl.GL_TEXTURE0 + i)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)


@contextmanager
def bindVertexBuffer(vao_location: int, *vbos: vbo.VBO):
    """Context manager to bind vertex buffer objects."""
    gl.glBindVertexArray(vao_location)
    try:
        for buffer in vbos:
            buffer.bind()
        yield
    finally:
        for buffer in vbos:
            buffer.unbind()
        gl.glBindVertexArray(0)


@contextmanager
def bindVertexAttributes(size: int | list[int], dtype: Any | list[Any], stride: int | list[int] = 0,
                         offset: list[int] = None):
    """Context manager to bind and configure vertex attributes."""
    if isinstance(size, list):
        if not isinstance(dtype, list):
            dtype = [dtype] * len(size)
        if isinstance(stride, int):
            stride = [stride] * len(size)
        if offset is None:
            offset = [0] * len(size)

        assert len(size) == len(dtype) == len(stride), 'Size, dtype, and stride must have the same length.'

    try:
        if isinstance(size, list):
            # gl.glEnableVertexAttribArray(0)
            # gl.glVertexAttribPointer(0, size[0], dtype[0], False, stride[0], None)
            for i, (s, d, st, o) in enumerate(zip(size, dtype, stride, offset)):
                gl.glEnableVertexAttribArray(i)
                gl.glVertexAttribPointer(i, s, d, False, st, c_void_p(o))
                gl.glVertexAttribDivisor(i, 0)
        else:
            gl.glEnableVertexAttribArray(0)
            gl.glVertexAttribPointer(0, size, dtype, False, stride, None)

        yield
    finally:
        if isinstance(size, list):
            for i in range(len(size)):
                gl.glDisableVertexAttribArray(i)
        else:
            gl.glDisableVertexAttribArray(0)


def fillUniforms(uniforms: dict[str, tuple[int, str, tuple]], values: dict[str, Any]):
    """Fills the given uniforms with the given values."""
    store_funcs = {
        'float': 'glUniform1fv',
        'vec2': 'glUniform2fv',
        'vec3': 'glUniform3fv',
        'vec4': 'glUniform4fv',
        'int': 'glUniform1iv',
        'ivec2': 'glUniform2iv',
        'ivec3': 'glUniform3iv',
        'ivec4': 'glUniform4iv',
        'mat2': 'glUniformMatrix2fv',
        'mat3': 'glUniformMatrix3fv',
        'mat4': 'glUniformMatrix4fv',
    }

    for uniform_name, (location, dtype, args) in uniforms.items():
        if uniform_name not in values:
            continue
        value = values[uniform_name]
        func = getattr(gl, store_funcs[dtype])

        # gl uniform functions don't accept their arguments as kwargs
        func(location, 1, *args, value)
