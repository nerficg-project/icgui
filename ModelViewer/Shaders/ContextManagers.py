"""ModelViewer/Shaders/ContextManagers.py: Context managers for binding and unbinding OpenGL resources
such as shader programs, textures, vertex buffers, and vertex attributes."""

from contextlib import contextmanager
from ctypes import c_void_p
from typing import Any

import OpenGL.GL as gl
from OpenGL.arrays import vbo
from OpenGL.GL.shaders import ShaderProgram as GLShaderProgram


@contextmanager
def bind_shader_program(shader_program: GLShaderProgram):
    """Context manager to bind a shader program."""
    gl.glUseProgram(shader_program)
    try:
        yield
    finally:
        gl.glUseProgram(0)


# For some reason, PyCharm marks the finally block as unreachable, but it is reachable.
# noinspection PyUnreachableCode
@contextmanager
def bind_textures(*texture_locations: int):
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
def bind_vertex_buffer(vao_location: int, *vbos: vbo.VBO):
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
def bind_vertex_attributes(size: int | list[int], dtype: Any | list[Any], stride: int | list[int] = 0,
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
