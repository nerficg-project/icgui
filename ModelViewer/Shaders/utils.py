"""ModelViewer/Shaders/utils.py: Utility functions related to Shaders."""

from typing import Any

import OpenGL.GL as gl


_UNIFORM_STORE_NAMES = {
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


def fill_uniforms(uniforms: dict[str, tuple[int, str, tuple]], values: dict[str, Any]):
    """Fills the given uniforms with the given values."""
    for uniform_name, (location, dtype, args) in uniforms.items():
        if uniform_name not in values:
            continue
        value = values[uniform_name]
        func = getattr(gl, _UNIFORM_STORE_NAMES[dtype])

        # gl uniform functions don't accept their arguments as kwargs
        func(location, 1, *args, value)
