import numpy as np
from OpenGL import GL as gl
from OpenGL.arrays import vbo


def get_cube_wireframe():
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