#version 450

in vec3 vertex;

void main() {
    // Vertex transformations done in geometry shader
    gl_Position = vec4(vertex, 1.0);
}
