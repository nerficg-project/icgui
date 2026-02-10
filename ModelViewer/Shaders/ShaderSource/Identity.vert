#version 450

in vec3 vertex;
out float highlighted;

void main() {
    // Vertex transformations done in geometry shader
    gl_Position = vec4(vertex, 1.0);
    highlighted = 0.0;
}
