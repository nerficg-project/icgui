#version 450

smooth out vec2 texcoords;

// Span a triangle with 2 vertices outside the viewport to fully cover the viewport
vec4 positions[3] = vec4[3](
    vec4(-1.0, 1.0, 0.0, 1.0),
    vec4(3.0, 1.0, 0.0, 1.0),
    vec4(-1.0, -3.0, 0.0, 1.0)
);

// Texture coordinates in the viewport correspond to 0, 1 range, since 2 triangle vertices are outside the viewport
vec2 texpos[3] = vec2[3](
    vec2(0, 0),
    vec2(2, 0),
    vec2(0, 2)
);

void main() {
    // Simply output the constant vertices defined in this shader, regardless of inputs
    gl_Position = positions[gl_VertexID];
    texcoords = texpos[gl_VertexID];
}
