#version 450

in vec3 vertex;

out vec2 uv;
out float depth;

uniform mat4 view;
uniform mat4 projection;
uniform float pointSize;


vec3 vertexPositions[] = {
    vec3(-1.0, -1.0, 0),
    vec3(-1.0,  1.0, 0),
    vec3( 1.0,  1.0, 0),
    vec3(-1.0, -1.0, 0),
    vec3( 1.0,  1.0, 0),
    vec3( 1.0, -1.0, 0)
};

vec2 uvs[] = {
    vertexPositions[0].xy,
    vertexPositions[1].xy,
    vertexPositions[2].xy,
    vertexPositions[3].xy,
    vertexPositions[4].xy,
    vertexPositions[5].xy
};


void main() {
    int vertexID = gl_VertexID % 6;

    gl_Position = view * vec4(vertex, 1.0);
    depth = -gl_Position.z;

    gl_Position = projection * (gl_Position + vec4(pointSize * vertexPositions[vertexID], 0.0));

    uv = uvs[vertexID];
}
