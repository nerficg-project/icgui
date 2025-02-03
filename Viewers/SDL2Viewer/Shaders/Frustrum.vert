#version 450

layout(location=0) in vec4 vertex;
layout(location=1) in mat4 c2w;

uniform float frustrumScale;
uniform int highlightedCamera;

out float highlighted;

void main() {
    int cameraID = gl_VertexID / 5;
    highlighted = cameraID == highlightedCamera ? 1.0 : 0.0;

    // View/Projection transformations done in geometry shader
    gl_Position = c2w * (vec4(frustrumScale, frustrumScale, frustrumScale, 1.0) * vertex);
}
