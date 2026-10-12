#version 450

layout(location=0) in vec4 vertex;
layout(location=1) in mat4 c2w;

uniform float frustumScale;
uniform vec2 focalAspectScale;


void main() {
    // View/Projection transformations done in geometry shader
    gl_Position = c2w * (vec4(frustumScale * focalAspectScale.x, frustumScale * focalAspectScale.y, frustumScale, 1.0) * vertex);
}
