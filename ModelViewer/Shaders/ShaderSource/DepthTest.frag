#version 420

layout(location=0) in float depth;

out vec4 fragColor;

uniform float near;
uniform float far;
uniform vec2 viewportSize;
uniform vec4 color;

layout(binding=0) uniform sampler2D depthSampler;
layout(binding=1) uniform sampler2D alphaSampler;

void main() {
    vec2 uv = gl_FragCoord.xy / viewportSize;
    uv.y = 1.0 - uv.y;

    float scene_depth = texture(depthSampler, uv).r;
    float alpha = texture(alphaSampler, uv).r;

    if (scene_depth < near) { fragColor = color; return; }
    if (scene_depth > depth) { fragColor = color; return; }
    fragColor = vec4(color.rgb, color.a * (1.0 - alpha));
}
