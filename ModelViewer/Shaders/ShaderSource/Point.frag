#version 450

in vec2 uv;
in float depth;

uniform float near;
uniform float far;
uniform vec2 viewportSize;
uniform vec4 color;

layout(binding=0) uniform sampler2D depthSampler;
layout(binding=1) uniform sampler2D alphaSampler;

out vec4 fragColor;

void main() {
    if (uv.x * uv.x + uv.y * uv.y > 1.0) {
        discard;
    }

    vec2 screenUv = gl_FragCoord.xy / viewportSize;
    screenUv.y = 1.0 - screenUv.y;

    float scene_depth = texture(depthSampler, screenUv).r;
    float alpha = texture(alphaSampler, screenUv).r;

    if (scene_depth < near) { fragColor = color; return; }
    if (scene_depth > depth) { fragColor = color; return; }
    fragColor = vec4(color.rgb, color.a * (1.0 - alpha));
}
