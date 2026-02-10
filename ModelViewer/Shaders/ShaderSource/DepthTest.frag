#version 420

layout(location=0) in float depth;
layout(location=1) in float highlighted;

out vec4 fragColor;

uniform float near;
uniform float far;
uniform vec2 viewportSize;
uniform vec4 color;
uniform vec4 highlightColor;

layout(binding=0) uniform sampler2D depthSampler;
layout(binding=1) uniform sampler2D alphaSampler;

void main() {
    vec4 col = color;
    if (highlighted > 0.5) { col = highlightColor; }

    vec2 uv = gl_FragCoord.xy / viewportSize;
    uv.y = 1.0 - uv.y;

    float scene_depth = texture(depthSampler, uv).r;
    float alpha = texture(alphaSampler, uv).r;

    if (scene_depth < near) { fragColor = col; return; }
    if (scene_depth > depth) { fragColor = col; return; }
    fragColor = vec4(col.rgb, col.a * (1.0 - alpha));
}
