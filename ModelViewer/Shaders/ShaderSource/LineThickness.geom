#version 420
layout (lines) in;
layout (triangle_strip, max_vertices = 14) out;

in float highlighted[];

uniform float lineWidth;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

layout(location=0) out float depth;
layout(location=1) out float outHighlighted;


const int p_choice[14] = int[](0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1);
const float up_sign[14] = float[](1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, 1, 1);
const float rg_sign[14] = float[](-1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 1);

void main() {
    vec3 p[2] = vec3[](gl_in[0].gl_Position.xyz / gl_in[0].gl_Position.w,
                       gl_in[1].gl_Position.xyz / gl_in[1].gl_Position.w);

    vec3 fw = normalize(p[1] - p[0]);
    vec3 up = vec3(sign(fw.x + 1e6) * fw.z, sign(fw.y + 1e6) * fw.z, -sign(fw.z + 1e6) * (abs(fw.x) + abs(fw.y)));
         up = normalize(up - dot(up, fw) * fw) * lineWidth; // Orthogonalize
    vec3 rg = normalize(cross(fw, up)) * lineWidth;

    p[0] = p[0] - fw * lineWidth * 0.5;
    p[1] = p[1] + fw * lineWidth * 0.5;

    // Generate the triangle strip
    for (int i = 0; i < 14; i++) {
        vec4 view_position = viewMatrix * vec4(p[p_choice[i]] + up_sign[i] * up + rg_sign[i] * rg, 1.0);
        gl_Position = projectionMatrix * view_position;

        depth = -view_position.z;
        outHighlighted = highlighted[0] + highlighted[1];

        EmitVertex();
    }

    EndPrimitive();
}
