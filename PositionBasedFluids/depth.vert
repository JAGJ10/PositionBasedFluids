#version 400 core

in vec3 vertexPos;

uniform mat4 projection;
uniform mat4 mView;
uniform vec2 screenSize;

out vec3 pos;
out float radius;

void main() {
	vec4 viewPos = mView * vec4(vertexPos, 1.0);
    float dist = length(viewPos);
    gl_Position = projection * viewPos;
	pos = viewPos.xyz;
	radius = 1.25 * viewPos.w * (600.0 / dist);
    gl_PointSize = radius;
}