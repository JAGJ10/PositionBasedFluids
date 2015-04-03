#version 400 core

in vec4 vertexPos;

uniform mat4 projection;
uniform mat4 mView;
uniform vec2 screenSize;
uniform float pointRadius;
uniform float pointScale;

out vec3 pos;

void main() {
	vec4 viewPos = mView * vec4(vertexPos.xyz, 1.0);
    gl_Position = projection * viewPos;
	pos = viewPos.xyz;
	gl_PointSize = pointScale * (pointRadius / gl_Position.w);
}