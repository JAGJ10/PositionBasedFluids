#version 400 core

in vec4 vertexPos;

uniform mat4 projection;
uniform mat4 mView;
uniform float pointRadius;
uniform float pointScale;

out vec3 pos;

void main() {
	vec4 viewPos = mView * vec4(vertexPos.xyz, 1.0);
    gl_Position = projection * viewPos;
	pos = viewPos.xyz;
	gl_PointSize = (pointScale / (((int(vertexPos.x * 1000)) % 5) + 1)) * (pointRadius / gl_Position.w);
}