#version 400 core

in vec4 vertexPos;

uniform mat4 projection;
uniform mat4 mView;
uniform vec2 screenSize;
uniform float pointRadius;
uniform float pointScale;

out vec4 pos;
out float lifetime;

void main() {
	vec4 viewPos = mView * vec4(vertexPos.xyz, 1.0);
    gl_Position = projection * viewPos;
	pos = viewPos;
	float ri = pointRadius / ((int(vertexPos.w) % 5) + 1);
	gl_PointSize = pointScale * (ri / gl_Position.w);
	lifetime = vertexPos.w - int(vertexPos.w);
}